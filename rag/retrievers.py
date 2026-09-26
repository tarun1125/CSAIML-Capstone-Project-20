# The retrieval seam (docs/AZURE-PLAN.md, Phase 1).
#
# Everything downstream of "which exemplars, in what order" -- the database
# vote, the schema block, the prompt text, generation, scoring -- is identical
# whichever retriever produced the list. This module is the one place that
# differs, so the Azure experiment can swap FAISS for Azure AI Search and
# change NOTHING else. That is the whole point of the Colab->MLX lesson: one
# variable at a time.
#
# A retriever returns ROW INDICES into rag/data/fewshot_metadata.json, never
# exemplar ids. The ids are a mix of ints and strings (12 and "spider-car_1-44"
# both occur), and an Azure Search key is always a string, so an id that goes
# in as 12 comes back as "12" and silently misses a lookup keyed on 12. A row
# index has one representation everywhere.
#
# IMPORT ORDER: faiss is imported inside FaissRetriever.__init__, never at
# module level. Every caller imports embed_utils (torch) at its own module
# top, so by the time a FaissRetriever is constructed torch's libomp has
# already won. See the long note in rag/build_prompts.py for why that matters.

import logging
import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

# The Azure SDK logs every HTTP request and response at INFO, which buries the
# per-case log lines of build_prompts.py / eval_retriever.py. Keys are redacted
# either way; this is about noise, not secrecy.
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

RAG_DATA = Path(__file__).resolve().parent / "data"
REPO_ROOT = RAG_DATA.parents[1]
DEFAULT_FAISS_INDEX = RAG_DATA / "fewshot.index"
AZURE_ENV_FILE = REPO_ROOT / "azure.env"

# Shared with rag/azure_index.py, so the index and its reader cannot disagree.
AZURE_VECTOR_FIELD = "question_vector"
SEMANTIC_CONFIG_NAME = "default"
# Hybrid: how many neighbours the VECTOR leg contributes to Reciprocal Rank
# Fusion before the fused list is cut to k. Fixed at the FAISS candidate depth
# the rest of this repo uses (rerank.DEFAULT_TOP_N), not tied to k, so K=5 and
# K=10 arms fuse the same candidate pool and differ only in where they cut.
HYBRID_VECTOR_K = 50

# CLI name -> (Azure query mode, filename label). The label goes into output
# filenames (rag_prompts_azvec_k10.json); FAISS has none, so every existing
# filename stays exactly what it was.
AZURE_MODES = {
    "azure-vector": ("vector", "azvec"),
    "azure-hybrid": ("hybrid", "azhyb"),
    "azure-semantic": ("semantic", "azsem"),
}
RETRIEVERS = ("faiss", *AZURE_MODES)


class Retriever(Protocol):
    name: str
    label: str        # "" for faiss; used in output filenames
    score_kind: str   # what retrieved_scores means for this retriever

    def search(self, question: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (row, score) pairs, best first. `question` is for retrievers
        with a lexical leg (BM25); `qvec` is embed_utils.embed([question])."""
        ...

    def describe(self) -> dict:
        """Provenance for write_manifest()."""
        ...


class FaissRetriever:
    name = "faiss"
    label = ""
    score_kind = "cosine similarity (IndexFlatIP over L2-normalised MiniLM vectors)"

    def __init__(self, index_path: Path = DEFAULT_FAISS_INDEX):
        import faiss  # noqa: PLC0415  deliberately late -- see the module note

        self.index_path = Path(index_path)
        self.index = faiss.read_index(str(self.index_path))

    def search(self, question: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        # Exactly the call build_prompts.py and rerank.py used to make inline.
        scores, idxs = self.index.search(qvec, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0])]

    def describe(self) -> dict:
        try:
            index_rel = str(self.index_path.relative_to(RAG_DATA.parents[1]))
        except ValueError:
            index_rel = str(self.index_path)
        return {
            "retriever": self.name,
            "score_kind": self.score_kind,
            "faiss_index": index_rel,
            "faiss_index_ntotal": self.index.ntotal,
            "faiss_index_type": type(self.index).__name__,
        }


def load_azure_settings(require_admin: bool = False, env_file: Path = AZURE_ENV_FILE,
                        allow_identity: bool = False) -> dict:
    """Search endpoint, index name and keys: process environment first (the
    deployed container gets them that way), then the gitignored azure.env.
    Never logged -- describe() reports the endpoint and index, not keys."""
    values = {}
    if env_file.exists():
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from atlas_env import load_env_file  # noqa: PLC0415  repo-root helper, reused

        values = load_env_file(env_file)
    get = lambda k: os.environ.get(k) or values.get(k)  # noqa: E731
    cfg = {
        "endpoint": get("AZURE_SEARCH_ENDPOINT"),
        "index": get("AZURE_SEARCH_INDEX") or "fewshot-exemplars",
        "query_key": get("AZURE_SEARCH_QUERY_KEY"),
        "admin_key": get("AZURE_SEARCH_ADMIN_KEY"),
    }
    # allow_identity: the deployed service has no key and authenticates with
    # its managed identity (role "Search Index Data Reader") instead.
    needed = ["endpoint"] + (["admin_key"] if require_admin else [] if allow_identity else ["query_key"])
    missing = [k for k in needed if not cfg[k]]
    if missing:
        raise RuntimeError(
            f"Azure Search settings missing: {missing}. Set AZURE_SEARCH_* in the environment "
            f"or in {env_file.name} (see docs/AZURE-PLAN.md, Phase 2)."
        )
    return cfg


class AzureSearchRetriever:
    """Azure AI Search over the same 1,213 exemplars and the same MiniLM vectors
    (rag/azure_index.py builds the index). Uses the READ-ONLY query key.

    Modes -- the arms of the experiment:
      vector    exhaustive-KNN cosine on the question vector only. A1, the
                parity control: should reproduce FAISS's lists.
      hybrid    BM25 over `question` + the same vector leg, fused by RRF. A2.
      semantic  hybrid, re-ordered by Azure's semantic ranker. A3 (Basic tier+).

    Scores are whatever Azure returns as @search.score and are NOT cosine:
    for vector search they are a monotone transform of cosine distance, for
    hybrid an RRF score, for semantic the ranker's score. score_kind says
    which, and it is recorded in every manifest."""

    SCORE_KINDS = {
        # Measured 2026-09-24 against FAISS on the same vectors, equal to 6 d.p.:
        # score = 1 / (2 - cosine), i.e. cosine = 2 - 1/score. Monotone, so the
        # ORDER is cosine order; the VALUE is not a cosine.
        "vector": "azure @search.score for exhaustive-KNN cosine = 1/(2 - cosine)",
        "hybrid": "azure @search.score, Reciprocal Rank Fusion of BM25 + vector",
        "semantic": "azure @search.rerankerScore from the semantic ranker",
    }

    def __init__(self, mode: str, settings: dict | None = None):
        names = {m: (n, lbl) for n, (m, lbl) in AZURE_MODES.items()}
        if mode not in names:
            raise ValueError(f"unknown Azure mode {mode!r}")
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
        from azure.search.documents import SearchClient  # noqa: PLC0415

        self.mode = mode
        self.name, self.label = names[mode]
        self.score_kind = self.SCORE_KINDS[mode]
        self.settings = settings or load_azure_settings()
        if self.settings.get("query_key"):
            credential, self.auth_mode = AzureKeyCredential(self.settings["query_key"]), "query_key"
        else:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            credential, self.auth_mode = DefaultAzureCredential(), "entra_id"
        self.client = SearchClient(self.settings["endpoint"], self.settings["index"], credential)

    def search(self, question: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        from azure.search.documents.models import VectorizedQuery  # noqa: PLC0415

        vector_k = k if self.mode == "vector" else max(k, HYBRID_VECTOR_K)
        vq = VectorizedQuery(vector=qvec[0].tolist(), k_nearest_neighbors=vector_k,
                             fields=AZURE_VECTOR_FIELD, exhaustive=True)
        kwargs = {"vector_queries": [vq], "top": k, "select": ["id"]}
        if self.mode in ("hybrid", "semantic"):
            kwargs.update(search_text=question, search_fields=["question"])
        if self.mode == "semantic":
            kwargs.update(query_type="semantic", semantic_configuration_name=SEMANTIC_CONFIG_NAME)
        score_key = "@search.rerankerScore" if self.mode == "semantic" else "@search.score"
        hits = [(int(r["id"]), float(r[score_key])) for r in self.client.search(**kwargs)]
        if len(hits) != k:
            raise RuntimeError(f"Azure returned {len(hits)} hits for k={k} (mode={self.mode})")
        return hits

    def describe(self) -> dict:
        import azure.search.documents as sdk  # noqa: PLC0415

        return {
            "retriever": self.name,
            "score_kind": self.score_kind,
            "azure_mode": self.mode,
            "azure_endpoint": self.settings["endpoint"],
            "azure_index": self.settings["index"],
            "azure_vector_field": AZURE_VECTOR_FIELD,
            "azure_exhaustive_knn": True,
            "azure_hybrid_vector_k": HYBRID_VECTOR_K if self.mode != "vector" else None,
            "azure_sdk_version": sdk.__version__,
            "azure_auth": self.auth_mode,
        }


def make_retriever(name: str) -> Retriever:
    if name == "faiss":
        return FaissRetriever()
    if name in AZURE_MODES:
        return AzureSearchRetriever(AZURE_MODES[name][0])
    raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVERS}")


def retriever_label(name: str) -> str:
    """Filename label without constructing the retriever (no index, no network)."""
    if name == "faiss":
        return ""
    if name in AZURE_MODES:
        return AZURE_MODES[name][1]
    raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVERS}")
