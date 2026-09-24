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

from pathlib import Path
from typing import Protocol

import numpy as np

RAG_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_FAISS_INDEX = RAG_DATA / "fewshot.index"

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


class AzureSearchRetriever:
    """Azure AI Search over the same 1,213 exemplars and the same MiniLM vectors.
    Built in Phase 2, once the index exists (rag/azure_index.py)."""

    def __init__(self, mode: str):
        if mode not in {m for m, _ in AZURE_MODES.values()}:
            raise ValueError(f"unknown Azure mode {mode!r}")
        raise NotImplementedError(
            f"AzureSearchRetriever(mode={mode!r}) is Phase 2 of docs/AZURE-PLAN.md -- "
            "create the search service and run rag/azure_index.py first."
        )


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
