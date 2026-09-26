# Builds the Azure AI Search index for the retrieval experiment
# (docs/AZURE-PLAN.md, Phase 2): the same 1,213 few-shot exemplars FAISS
# searches, with the same vectors.
#
#   python rag/azure_index.py              # create/update the index, upload, verify
#   python rag/azure_index.py --recreate   # delete and rebuild from scratch
#   python rag/azure_index.py --semantic   # also add a semantic config (Basic tier+, arm A3)
#
# Reads AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_INDEX / AZURE_SEARCH_ADMIN_KEY from
# the environment or the gitignored azure.env at the repo root.
#
# DESIGN CHOICES, each one to hold everything but the search engine fixed:
#
# * THE VECTORS ARE FAISS'S OWN. They are read back out of rag/data/fewshot.index
#   (reconstruct_n), not re-embedded. Re-embedding would very probably give the
#   same floats, but "very probably" is a second variable in the A1 parity check;
#   reading the index makes it impossible.
#
# * EXHAUSTIVE KNN, COSINE -- not HNSW. FAISS here is IndexFlatIP, i.e. exact;
#   HNSW is approximate and would add ANN recall error on top of the engine
#   change. At 1,213 vectors, exact search costs nothing. (Exhaustive KNN is
#   also documented as not consuming the tier's vector quota.) The vectors are
#   L2-normalised, so cosine ranks identically to FAISS's inner product.
#
# * KEY = ROW INDEX, as a string. The exemplar ids mix ints and strings, and
#   an Azure key is always a string; the row index maps a hit straight back to
#   fewshot_metadata.json with one representation. The original id is kept in
#   `exemplar_id` for humans.
#
# * BM25 SEES ONLY `question`, exactly as FAISS embeds only the question. The
#   exemplar's query text is retrievable but not searchable.

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from retrievers import (
    AZURE_VECTOR_FIELD,
    SEMANTIC_CONFIG_NAME,
    FaissRetriever,
    load_azure_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from run_manifest import write_manifest  # noqa: E402

RAG_DATA = REPO_ROOT / "rag" / "data"
RESULTS_DIR = REPO_ROOT / "results"
DIMENSIONS = 384
UPLOAD_BATCH = 500

log = logging.getLogger("rag.azure_index")


def index_definition(name: str, semantic: bool) -> SearchIndex:
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="exemplar_id", type=SearchFieldDataType.String),
        SearchField(name="question", type=SearchFieldDataType.String, searchable=True,
                    analyzer_name="en.microsoft"),
        SearchField(name="database", type=SearchFieldDataType.String, filterable=True,
                    facetable=True),
        SearchField(name="complexity", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="normalized_query", type=SearchFieldDataType.String, searchable=False),
        SearchField(
            name=AZURE_VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,              # required for a vector field to be queryable
            retrievable=False,            # 384 floats per hit is dead weight in responses
            vector_search_dimensions=DIMENSIONS,
            vector_search_profile_name="exact",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[ExhaustiveKnnAlgorithmConfiguration(
            name="exhaustive-cosine",
            parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
        )],
        profiles=[VectorSearchProfile(name="exact", algorithm_configuration_name="exhaustive-cosine")],
    )
    semantic_search = None
    if semantic:
        semantic_search = SemanticSearch(configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG_NAME,
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="question")]),
        )])
    return SearchIndex(name=name, fields=fields, vector_search=vector_search,
                       semantic_search=semantic_search)


def documents() -> list[dict]:
    metadata = json.loads((RAG_DATA / "fewshot_metadata.json").read_text(encoding="utf-8"))
    faiss = FaissRetriever()
    if faiss.index.ntotal != len(metadata):
        raise SystemExit(f"fewshot.index has {faiss.index.ntotal} rows but metadata has "
                         f"{len(metadata)} -- rebuild with rag/build_retrieval_index.py first")
    vecs = faiss.index.reconstruct_n(0, faiss.index.ntotal)
    return [
        {
            "id": str(row),
            "exemplar_id": str(m["id"]),
            "question": m["question"],
            "database": m["database"],
            "complexity": m.get("complexity"),
            "normalized_query": m["normalized_query"],
            # float32 -> Python float is exact, and Edm.Single parses it back to
            # the identical float32.
            AZURE_VECTOR_FIELD: vecs[row].tolist(),
        }
        for row, m in enumerate(metadata)
    ]


def wait_for_count(client: SearchClient, expected: int, timeout_s: float = 60) -> int:
    """Uploads are indexed asynchronously; the count lags by a few seconds."""
    deadline = time.monotonic() + timeout_s
    count = client.get_document_count()
    while count != expected and time.monotonic() < deadline:
        time.sleep(2)
        count = client.get_document_count()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="delete the index first")
    parser.add_argument("--semantic", action="store_true",
                        help="add a semantic configuration (needs Basic tier or above)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_azure_settings(require_admin=True)
    credential = AzureKeyCredential(cfg["admin_key"])
    index_client = SearchIndexClient(cfg["endpoint"], credential)

    if args.recreate:
        try:
            index_client.delete_index(cfg["index"])
            log.info("deleted index %s", cfg["index"])
        except ResourceNotFoundError:
            pass

    definition = index_definition(cfg["index"], args.semantic)
    created = index_client.create_or_update_index(definition)
    log.info("index %s ready on %s", created.name, cfg["endpoint"])

    docs = documents()
    search_client = SearchClient(cfg["endpoint"], cfg["index"], credential)
    for start in range(0, len(docs), UPLOAD_BATCH):
        batch = docs[start:start + UPLOAD_BATCH]
        results = search_client.merge_or_upload_documents(batch)
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise SystemExit(f"{len(failed)} documents failed to upload, e.g. {failed[:5]}")
        log.info("uploaded %d-%d", start, start + len(batch) - 1)

    count = wait_for_count(search_client, len(docs))
    if count != len(docs):
        raise SystemExit(f"index reports {count} documents, expected {len(docs)}")
    log.info("VERIFY document count: %d == %d", count, len(docs))

    # Keep the definition: the service is deleted at teardown, this file is not.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "azure_index_definition.json"
    out.write_text(json.dumps(created.as_dict(), indent=2, default=str), encoding="utf-8")
    write_manifest(out, script=__file__, stage="azure index build", index=cfg["index"],
                   endpoint=cfg["endpoint"], n_documents=count, semantic=args.semantic,
                   vectors_from="rag/data/fewshot.index (reconstruct_n)")
    log.info("wrote %s", out)


if __name__ == "__main__":
    sys.exit(main())
