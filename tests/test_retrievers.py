# Tests for the retrieval seam (rag/retrievers.py) and how build_prompts.py
# names and configures a run (docs/AZURE-PLAN.md, Phase 1).
#
# The parity gate -- rebuild rag_prompts.json and friends and compare bytes --
# is the real test of the seam, and it needs the embedding model, so it lives
# in the plan's checklist rather than here. These pin the two things that can
# break without the gate noticing: an existing FAISS filename being renamed
# (other scripts and committed artifacts depend on every one of them), and
# FaissRetriever returning something other than rows + cosines in rank order.
#
#   python -m pytest tests/ -q

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rag"))

# IMPORT ORDER: build_prompts pulls in embed_utils (torch) before anything
# constructs a FaissRetriever (faiss). See rag/build_prompts.py.
from build_prompts import output_name, parse_args  # noqa: E402
from retrievers import FaissRetriever, load_azure_settings, retriever_label  # noqa: E402


# Every invocation that has produced a committed rag/data/rag_prompts*.json,
# with the name it must keep.
@pytest.mark.parametrize("argv, name", [
    ([], "rag_prompts.json"),
    (["10"], "rag_prompts.json"),
    (["3"], "rag_prompts_k3.json"),
    (["5"], "rag_prompts_k5.json"),
    (["10", "nofk"], "rag_prompts_nofk.json"),
    (["--db-policy", "rank1", "10"], "rag_prompts_rank1_k10.json"),
    (["--rerank", "f.json", "rerankQ", "10"], "rag_prompts_rerankQ_k10.json"),
    (["--rerank", "f.json", "rerankQQ", "5"], "rag_prompts_rerankQQ_k5.json"),
    (["--retriever", "faiss", "10"], "rag_prompts.json"),
])
def test_faiss_filenames_unchanged(argv, name):
    assert output_name(parse_args(argv)) == name


@pytest.mark.parametrize("argv, name", [
    (["--retriever", "azure-vector", "10"], "rag_prompts_azvec_k10.json"),
    (["--retriever", "azure-hybrid", "10"], "rag_prompts_azhyb_k10.json"),
    (["--retriever", "azure-semantic", "10"], "rag_prompts_azsem_k10.json"),
    (["--retriever", "azure-hybrid", "--db-policy", "rank1", "10"], "rag_prompts_rank1_azhyb_k10.json"),
    (["--retriever", "azure-vector", "10", "nofk"], "rag_prompts_azvec_nofk_k10.json"),
])
def test_azure_filenames(argv, name):
    assert output_name(parse_args(argv)) == name


def test_flags_combine_in_any_order():
    cfg = parse_args(["--retriever", "azure-hybrid", "5", "--db-policy", "rank1"])
    assert (cfg["retriever"], cfg["db_policy"], cfg["top_k"], cfg["include_fk"]) == \
        ("azure-hybrid", "rank1", 5, True)


def test_rerank_with_non_faiss_retriever_refused():
    # A rerank file re-orders FAISS candidates; combining it with another
    # retriever would change two variables at once.
    with pytest.raises(SystemExit):
        parse_args(["--retriever", "azure-vector", "--rerank", "f.json", "x", "10"])


def test_unknown_retriever_refused():
    with pytest.raises(SystemExit):
        parse_args(["--retriever", "pinecone", "10"])
    with pytest.raises(ValueError):
        retriever_label("pinecone")


def test_azure_settings_missing_is_a_clear_error(tmp_path, monkeypatch):
    for k in ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_QUERY_KEY", "AZURE_SEARCH_ADMIN_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="missing"):
        load_azure_settings(env_file=tmp_path / "absent.env")


def test_azure_settings_environment_wins_over_file(tmp_path, monkeypatch):
    env = tmp_path / "azure.env"
    env.write_text("AZURE_SEARCH_ENDPOINT=https://file\nAZURE_SEARCH_QUERY_KEY=filekey\n")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://env")
    monkeypatch.delenv("AZURE_SEARCH_QUERY_KEY", raising=False)
    cfg = load_azure_settings(env_file=env)
    assert (cfg["endpoint"], cfg["query_key"], cfg["index"]) == ("https://env", "filekey", "fewshot-exemplars")


def test_faiss_retriever_returns_rows_and_cosines_in_rank_order():
    # No embedding model needed: a stored vector searched against its own
    # index must come back as its own row at cosine ~1.
    r = FaissRetriever()
    vecs = r.index.reconstruct_n(0, r.index.ntotal)
    hits = r.search("unused by FAISS", vecs[42:43], 10)
    assert len(hits) == 10
    assert all(isinstance(row, int) and isinstance(s, float) for row, s in hits)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert vecs[hits[0][0]] @ vecs[42] == pytest.approx(1.0, abs=1e-5)  # row 42 or an exact duplicate
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert r.describe()["faiss_index_type"] == "IndexFlatIP"
