# Ranking metrics and the A1 parity check for ONE retriever
# (docs/AZURE-PLAN.md, Phase 3, Gates 1 and 2). No generation, no Atlas.
#
#   python rag/eval_retriever.py --retriever faiss          # positive control
#   python rag/eval_retriever.py --retriever azure-vector   # A1: Gate 1 + Gate 2
#   python rag/eval_retriever.py --retriever azure-hybrid   # A2: Gate 2
#
# Writes results/retrieval_eval_<retriever>.json (+ manifest).
#
# WHY NOT rag/eval_retrieval.py --retriever: that script IS the reranking
# experiment. Its cross-encoder cache is keyed on FAISS's top-50 candidates, and
# its sanity check is pinned to the FAISS run's recorded vote accuracy, so an
# Azure candidate list would miss the cache and fail the check. This script
# reuses its grading (grade_ranking) and the shared metrics (summarize) instead,
# so the numbers come from the same code -- only the candidate list differs.
#
# THE FAISS RUN IS NOT OPTIONAL. Before believing anything this script says
# about Azure, run it with --retriever faiss: it must reproduce the bi_encoder
# row of results/retrieval_eval.json EXACTLY (it refuses to write otherwise).
# A harness that cannot reproduce a known result cannot measure a new one --
# the same job A1 does one level up.
#
# GATE 1 (parity), for any non-FAISS retriever: the top-10 row list per case,
# compared with FAISS's. Exact KNN on the same vectors should agree everywhere
# except exact score ties; for each mismatch the report carries both lists
# and the true cosine of every row in them, which is what separates a tie
# (equal cosines, swapped order) from a bug (a different set).

import argparse
import json
import logging
import sys
from pathlib import Path

# IMPORT ORDER IS LOAD-BEARING -- torch (via rerank/embed_utils) BEFORE faiss.
# See rag/build_prompts.py. faiss itself only loads inside FaissRetriever.
from rerank import DEFAULT_TOP_N, build_candidates  # noqa: E402

from build_prompts import majority_vote_database  # noqa: E402
from embed_utils import MODEL_NAME as EMBED_MODEL_NAME  # noqa: E402
from embed_utils import embed  # noqa: E402
from eval_retrieval import grade_ranking  # noqa: E402
from metrics_retrieval import summarize  # noqa: E402
from retrievers import RETRIEVERS, FaissRetriever, make_retriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from run_manifest import write_manifest  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PUBLISHED = RESULTS_DIR / "retrieval_eval.json"
PARITY_K = 10

log = logging.getLogger("rag.eval_retriever")


def ranking_metrics(test_cases, metadata, rows_per_case) -> dict:
    binary, graded = [], []
    for case, rows in zip(test_cases, rows_per_case):
        b, g = grade_ranking(rows, metadata, case)
        binary.append(b)
        graded.append(g)
    out = summarize(binary, graded)
    n = len(test_cases)
    out["db_vote_accuracy"] = {
        str(k): sum(
            majority_vote_database([metadata[r]["database"] for r in rows[:k]]) == case["database"]
            for case, rows in zip(test_cases, rows_per_case)
        ) / n
        for k in (5, 10)
    }
    # The rank-1 policy (docs/FINDING-rank1-vote.md) -- the other database
    # rule the Azure arms are run under.
    out["db_rank1_accuracy"] = sum(
        metadata[rows[0]]["database"] == case["database"]
        for case, rows in zip(test_cases, rows_per_case)
    ) / n
    return out


def positive_control(metrics: dict) -> dict:
    """FAISS through this script vs the published bi_encoder row."""
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))["arms"]["bi_encoder"]
    keys = [k for k in published if k != "per_case"]
    mismatched = [k for k in keys if metrics.get(k) != published[k]]
    return {"compared_to": str(PUBLISHED.relative_to(REPO_ROOT)), "fields": keys,
            "mismatched_fields": mismatched, "passed": not mismatched}


def parity(test_cases, metadata, arm_rows, faiss_rows, k: int = PARITY_K) -> dict:
    """Gate 1: does this retriever return FAISS's top-k, in FAISS's order?"""
    vecs = FaissRetriever().index.reconstruct_n(0, len(metadata))
    identical = same_set = 0
    cases = []
    for case, a, f in zip(test_cases, arm_rows, faiss_rows):
        a, f = a[:k], f[:k]
        if a == f:
            identical += 1
            continue
        if set(a) == set(f):
            same_set += 1
        qvec = embed([case["question"]])[0]
        cos = {r: float(vecs[r] @ qvec) for r in set(a) | set(f)}
        first = next(i for i, (x, y) in enumerate(zip(a, f)) if x != y)
        cases.append({
            "id": case["id"],
            "same_set": set(a) == set(f),
            "first_divergence_rank": first + 1,
            "arm_rows": a,
            "faiss_rows": f,
            "arm_cosines": [cos[r] for r in a],
            "faiss_cosines": [cos[r] for r in f],
            # Equal cosines at the divergence point is a tie, not a bug.
            "tie_at_divergence": abs(cos[a[first]] - cos[f[first]]) < 1e-6,
        })
    n = len(test_cases)
    return {
        "k": k,
        "n_cases": n,
        "n_identical": identical,
        "n_same_set_diff_order": same_set,
        "n_diff_set": n - identical - same_set,
        "n_mismatch_explained_by_tie": sum(c["tie_at_divergence"] for c in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", required=True, choices=RETRIEVERS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="candidate depth; 50 matches results/retrieval_eval.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    retriever = make_retriever(args.retriever)
    test_cases, metadata, rows, scores = build_candidates(args.top_n, retriever)
    metrics = ranking_metrics(test_cases, metadata, rows)
    log.info("%s: recall@5=%.4f MRR@10=%.4f nDCG@10=%.4f dbvote@10=%.4f dbrank1=%.4f",
             args.retriever, metrics["recall_at_k"]["5"], metrics["mrr_at_10"],
             metrics["ndcg_at_10"], metrics["db_vote_accuracy"]["10"], metrics["db_rank1_accuracy"])

    payload = {
        "config": {
            **retriever.describe(),
            "top_n_candidates": args.top_n,
            "n_cases": len(test_cases),
            "pool_size": len(metadata),
            "embedding_model": EMBED_MODEL_NAME,
        },
        "metrics": metrics,
        "per_case_rows": {str(c["id"]): r for c, r in zip(test_cases, rows)},
        "per_case_scores": {str(c["id"]): s for c, s in zip(test_cases, scores)},
    }

    if args.retriever == "faiss":
        control = positive_control(metrics)
        payload["positive_control"] = control
        if not control["passed"]:
            raise SystemExit(
                f"POSITIVE CONTROL FAILED: {control['mismatched_fields']} differ from "
                f"{PUBLISHED.name}. This harness does not reproduce the published FAISS "
                "numbers, so nothing it says about another retriever means anything yet."
            )
        log.info("positive control PASSED: all %d fields match %s exactly",
                 len(control["fields"]), PUBLISHED.name)
    else:
        _, _, faiss_rows, _ = build_candidates(args.top_n, FaissRetriever())
        payload["parity_vs_faiss"] = gate = parity(test_cases, metadata, rows, faiss_rows)
        log.info("GATE 1 (top-%d vs FAISS): %d/%d identical, %d same set/different order, "
                 "%d different set; %d mismatches sit on an exact cosine tie",
                 gate["k"], gate["n_identical"], gate["n_cases"], gate["n_same_set_diff_order"],
                 gate["n_diff_set"], gate["n_mismatch_explained_by_tie"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_eval_{args.retriever}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest(out, script=__file__, stage="retrieval evaluation / one retriever",
                   **retriever.describe(), top_n=args.top_n, n_cases=len(test_cases))
    log.info("wrote %s", out)


if __name__ == "__main__":
    sys.exit(main())
