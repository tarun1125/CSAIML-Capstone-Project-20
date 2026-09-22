# Driver for the reranking experiment (docs/EXPERIMENT-reranking.md §3 and §5).
# Writes results/retrieval_eval.json.
#
# Runs every arm over the SAME FAISS top-N candidate set for the same 304
# held-out cases, so the only variable between arms is the ordering. Then does
# the part that actually matters: correlates per-case nDCG@10 against per-case
# execution correctness, which is the question almost no reranking work can ask
# because it has no task-level ground truth to check the ranking metric against.
# This repo has one -- the execution oracle.
#
# Usage:
#   python rag/eval_retrieval.py                      # all arms, ranking metrics
#   python rag/eval_retrieval.py --emit-neighbors     # also write reranked
#                                                     # neighbor orders for
#                                                     # build_prompts.py --rerank
#
# The cross-encoder scores must already be cached -- run rag/rerank.py (both
# arms) first. This script never calls the model.

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# IMPORT ORDER IS LOAD-BEARING -- torch (via rerank/embed_utils) BEFORE faiss.
# See rag/build_prompts.py for the full libomp writeup. Do not let isort merge.
from rerank import DEFAULT_TOP_N, CROSS_ENCODER_MODEL, build_candidates, load_cache, rerank_order, variant_key  # noqa: E402

from embed_utils import MODEL_NAME as EMBED_MODEL_NAME  # noqa: E402
from metrics_retrieval import summarize  # noqa: E402
from relevance import binary_relevance, graded_relevance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from run_manifest import write_manifest  # noqa: E402

RAG_DATA = REPO_ROOT / "rag" / "data"
RESULTS_DIR = REPO_ROOT / "results"

# The canonical 304-case RAG execution run: the K=10 bi-encoder baseline, scored
# against the execution oracle. `execution_accuracy is True` is the correctness
# field -- NOT `status`, which only says the query ran without throwing. 290 of
# 304 cases have status PASS but only 141 are actually correct, and reading the
# wrong field would inflate the baseline by 49 points.
BASELINE_EXECUTION_RESULTS = RAG_DATA / "qwen_rag_execution_results_mlx.json"

# From rag/data/rag_prompts.manifest.json -- the K=10 bi-encoder run this
# experiment is measured against. Used as the §3 sanity-check target.
KNOWN_DB_RETRIEVAL_ACCURACY = 0.8486842105263158

log = logging.getLogger("rag.eval_retrieval")


def majority_vote_database(neighbor_dbs: list[str]) -> str:
    """A BYTE-FOR-BYTE replica of build_prompts.py's function of the same name.
    Reimplemented rather than imported because importing build_prompts executes
    its module-level `TOP_K = int(sys.argv[1])`, which would parse THIS script's
    arguments and die. Twelve duplicated lines is the better trade -- and the
    §3 sanity check below is what keeps the copy honest: if this ever drifts
    from the original, the reconstructed accuracy stops matching the recorded
    0.8487 and the run refuses to continue.

    NOTE THE TIE-BREAK, which is not what its own docstring in build_prompts.py
    describes. On a tie it returns `neighbor_dbs[0]` -- the rank-1 neighbour's
    database -- WITHOUT checking that this database is among the tied ones. For
    neighbours [A, B, B, C, C] the tied set is {B, C} but the function returns
    A, a database that lost the vote outright.

    This fires on exactly 3 of the 304 cases, and on all 3 the rank-1 database
    was the correct one -- which is the entire difference between 258/304
    (0.8487, the recorded value) and 255/304 (0.8388, what a tie-break
    restricted to the tied set gives). Replicated here deliberately and NOT
    corrected: this experiment measures the pipeline that exists, and changing
    the vote while also changing the ranking would confound the two. Fixing it
    is a separate change with its own before/after.
    """
    from collections import Counter
    counts = Counter(neighbor_dbs)
    top_count = max(counts.values())
    tied = [db for db, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return tied[0]
    return neighbor_dbs[0]  # tie-break: rank-1 neighbor -- see docstring


def grade_ranking(rows: list[int], metadata: list[dict], case: dict) -> tuple[list[int], list[int]]:
    """A ranked list of metadata rows -> (binary grades, graded grades), both in
    rank order."""
    exemplars = [metadata[r] for r in rows]
    return (
        [binary_relevance(e, case) for e in exemplars],
        [graded_relevance(e, case) for e in exemplars],
    )


def arm_orders(arm: str, candidate_rows: list[int], case: dict, metadata: list[dict],
               cache: dict) -> list[int]:
    """The ranked row list for one arm. `bi_encoder` is the FAISS order
    untouched; the rerank arms re-sort the same candidates by cached
    cross-encoder score."""
    if arm == "bi_encoder":
        return candidate_rows
    bucket = cache.get(variant_key(arm == "rerank_question_plus_query"), {})
    scores = []
    for row in candidate_rows:
        key = f"{case['id']}||{metadata[row]['id']}"
        if key not in bucket:
            raise KeyError(
                f"no cached cross-encoder score for {key} (arm={arm}). "
                f"Run: python rag/rerank.py"
                + (" --include-query" if arm == "rerank_question_plus_query" else "")
            )
        scores.append(bucket[key])
    return rerank_order(candidate_rows, scores)


def point_biserial(x: list[float], y: list[bool]) -> tuple[float, float]:
    """Correlation between a continuous per-case metric and a binary per-case
    outcome, with a two-sided p-value. Point-biserial is just Pearson with one
    binary variable, which is exactly this situation -- per-case nDCG against
    per-case execution correctness.

    Returns (nan, nan) when either side has no variance: if every case is
    correct, or every nDCG is identical, "how strongly do they co-vary" has no
    answer, and reporting a 0 there would read as "no relationship" when the
    truth is "not measurable on this data".
    """
    from scipy import stats
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


# The §5 table's rows: label -> (ranking arm, K, execution results file).
# The bi-encoder K=10 row points at the pre-existing canonical run rather than a
# regenerated one -- moving the baseline underneath the comparison would defeat
# the purpose of having one.
EXECUTION_ARMS = [
    ("bi_encoder_k5",  "bi_encoder",                 5,  "qwen_rag_k5_execution_results.json"),
    ("rerankQ_k5",     "rerank_question_only",       5,  "qwen_rag_rerankQ_k5_execution_results.json"),
    ("rerankQQ_k5",    "rerank_question_plus_query", 5,  "qwen_rag_rerankQQ_k5_execution_results.json"),
    ("bi_encoder_k10", "bi_encoder",                 10, "qwen_rag_execution_results_mlx.json"),
    ("rerankQ_k10",    "rerank_question_only",       10, "qwen_rag_rerankQ_k10_execution_results.json"),
]


def load_correctness(path: Path) -> dict[str, bool] | None:
    """case id -> did the generated query return the gold rows.

    `execution_accuracy is True`, NOT `status`: status only says the query ran
    without throwing. On the canonical run 290/304 cases have status PASS while
    only 141 are actually correct -- reading the wrong field would inflate every
    arm by ~49 points and make the whole table meaningless.
    """
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["id"]): (r.get("execution_accuracy") is True) for r in rows}


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Paired comparison of two arms over the SAME cases (spec §7.3).

    WHY NOT JUST COMPARE THE TWO ACCURACY NUMBERS: at n=304 the 95% interval on
    a single proportion is roughly +/-5.5 points, so two independent-looking
    accuracies have to differ enormously before that framing calls anything
    significant. But the arms are not independent samples -- they answer the
    same 304 questions with the same model and the same decoding, differing only
    in the exemplar list. Almost every case comes out the same way in both arms
    and carries no information about which is better. Only the DISCORDANT cases
    do, and McNemar's test is the test that looks at exactly those.

    Exact binomial rather than the chi-square approximation: the discordant
    counts here are expected to be small (tens at most), which is where the
    approximation is least trustworthy.
    """
    from scipy import stats
    shared = sorted(set(a) & set(b))
    only_a = sum(1 for i in shared if a[i] and not b[i])
    only_b = sum(1 for i in shared if b[i] and not a[i])
    both = sum(1 for i in shared if a[i] and b[i])
    neither = sum(1 for i in shared if not a[i] and not b[i])
    n_disc = only_a + only_b
    p = float(stats.binomtest(only_b, n_disc, 0.5).pvalue) if n_disc else 1.0
    return {
        "n_paired": len(shared),
        "both_correct": both,
        "both_wrong": neither,
        "only_baseline_correct": only_a,
        "only_arm_correct": only_b,
        "n_discordant": n_disc,
        "mcnemar_exact_p": p,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--with-execution", action="store_true",
                        help="also read each arm's execution results (produced by "
                             "rag/score_rerank_arms.py) and fill the §5 table, "
                             "including the paired discordant-case comparison")
    parser.add_argument("--emit-neighbors", action="store_true",
                        help="also write rag/data/reranked_neighbors_<arm>.json, "
                             "consumed by build_prompts.py --rerank for the §5 "
                             "execution arms")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_cases, metadata, candidate_rows, faiss_sims = build_candidates(args.top_n)
    cache = load_cache()
    for arm in ("question_only", "question_plus_query"):
        log.info("cached cross-encoder scores [%s]: %d", arm, len(cache.get(arm, {})))

    # ---------------------------------------------------------------
    # §3 sanity check.
    #
    # THE SPEC'S VERSION OF THIS CHECK IS WRONG and would have sent someone
    # hunting a bug that does not exist. It asks that recall@10 under the binary
    # label equal build_prompts.py's db_match_count/304 at TOP_K=10. Those are
    # different quantities:
    #   recall@10  = "did AT LEAST ONE same-database exemplar land in the top 10"
    #   db_match   = "did the MAJORITY VOTE over those 10 databases pick the
    #                 right one"
    # Measured: recall@10 = 0.987, db_match = 0.849. Both are correct; they
    # answer different questions. A majority vote can lose 4-3-3 while the right
    # database is sitting at rank 1.
    #
    # The check that actually tests what the spec wanted -- "am I reading the
    # same retrieval build_prompts.py read" -- is to reconstruct the majority
    # vote from the ranked list and compare THAT to the recorded 0.8487.
    # ---------------------------------------------------------------
    votes_correct = 0
    for case, rows in zip(test_cases, candidate_rows):
        top10 = [metadata[r]["database"] for r in rows[:10]]
        votes_correct += majority_vote_database(top10) == case["database"]
    reconstructed = votes_correct / len(test_cases)
    sanity_ok = abs(reconstructed - KNOWN_DB_RETRIEVAL_ACCURACY) < 1e-9
    log.info("SANITY: reconstructed majority-vote db accuracy @K=10 = %.6f "
             "(manifest recorded %.6f) -> %s",
             reconstructed, KNOWN_DB_RETRIEVAL_ACCURACY, "OK" if sanity_ok else "MISMATCH")
    if not sanity_ok:
        raise SystemExit(
            "Sanity check failed: this script is not reading the same retrieval "
            "build_prompts.py read. Fix that before trusting any metric below."
        )

    # ---------------------------------------------------------------
    # Ranking metrics, every arm x every K.
    # ---------------------------------------------------------------
    arms = ["bi_encoder", "rerank_question_only", "rerank_question_plus_query"]
    ranked_by_arm: dict[str, list[list[int]]] = {}
    for arm in arms:
        ranked_by_arm[arm] = [
            arm_orders(arm, rows, case, metadata, cache)
            for case, rows in zip(test_cases, candidate_rows)
        ]

    results = {}
    for arm in arms:
        binary, graded = [], []
        for case, rows in zip(test_cases, ranked_by_arm[arm]):
            b, g = grade_ranking(rows, metadata, case)
            binary.append(b)
            graded.append(g)
        results[arm] = summarize(binary, graded)
        # Majority-vote database accuracy at each K the §5 table uses -- this is
        # the mechanism by which a reranking can change the PROMPT, not just the
        # ranking, since build_prompts.py picks the schema from this vote.
        results[arm]["db_vote_accuracy"] = {
            str(k): sum(
                majority_vote_database([metadata[r]["database"] for r in rows[:k]]) == case["database"]
                for case, rows in zip(test_cases, ranked_by_arm[arm])
            ) / len(test_cases)
            for k in (5, 10)
        }
        r = results[arm]
        log.info("%-28s recall@5=%.4f  MRR@10=%.4f  nDCG@10=%.4f  dbvote@10=%.4f",
                 arm, r["recall_at_k"]["5"], r["mrr_at_10"], r["ndcg_at_10"],
                 r["db_vote_accuracy"]["10"])

    # ---------------------------------------------------------------
    # §5 -- does ranking predict the task?
    #
    # Available for the bi-encoder arm RIGHT NOW, with no new generation: the
    # canonical K=10 RAG execution run already covers these exact 304 cases.
    # This is the headline number, and it costs nothing.
    # ---------------------------------------------------------------
    correlation = None
    if BASELINE_EXECUTION_RESULTS.exists():
        exec_rows = json.loads(BASELINE_EXECUTION_RESULTS.read_text(encoding="utf-8"))
        correct_by_id = {r["id"]: (r.get("execution_accuracy") is True) for r in exec_rows}
        ids = [c["id"] for c in test_cases]
        missing = [i for i in ids if i not in correct_by_id]
        if missing:
            log.warning("%d test ids absent from %s -- excluded from the correlation",
                        len(missing), BASELINE_EXECUTION_RESULTS.name)
        paired = [(nd, correct_by_id[i])
                  for i, nd in zip(ids, results["bi_encoder"]["per_case"]["ndcg"])
                  if i in correct_by_id]
        ndcgs = [p[0] for p in paired]
        correct = [p[1] for p in paired]
        r_ndcg, p_ndcg = point_biserial(ndcgs, correct)
        r_rr, p_rr = point_biserial(
            [rr for i, rr in zip(ids, results["bi_encoder"]["per_case"]["reciprocal_rank"])
             if i in correct_by_id],
            correct,
        )
        correlation = {
            "source": str(BASELINE_EXECUTION_RESULTS.relative_to(REPO_ROOT)),
            "arm": "bi_encoder",
            "n_paired": len(paired),
            "execution_accuracy": sum(correct) / len(correct) if correct else None,
            "ndcg_at_10_vs_correct": {"point_biserial_r": r_ndcg, "p_value": p_ndcg},
            "mrr_at_10_vs_correct": {"point_biserial_r": r_rr, "p_value": p_rr},
            "mean_ndcg_when_correct": (
                float(np.mean([n for n, c in paired if c])) if any(correct) else None),
            "mean_ndcg_when_wrong": (
                float(np.mean([n for n, c in paired if not c])) if not all(correct) else None),
        }
        log.info("§5 CORRELATION (bi-encoder, n=%d): nDCG@10 vs execution correctness "
                 "r=%.4f (p=%.4g); mean nDCG correct=%.4f wrong=%.4f",
                 correlation["n_paired"], r_ndcg, p_ndcg,
                 correlation["mean_ndcg_when_correct"] or float("nan"),
                 correlation["mean_ndcg_when_wrong"] or float("nan"))
    else:
        log.warning("no baseline execution results at %s -- skipping the §5 correlation",
                    BASELINE_EXECUTION_RESULTS)

    # ---------------------------------------------------------------
    # §5 -- the experiment. Ranking metrics beside execution accuracy, on the
    # same 304 cases, with everything but the exemplar list held fixed.
    # ---------------------------------------------------------------
    execution_table = None
    if args.with_execution:
        execution_table = {"rows": [], "paired_vs_same_k_baseline": {}}
        correctness = {}
        for label, arm, k, fname in EXECUTION_ARMS:
            corr = load_correctness(RAG_DATA / fname)
            if corr is None:
                log.warning("§5 row %s: %s not present -- run rag/score_rerank_arms.py", label, fname)
                continue
            correctness[label] = corr
            r = results[arm]
            execution_table["rows"].append({
                "label": label,
                "ranking_arm": arm,
                "k": k,
                "source": fname,
                "recall_at_5": r["recall_at_k"]["5"],
                "mrr_at_10": r["mrr_at_10"],
                "ndcg_at_10": r["ndcg_at_10"],
                "db_vote_accuracy_at_k": r["db_vote_accuracy"][str(k)],
                "execution_accuracy": sum(corr.values()) / len(corr),
                "n_correct": sum(corr.values()),
                "n_cases": len(corr),
            })
            log.info("§5 %-16s recall@5=%.4f nDCG@10=%.4f  execution=%d/%d (%.2f%%)",
                     label, r["recall_at_k"]["5"], r["ndcg_at_10"],
                     sum(corr.values()), len(corr), sum(corr.values()) / len(corr) * 100)

        for base, arms_at_k in (("bi_encoder_k5", ("rerankQ_k5", "rerankQQ_k5")),
                                ("bi_encoder_k10", ("rerankQ_k10",))):
            if base not in correctness:
                continue
            for arm_label in arms_at_k:
                if arm_label not in correctness:
                    continue
                m = mcnemar_exact(correctness[base], correctness[arm_label])
                execution_table["paired_vs_same_k_baseline"][f"{arm_label}_vs_{base}"] = m
                log.info("§5 PAIRED %s vs %s: discordant %d (%d only-baseline, "
                         "%d only-arm), exact p=%.4g",
                         arm_label, base, m["n_discordant"],
                         m["only_baseline_correct"], m["only_arm_correct"],
                         m["mcnemar_exact_p"])

        # Per-arm correlation of per-case nDCG against per-case correctness --
        # the number the whole experiment exists to produce.
        ids = [str(c["id"]) for c in test_cases]
        per_arm_corr = {}
        for label, arm, _k, _f in EXECUTION_ARMS:
            if label not in correctness:
                continue
            corr = correctness[label]
            paired = [(nd, corr[i]) for i, nd in zip(ids, results[arm]["per_case"]["ndcg"]) if i in corr]
            if not paired:
                continue
            r_v, p_v = point_biserial([x for x, _ in paired], [y for _, y in paired])
            per_arm_corr[label] = {"point_biserial_r": r_v, "p_value": p_v, "n": len(paired)}
        execution_table["ndcg_vs_execution_per_arm"] = per_arm_corr

    # ---------------------------------------------------------------
    # Emit reranked neighbour orders for the §5 execution arms.
    # ---------------------------------------------------------------
    if args.emit_neighbors:
        for arm in ("rerank_question_only", "rerank_question_plus_query"):
            payload = {
                case["id"]: [metadata[r]["id"] for r in rows]
                for case, rows in zip(test_cases, ranked_by_arm[arm])
            }
            out = RAG_DATA / f"reranked_neighbors_{arm}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("wrote %d reranked orders -> %s", len(payload), out)

    # ---------------------------------------------------------------
    # Persist. Every number carries the config that produced it, so it can
    # always be traced back to the run that made it.
    # ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_eval.json"
    payload = {
        "config": {
            "top_n_candidates": args.top_n,
            "n_cases": len(test_cases),
            "pool_size": len(metadata),
            "embedding_model": EMBED_MODEL_NAME,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "faiss_index": "rag/data/fewshot.index",
            "recall_ks": [1, 3, 5, 10, 20],
            "mrr_k": 10,
            "ndcg_k": 10,
            "relevance_binary": "exemplar.database == case.database",
            "relevance_graded": "0 diff-db / 1 same-db no overlap / 2 partial / 3 exact "
                                "collection-set match against case.gold_collections",
            "tie_break": "cross-encoder score ties fall back to original FAISS rank",
        },
        "sanity_check": {
            "reconstructed_majority_vote_db_accuracy_k10": reconstructed,
            "manifest_recorded_db_retrieval_accuracy": KNOWN_DB_RETRIEVAL_ACCURACY,
            "passed": sanity_ok,
            "note": "The spec's §3 check (recall@10 == db_match_count/304) compares two "
                    "different quantities and does NOT hold: recall@10 is 'at least one "
                    "same-db exemplar in the top 10', db_match is 'the majority vote over "
                    "those 10 picked the right database'. The majority-vote "
                    "reconstruction above is the check that tests what §3 intended.",
        },
        "arms": results,
        "ranking_vs_execution": correlation,
        "execution_table": execution_table,
        "faiss_score_range": {
            "min": float(min(min(s) for s in faiss_sims)),
            "max": float(max(max(s) for s in faiss_sims)),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest(
        out_path,
        script=__file__,
        stage="retrieval evaluation / reranking",
        top_n=args.top_n,
        n_cases=len(test_cases),
        pool_size=len(metadata),
        embedding_model=EMBED_MODEL_NAME,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        arms=arms,
        sanity_check_passed=sanity_ok,
    )
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
