# Decides, on evidence, how build_prompts.py's majority_vote_database()
# should break a tie -- and, more importantly, whether the majority vote
# is the right aggregator at all.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# majority_vote_database() breaks a tie by returning neighbor_dbs[0] (the
# rank-1 FAISS neighbor) WITHOUT checking that it is one of the tied
# databases. On neighbors [A, B, B, C, C] the tied set is {B, C} but the
# function returns A -- a database that lost the vote outright. That reads
# like an obvious bug, and the obvious "fix" is to restrict the tie-break
# to the tied set.
#
# It is not a bug, and that fix makes things worse. This script is the
# evidence, and it runs offline in about a second: rag_prompts.json already
# stores `retrieved_neighbor_databases` per case (the rank-ordered neighbor
# databases FAISS returned), so every aggregation policy can be replayed on
# the recorded retrieval without re-embedding anything or touching Atlas.
# No GPU, no model download, no network.
#
# WHAT IT FOUND (2026-09-22, rag/data/rag_prompts.json, TOP_K=10, n=304)
# ---------------------------------------------------------------------
#   current (rank-1 wins outright)  258/304 = 0.8487
#   restricted-to-tied-set          255/304 = 0.8388
#   PURE rank-1, no vote at all     275/304 = 0.9046
#
# The two tie-break variants differ on exactly 3 of 304 cases. A paired
# exact test on 3 discordant pairs bottoms out at p=0.25 -- that is the
# SMALLEST p-value attainable at n=3, so this split cannot resolve the
# tie-break question at all, in either direction. Anyone "fixing" the
# tie-break on these 3 cases is reading noise.
#
# The question the split CAN answer is the one underneath it: how much do
# you trust rank-1? Answer: a lot more than the vote. Pure rank-1 beats the
# vote on 20 cases and loses on 3 (McNemar exact p=0.0005). So deferring to
# rank-1 when the vote is inconclusive is not a lucky bug -- it is the
# better rule leaking through in the one place the code happens to apply it.
#
# WHY THE VOTE LOSES (the mechanism, not just the number)
# ------------------------------------------------------
# An unweighted k-NN vote throws away similarity MAGNITUDE and keeps only
# counts. When two databases are near-degenerate in embedding space -- the
# repo already documents two such families, chinook_1/store_1 and
# college_1/college_2/college_3 -- every pool example from either sibling
# sits at roughly the same cosine distance from the query. The expected
# number of neighbors from sibling c is then approximately
#
#     E[count_c] ~ K * N_c / sum_over_siblings(N_c')
#
# i.e. proportional to that sibling's POOL FREQUENCY, not to its relevance.
# argmax over counts collapses into argmax over pool frequency, so the
# rarer sibling can never win the vote however close it actually is. The
# recorded run shows exactly that: chinook_1 (33 pool examples) loses all
# 9 of its test cases to store_1 (84 pool examples), and college_3 (29)
# loses all 7 to college_1/college_2 (126/125). Rank-1 has no such failure
# mode -- it only asks which single example is closest, which is a genuine
# similarity comparison rather than a head-count against a prior.
#
# Put plainly: asking 10 townspeople to identify which of two identical
# twins you are looking at will always return whichever twin has more
# friends in that town. Asking only the one person standing closest to you
# will not.
#
# SCOPE -- what this script does NOT claim
# ----------------------------------------
# Database-retrieval accuracy is a diagnostic on step 2 of the pipeline. It
# selects which schema block goes into the prompt; it is not end-to-end
# query accuracy. Swapping the vote for pure rank-1 would change the schema
# block on ~17 additional cases and therefore invalidate every downstream
# generation artifact (qwen_rag_*.json, the score CSVs, the cross-arm
# figures), none of which can be regenerated without the MLX/Qwen serving
# stack. That swap is a real experiment with a real re-run cost, deliberately
# NOT performed here. This script documents the evidence for it; it does not
# silently make the change.
#
# Usage:
#   python rag/analyze_vote_policies.py                       # default K=10 run
#   python rag/analyze_vote_policies.py rag_prompts_k5.json   # any recorded run

import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.analyze_vote_policies")

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260922  # fixed so the reported CI is reproducible


# --- the policies under test -------------------------------------------------
# Each takes the rank-ordered neighbor databases and returns one database.

def policy_current(dbs: list[str]) -> str:
    """What build_prompts.py does today: majority wins; on a tie the rank-1
    neighbor wins OUTRIGHT, even if it is not in the tied set."""
    counts = Counter(dbs)
    top = max(counts.values())
    tied = [d for d, c in counts.items() if c == top]
    if len(tied) == 1:
        return tied[0]
    return dbs[0]


def policy_restricted(dbs: list[str]) -> str:
    """The 'obvious fix': on a tie, the highest-ranked member OF THE TIED SET."""
    counts = Counter(dbs)
    top = max(counts.values())
    tied = [d for d, c in counts.items() if c == top]
    return min(tied, key=dbs.index)


def policy_rank1(dbs: list[str]) -> str:
    """No vote at all -- just the nearest neighbor's database."""
    return dbs[0]


def policy_rrf(dbs: list[str], k: int = 60) -> str:
    """Reciprocal-rank-fusion weighted vote (rank-discounted, not flat)."""
    scores: Counter = Counter()
    for i, d in enumerate(dbs):
        scores[d] += 1.0 / (k + i + 1)
    return max(scores, key=lambda d: (scores[d], -dbs.index(d)))


def policy_linear(dbs: list[str]) -> str:
    """Linear rank-decay weighted vote: rank i contributes (K-i)/K."""
    scores: Counter = Counter()
    K = len(dbs)
    for i, d in enumerate(dbs):
        scores[d] += (K - i) / K
    return max(scores, key=lambda d: (scores[d], -dbs.index(d)))


def make_confidence_gated(threshold: int):
    """Trust the vote only when it is concentrated; below that, trust rank-1."""
    def policy(dbs: list[str]) -> str:
        if max(Counter(dbs).values()) < threshold:
            return dbs[0]
        return policy_current(dbs)
    policy.__doc__ = f"rank-1 when top_count < {threshold}, else the vote"
    return policy


def build_policies(K: int) -> list[tuple]:
    pols = [
        ("current (rank-1 wins outright)", policy_current),
        ("restricted to tied set", policy_restricted),
        ("pure rank-1 (no vote)", policy_rank1),
        ("RRF weighted vote", policy_rrf),
        ("linear-decay weighted vote", policy_linear),
    ]
    for t in range(2, K + 1):
        pols.append((f"gated: rank-1 if top_count<{t}", make_confidence_gated(t)))
    return pols


# --- statistics --------------------------------------------------------------

def mcnemar_exact(rows, pol_a, pol_b) -> tuple:
    """Paired two-sided exact (sign) test over the cases where the two
    policies disagree. Returns (a_only, b_only, p)."""
    a_only = b_only = 0
    for r in rows:
        dbs, gold = r["retrieved_neighbor_databases"], r["gold_database"]
        ok_a, ok_b = pol_a(dbs) == gold, pol_b(dbs) == gold
        a_only += ok_a and not ok_b
        b_only += ok_b and not ok_a
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = min(sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)
    return a_only, b_only, p


def bootstrap_delta(rows, pol_a, pol_b) -> tuple:
    """Percentile bootstrap 95% CI on (acc_a - acc_b), resampling cases."""
    rng = random.Random(BOOTSTRAP_SEED)
    per_case = [
        ((pol_a(r["retrieved_neighbor_databases"]) == r["gold_database"])
         - (pol_b(r["retrieved_neighbor_databases"]) == r["gold_database"]))
        for r in rows
    ]
    n = len(per_case)
    deltas = []
    for _ in range(BOOTSTRAP_ROUNDS):
        deltas.append(sum(per_case[rng.randrange(n)] for _ in range(n)) / n)
    deltas.sort()
    lo = deltas[int(0.025 * BOOTSTRAP_ROUNDS)]
    hi = deltas[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
    return sum(per_case) / n, lo, hi


# --- report ------------------------------------------------------------------

def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "rag_prompts.json"
    path = Path(__file__).resolve().parent / "data" / fname
    rows = json.loads(path.read_text(encoding="utf-8"))
    n = len(rows)
    K = len(rows[0]["retrieved_neighbor_databases"])
    log.info("Loaded %d recorded cases from %s (K=%d)", n, path.name, K)

    # Guard: the recorded predictions must match what policy_current does
    # now, or this file was written by different code and every number
    # below is measuring the wrong thing.
    drift = [r["id"] for r in rows
             if r["predicted_database"] != policy_current(r["retrieved_neighbor_databases"])]
    if drift:
        log.error("%d recorded prediction(s) disagree with policy_current -- "
                  "%s is stale relative to build_prompts.py. Regenerate it "
                  "before trusting this analysis. First few: %s",
                  len(drift), path.name, drift[:5])
        sys.exit(1)
    log.info("Integrity check passed: all %d recorded predictions replay exactly.", n)

    log.info("--- policy accuracy ---")
    print(f"\n{'policy':<36}{'correct':>10}{'accuracy':>11}")
    print("-" * 57)
    for name, pol in build_policies(K):
        ok = sum(pol(r["retrieved_neighbor_databases"]) == r["gold_database"] for r in rows)
        print(f"{name:<36}{f'{ok}/{n}':>10}{ok / n:>11.4f}")

    # Vote reliability as a function of how concentrated the vote is. This
    # is the table that explains everything else.
    log.info("--- vote reliability vs vote concentration ---")
    buckets = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        dbs, gold = r["retrieved_neighbor_databases"], r["gold_database"]
        b = buckets[max(Counter(dbs).values())]
        b[0] += 1
        b[1] += policy_current(dbs) == gold
        b[2] += dbs[0] == gold
    print(f"\n{'top_count':>10}{'n':>6}{'vote acc':>11}{'rank-1 acc':>12}")
    print("-" * 39)
    for tc in sorted(buckets):
        cnt, vote_ok, r1_ok = buckets[tc]
        print(f"{tc:>10}{cnt:>6}{vote_ok / cnt:>11.3f}{r1_ok / cnt:>12.3f}")

    # The 3 cases the tie-break question actually turns on.
    log.info("--- cases where current and restricted diverge ---")
    diverged = [r for r in rows
                if policy_current(r["retrieved_neighbor_databases"])
                != policy_restricted(r["retrieved_neighbor_databases"])]
    ties = sum(1 for r in rows
               if len([d for d, c in Counter(r["retrieved_neighbor_databases"]).items()
                       if c == max(Counter(r["retrieved_neighbor_databases"]).values())]) > 1)
    print(f"\nexact ties: {ties}/{n}   of which the two tie-breaks diverge: {len(diverged)}")
    for r in diverged:
        dbs = r["retrieved_neighbor_databases"]
        counts = Counter(dbs)
        top = max(counts.values())
        tied = sorted(d for d, c in counts.items() if c == top)
        print(f"\n  {r['id']}  gold={r['gold_database']}")
        print(f"    question   : {r['question']}")
        print(f"    neighbors  : {dbs}")
        print(f"    tied set   : {tied} (each {top}/{K})")
        print(f"    current    -> {policy_current(dbs)}"
              f"  [{'CORRECT' if policy_current(dbs) == r['gold_database'] else 'wrong'}]")
        print(f"    restricted -> {policy_restricted(dbs)}"
              f"  [{'CORRECT' if policy_restricted(dbs) == r['gold_database'] else 'wrong'}]")

    log.info("--- paired significance tests ---")
    print()
    for label, a, b in [
        ("current vs restricted", policy_current, policy_restricted),
        ("pure rank-1 vs current", policy_rank1, policy_current),
    ]:
        a_only, b_only, p = mcnemar_exact(rows, a, b)
        disc = a_only + b_only
        floor = min(2 / 2 ** disc, 1.0) if disc else 1.0
        delta, lo, hi = bootstrap_delta(rows, a, b)
        print(f"{label}:")
        print(f"    discordant pairs : {disc}  (first-only {a_only}, second-only {b_only})")
        print(f"    exact two-sided p: {p:.5f}   (floor attainable at n={disc}: {floor:.4f})")
        print(f"    delta accuracy   : {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        if p > 0.05 and math.isclose(p, floor, rel_tol=1e-9):
            print("    -> UNRESOLVABLE on this split: too few discordant pairs for ANY "
                  "result to reach significance. Decide on mechanism, not on this number.")
        elif p <= 0.05:
            print("    -> significant.")
        else:
            print("    -> not significant.")
        print()


if __name__ == "__main__":
    main()
