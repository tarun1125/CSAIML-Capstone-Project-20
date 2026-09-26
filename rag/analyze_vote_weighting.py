# Does a SIMILARITY-WEIGHTED vote keep the vote's noise-averaging while fixing
# the pool-frequency degeneracy that makes the unweighted vote lose to rank-1?
#
# Complements rag/analyze_vote_policies.py, which asks which AGGREGATION to use
# (vote vs restricted tie-break vs pure rank-1). This asks a different question:
# given that we aggregate, what should each neighbour's vote be WORTH?
#
# Runs offline in about a second. rag_prompts.json already records, per case,
# `retrieved_neighbor_databases` and `retrieved_scores` (the FAISS cosine
# similarities, added earlier precisely so rank could be separated from
# distance), so every weighting can be replayed on the recorded retrieval with
# no re-embedding, no Atlas and no model.
#
# THE FAMILY, and why softmax is the right one to sweep:
#
#   score(db) = sum over neighbours i of that db, of w(s_i)
#
# with w(s) = exp(s / tau), gives a one-parameter interpolation between the two
# policies already measured:
#
#   tau -> infinity   every w -> 1, so this IS the unweighted count vote (258/304)
#   tau -> 0          the single largest s dominates every sum, so argmax over
#                     databases becomes "the database of the nearest neighbour"
#                     -- i.e. exactly pure rank-1 (275/304)
#
# So the sweep cannot do worse than the better endpoint by much, and the
# interesting question is whether anything in the MIDDLE beats both ends. If
# something does, there is real signal in averaging that rank-1 throws away. If
# nothing does, the honest conclusion is that the vote contributes nothing once
# magnitude is respected, and rank-1 is not just better but sufficient.
#
# NOTE ON PLAIN SIMILARITY-SUM (w(s) = s), the obvious first idea: it does NOT
# fix the degeneracy, and it is included here to show that rather than assert
# it. Cosine similarities in this corpus run 0.31-0.999 with mean 0.66, so
# summing them is barely distinguishable from counting -- 84 pool examples at
# ~0.60 still outweigh 33 at ~0.65. Any weighting that SUMS over members inherits
# the pool-frequency prior; only a weighting sharp enough to be dominated by its
# maximum escapes it.
#
#   python rag/analyze_vote_weighting.py

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "rag" / "data" / "rag_prompts.json"


def unweighted_vote(dbs, sims):
    """build_prompts.py's current policy, replicated exactly -- including the
    tie-break that returns neighbour[0] without checking it is among the tied
    databases. See rag/eval_retrieval.py for why that is replicated, not fixed."""
    counts = Counter(dbs)
    top = max(counts.values())
    tied = [d for d, c in counts.items() if c == top]
    return tied[0] if len(tied) == 1 else dbs[0]


def rank1(dbs, sims):
    return dbs[0]


def weighted(weight_fn):
    """argmax over sum of per-neighbour weights. Ties break to the better-ranked
    database, so the policy is reproducible across runs -- same convention as
    rag/rerank.py's rerank_order()."""
    def policy(dbs, sims):
        totals = defaultdict(float)
        best_rank = {}
        for i, (d, s) in enumerate(zip(dbs, sims)):
            totals[d] += weight_fn(s, i)
            best_rank.setdefault(d, i)
        return min(totals, key=lambda d: (-totals[d], best_rank[d]))
    return policy


def mcnemar_exact(a: list[bool], b: list[bool]):
    from scipy import stats
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = float(stats.binomtest(only_b, n, 0.5).pvalue) if n else 1.0
    return only_a, only_b, p


def main():
    cases = json.loads(PROMPTS.read_text(encoding="utf-8"))
    gold = [c["gold_database"] for c in cases]
    dbs = [c["retrieved_neighbor_databases"] for c in cases]
    sims = [c["retrieved_scores"] for c in cases]
    n = len(cases)

    policies = [
        ("unweighted vote (current)", unweighted_vote),
        ("pure rank-1", rank1),
        ("sum of similarity", weighted(lambda s, i: s)),
        ("sum of similarity^4", weighted(lambda s, i: s ** 4)),
        ("sum of 1/rank", weighted(lambda s, i: 1.0 / (i + 1))),
    ]
    for tau in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        # exp(s/tau) overflows for small tau, and softmax is shift-invariant, so
        # subtract the per-case max. That is a numerical detail, not a change of
        # policy: it scales every weight in a case by the same constant.
        policies.append((
            f"softmax tau={tau}",
            weighted(lambda s, i, t=tau: math.exp(s / t)),
        ))

    correct_by_policy = {}
    print(f"{'policy':<28}{'correct':>10}{'accuracy':>11}   vs vote        vs rank-1")
    print("-" * 82)
    for name, fn in policies:
        preds = []
        for d, s, case in zip(dbs, sims, cases):
            mx = max(s)
            preds.append(fn(d, [x - mx for x in s]) if name.startswith("softmax") else fn(d, s))
        correct = [p == g for p, g in zip(preds, gold)]
        correct_by_policy[name] = correct
        line = f"{name:<28}{sum(correct):>7}/{n}{sum(correct)/n:>10.4f}"
        for ref in ("unweighted vote (current)", "pure rank-1"):
            if ref in correct_by_policy and ref != name:
                a, b, p = mcnemar_exact(correct_by_policy[ref], correct)
                line += f"   {b:+d}/{-a:d} p={p:.4f}"
            else:
                line += "   " + " " * 15
        print(line)

    print()
    best = max(correct_by_policy, key=lambda k: sum(correct_by_policy[k]))
    print(f"best: {best} at {sum(correct_by_policy[best])}/{n}")
    a, b, p = mcnemar_exact(correct_by_policy["pure rank-1"], correct_by_policy[best])
    print(f"best vs pure rank-1: +{b} / -{a}, exact p={p:.4f}")
    if b == a == 0:
        print("  -> IDENTICAL predictions to rank-1 on all 304 cases.")


if __name__ == "__main__":
    sys.exit(main())
