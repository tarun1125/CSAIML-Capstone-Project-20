# Ranking metrics for the reranking experiment (docs/EXPERIMENT-reranking.md §3).
#
# PURE FUNCTIONS over lists of relevance grades. No FAISS, no torch, no file
# I/O, nothing to mock -- which is the only reason tests/test_metrics_retrieval.py
# can assert against hand-worked numbers. A silently wrong nDCG is the easiest
# way to produce a confident wrong conclusion in this experiment, and it will
# not look wrong: it will be a plausible number between 0 and 1 that moves in
# the right direction for the wrong reason. Hand-computed fixtures are the
# defence.
#
# CONVENTION throughout: `grades` is a list ordered BY RANK, best-first, one
# entry per retrieved item. Element 0 is rank 1. Functions never re-sort their
# input -- the ranking is the thing under test.

import math

DEFAULT_RECALL_KS = (1, 3, 5, 10, 20)


def recall_at_k(grades: list[int], k: int) -> int:
    """1 if at least one relevant item (grade > 0) appears in the top k, else 0.

    Per-case, deliberately: the reported recall@k is the MEAN of this over
    cases, which is the standard "fraction of cases with a hit" definition the
    spec asks for. Returning a per-case 0/1 also keeps the paired structure --
    arm A and arm B answer the same 304 cases, so discordant cases can be
    compared directly instead of comparing two aggregate numbers (§7.3).
    """
    return int(any(g > 0 for g in grades[:k]))


def rank_of_first_relevant(grades: list[int], k: int | None = None) -> int | None:
    """1-indexed rank of the first grade > 0 within the top k, or None if there
    is none. None, not 0 and not len+1: "no relevant item was retrieved" is a
    different fact from "it was retrieved at rank 0", and collapsing them is how
    mean_rank quietly becomes uninterpretable."""
    window = grades if k is None else grades[:k]
    for i, g in enumerate(window):
        if g > 0:
            return i + 1
    return None


def reciprocal_rank(grades: list[int], k: int = 10) -> float:
    """1/rank of the FIRST relevant item; 0.0 if none in the top k.

    Binary by construction -- MRR does not read grade magnitude, only whether a
    grade is nonzero. That is not a simplification, it is what MRR is; nDCG is
    the metric that reads the grades.
    """
    rank = rank_of_first_relevant(grades, k)
    return 0.0 if rank is None else 1.0 / rank


def dcg(grades: list[int], k: int) -> float:
    """Standard log2 discount with the (2**g - 1) gain: sum over i of
    (2**g_i - 1) / log2(i + 2), i 0-indexed.

    The exponential gain is the standard formulation and the one that makes the
    graded label worth having -- with a linear gain, a grade-3 exemplar is worth
    exactly three grade-1s, which understates how much more useful an exact
    collection-set match is.
    """
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(grades: list[int], k: int = 10) -> float:
    """DCG normalised by the DCG of the ideal ordering of the SAME grades.

    IDEAL = this ranking's own grades sorted descending, NOT the best possible
    ordering of the whole 1,213-exemplar pool. That distinction matters and is
    worth stating in the write-up: it makes nDCG measure "did the ranker order
    what it retrieved well", not "did it retrieve the best possible set". The
    retrieval half of that question is what recall@k answers, and conflating the
    two into one number is how a reranker gets credit for a candidate set it did
    not choose.

    Returns 0.0 when no item is relevant -- ideal DCG is 0 there, and 0/0 is
    genuinely undefined; 0.0 is the convention and keeps the mean computable.

    Ties: sorted() is stable, so equal grades keep their retrieved order. The
    ideal DCG depends only on the multiset of grades, so this cannot change the
    value -- it is stated because reproducibility across runs should not rest on
    an implementation detail nobody wrote down.
    """
    ideal = sorted(grades[:k], reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(grades, k) / ideal_dcg


def summarize(
    per_case_binary: list[list[int]],
    per_case_graded: list[list[int]],
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict:
    """Aggregate one arm. Takes both label views because they answer different
    questions: recall/MRR run on the binary label (same-database), nDCG on the
    graded one. Returns the aggregates AND the per-case vectors, because §5's
    correlation between per-case nDCG and per-case execution correctness needs
    the per-case numbers, and recomputing them in the driver would be a second
    implementation to keep in step with this one.
    """
    n = len(per_case_binary)
    if n == 0:
        raise ValueError("no cases to summarize")
    if len(per_case_graded) != n:
        raise ValueError(f"binary/graded case counts disagree: {n} vs {len(per_case_graded)}")

    per_case_rr = [reciprocal_rank(g, mrr_k) for g in per_case_binary]
    per_case_ndcg = [ndcg_at_k(g, ndcg_k) for g in per_case_graded]
    first_ranks = [rank_of_first_relevant(g) for g in per_case_binary]
    found = sorted(r for r in first_ranks if r is not None)

    return {
        "n_cases": n,
        "recall_at_k": {
            str(k): sum(recall_at_k(g, k) for g in per_case_binary) / n for k in recall_ks
        },
        f"mrr_at_{mrr_k}": sum(per_case_rr) / n,
        f"ndcg_at_{ndcg_k}": sum(per_case_ndcg) / n,
        # Diagnostic, and skewed -- a handful of cases where the right database
        # never appears drag the mean far from the typical case, so the median
        # is reported next to it rather than instead of it (§3).
        "mean_rank_first_relevant": (sum(found) / len(found)) if found else None,
        "median_rank_first_relevant": _median(found),
        "n_cases_with_no_relevant": n - len(found),
        "per_case": {
            "reciprocal_rank": per_case_rr,
            "ndcg": per_case_ndcg,
            "rank_first_relevant": first_ranks,
        },
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2
