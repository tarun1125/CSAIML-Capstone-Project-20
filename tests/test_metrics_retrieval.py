# Hand-worked assertions for rag/metrics_retrieval.py and rag/relevance.py.
#
# WHY THESE ARE HAND-COMPUTED AND NOT GENERATED: the reranking experiment's
# conclusion is a comparison of nDCG/MRR across arms. A wrong nDCG would not
# announce itself -- it would be a plausible number in [0, 1] that still moves
# in the direction the experiment expects, and the write-up would report a
# confident wrong finding. So every expected value below is derived on paper
# first and written here as a literal, with the arithmetic in the comment. If a
# future change to dcg() breaks one of these, the test is right and the change
# is wrong until proven otherwise.
#
#   python -m pytest tests/test_metrics_retrieval.py -q

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rag"))

from metrics_retrieval import (  # noqa: E402
    dcg,
    ndcg_at_k,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from relevance import (  # noqa: E402
    binary_relevance,
    collections_in,
    gold_collections_of,
    graded_relevance,
)


# --------------------------------------------------------------------------
# recall@k / MRR -- binary label
# --------------------------------------------------------------------------

def test_recall_at_k_is_a_hit_indicator_not_a_count():
    # Two relevant items in the top 4 still scores 1, not 2: recall@k here is
    # "did at least one land", per the spec's definition.
    grades = [0, 1, 0, 1]
    assert recall_at_k(grades, 1) == 0
    assert recall_at_k(grades, 2) == 1
    assert recall_at_k(grades, 3) == 1
    assert recall_at_k(grades, 10) == 1  # k past the end of the list is fine


def test_recall_at_k_all_irrelevant():
    assert recall_at_k([0, 0, 0], 3) == 0


def test_rank_of_first_relevant_distinguishes_absent_from_rank_zero():
    assert rank_of_first_relevant([0, 1, 1]) == 2       # 1-indexed
    assert rank_of_first_relevant([1, 0, 0]) == 1
    assert rank_of_first_relevant([0, 0, 0]) is None    # not 0, not 4
    # Outside the window is the same as absent.
    assert rank_of_first_relevant([0, 0, 1], k=2) is None


def test_reciprocal_rank_hand_worked():
    assert reciprocal_rank([1, 0, 0]) == 1.0            # 1/1
    assert reciprocal_rank([0, 1, 0]) == 0.5            # 1/2
    assert reciprocal_rank([0, 0, 0, 1]) == 0.25        # 1/4
    assert reciprocal_rank([0] * 10 + [1], k=10) == 0.0  # first hit at 11, outside @10


def test_reciprocal_rank_ignores_grade_magnitude():
    # MRR is binary by construction -- a grade-3 at rank 2 and a grade-1 at
    # rank 2 are the same reciprocal rank. nDCG is the metric that reads grades.
    assert reciprocal_rank([0, 3, 0]) == reciprocal_rank([0, 1, 0])


# --------------------------------------------------------------------------
# nDCG -- graded label. Expected values derived on paper:
#
#   dcg(g) = sum_i (2**g_i - 1) / log2(i + 2)
#   log2(2) = 1, log2(3) = 1.5849625007, log2(4) = 2
# --------------------------------------------------------------------------

def test_dcg_hand_worked():
    #  7/1 + 0/1.5849625007 + 3/2  =  7 + 0 + 1.5  =  8.5
    assert dcg([3, 0, 2], 3) == pytest.approx(8.5, abs=1e-9)


def test_ndcg_hand_worked_imperfect_ordering():
    #   actual [3, 0, 2] -> 7 + 0 + 1.5                      = 8.5
    #   ideal  [3, 2, 0] -> 7 + 3/1.5849625007 + 0           = 8.8927892607
    #   8.5 / 8.8927892607                                   = 0.9558305893
    assert ndcg_at_k([3, 0, 2], 3) == pytest.approx(0.9558305893, abs=1e-9)


def test_ndcg_hand_worked_exactly_reversed():
    #   actual [1, 2, 3] -> 1 + 3/1.5849625007 + 7/2         = 6.3927892607
    #   ideal  [3, 2, 1] -> 7 + 3/1.5849625007 + 1/2         = 9.3927892607
    #   6.3927892607 / 9.3927892607                          = 0.6806060568
    assert ndcg_at_k([1, 2, 3], 3) == pytest.approx(0.6806060568, abs=1e-9)


def test_ndcg_hand_worked_with_gap():
    #   actual [2, 0, 0, 3] -> 3 + 0 + 0 + 7/log2(5)         = 6.0147359065
    #   ideal  [3, 2, 0, 0] -> 7 + 3/1.5849625007            = 8.8927892607
    #   6.0147359065 / 8.8927892607                          = 0.6763610078
    assert ndcg_at_k([2, 0, 0, 3], 4) == pytest.approx(0.6763610078, abs=1e-9)


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k([3, 2, 1], 3) == pytest.approx(1.0, abs=1e-12)
    # Already-sorted with ties is still perfect.
    assert ndcg_at_k([3, 3, 1, 0], 4) == pytest.approx(1.0, abs=1e-12)


def test_ndcg_no_relevant_items_is_zero_not_nan():
    # Ideal DCG is 0 here, so the ratio is genuinely undefined. 0.0 keeps the
    # mean over 304 cases computable instead of poisoning it with a nan.
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_normalises_against_its_own_candidates_not_the_pool():
    # A ranking that retrieved only weak items but ordered them perfectly scores
    # 1.0. This is deliberate (see ndcg_at_k's docstring): nDCG here measures
    # ordering quality, recall@k measures whether the good items were fetched at
    # all. Conflating them would let a reranker take credit for a candidate set
    # it did not choose.
    assert ndcg_at_k([1, 1, 1], 3) == pytest.approx(1.0, abs=1e-12)


def test_ndcg_truncates_at_k():
    # A grade-3 sitting at rank 11 contributes nothing to nDCG@10 -- and the
    # ideal is computed over the same truncated window, so this is 0.0, not a
    # penalised fraction.
    assert ndcg_at_k([0] * 10 + [3], 10) == 0.0


# --------------------------------------------------------------------------
# summarize -- the per-arm aggregate
# --------------------------------------------------------------------------

def test_summarize_aggregates_and_keeps_per_case_vectors():
    binary = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    graded = [[3, 0, 0], [0, 0, 2], [0, 0, 0]]
    out = summarize(binary, graded, recall_ks=(1, 3), mrr_k=10, ndcg_k=3)

    assert out["n_cases"] == 3
    assert out["recall_at_k"]["1"] == pytest.approx(1 / 3)
    assert out["recall_at_k"]["3"] == pytest.approx(2 / 3)
    # (1/1 + 1/3 + 0) / 3
    assert out["mrr_at_10"] == pytest.approx((1.0 + 1 / 3) / 3)
    # nDCG per case: perfect, perfect (only item is last but it is the only
    # nonzero one -> ideal is the same multiset ordered [2,0,0], so 2/log2(4)
    # over 2/log2(2) = 1.5/3 = 0.5), and 0.
    assert out["per_case"]["ndcg"][0] == pytest.approx(1.0)
    assert out["per_case"]["ndcg"][1] == pytest.approx(0.5)
    assert out["per_case"]["ndcg"][2] == 0.0
    # Ranks: 1, 3, and one case with nothing relevant -- excluded from the mean
    # rather than counted as a large rank.
    assert out["per_case"]["rank_first_relevant"] == [1, 3, None]
    assert out["mean_rank_first_relevant"] == pytest.approx(2.0)
    assert out["median_rank_first_relevant"] == pytest.approx(2.0)
    assert out["n_cases_with_no_relevant"] == 1


def test_summarize_rejects_mismatched_label_vectors():
    with pytest.raises(ValueError):
        summarize([[1]], [[1], [0]])


# --------------------------------------------------------------------------
# relevance labels
# --------------------------------------------------------------------------

def test_collections_in_root_and_lookup():
    q = ('db.country.aggregate([{"$lookup": {"from": "countrylanguage", '
         '"localField": "Code", "foreignField": "CountryCode", "as": "l"}}])')
    assert collections_in(q) == {"country", "countrylanguage"}


def test_collections_in_finds_lookup_nested_inside_facet():
    # The failure mode the spec calls out by name: a $lookup inside a $facet is
    # still a read of that collection, and 27 queries in this corpus nest that
    # way. A parser gets this for free; a shape-anchored regex does not.
    q = ('db.a.aggregate([{"$facet": {"branch": ['
         '{"$lookup": {"from": "b", "as": "x"}}]}}])')
    assert collections_in(q) == {"a", "b"}


def test_collections_in_union_with_and_subscript_form():
    q = ('db["model_list.json"].aggregate([{"$unionWith": '
         '{"coll": "Owners", "pipeline": []}}])')
    assert collections_in(q) == {"model_list.json", "Owners"}


def test_collections_in_unparseable_query_degrades_quietly():
    # One malformed exemplar should cost one relevance grade, not kill a
    # 15,200-pair evaluation run.
    assert collections_in("db.a.aggregate([{") == set()


def test_gold_collections_strips_only_a_matching_database_prefix():
    # Mixed qualification is real in rag_test.json: 331 of 450 entries carry a
    # `<db>.` prefix and 119 do not.
    case = {"database": "car_1",
            "gold_collections": ["car_1.cars_data", "car_1.model_list.json", "Department"]}
    # Split on the FIRST dot -- rsplit would yield "car_1.model_list", which is
    # not a collection that exists.
    assert gold_collections_of(case) == {"cars_data", "model_list.json", "Department"}


def test_gold_collections_leaves_a_foreign_prefix_alone():
    case = {"database": "car_1", "gold_collections": ["other_db.t"]}
    assert gold_collections_of(case) == {"other_db.t"}


def test_binary_relevance_is_database_identity():
    assert binary_relevance({"database": "car_1"}, {"database": "car_1"}) == 1
    assert binary_relevance({"database": "car_1"}, {"database": "world_1"}) == 0


def test_graded_relevance_full_scale():
    case = {"database": "car_1", "gold_collections": ["car_1.car_makers", "car_1.countries"]}

    def ex(db, query):
        return {"database": db, "normalized_query": query}

    exact = ex("car_1", 'db.car_makers.aggregate([{"$lookup": {"from": "countries", "as": "c"}}])')
    partial = ex("car_1", 'db.car_makers.find({}, {"_id": 0})')
    same_db_no_overlap = ex("car_1", 'db.cars_data.find({}, {"_id": 0})')
    other_db = ex("world_1", 'db.car_makers.find({}, {"_id": 0})')

    assert graded_relevance(exact, case) == 3
    assert graded_relevance(partial, case) == 2
    assert graded_relevance(same_db_no_overlap, case) == 1
    assert graded_relevance(other_db, case) == 0


def test_graded_relevance_superset_is_partial_not_exact():
    # Grade 3 is set EQUALITY. An exemplar touching every gold collection plus
    # another one is demonstrating a different join shape, and the grade should
    # say so.
    case = {"database": "car_1", "gold_collections": ["car_1.car_makers"]}
    superset = {"database": "car_1",
                "normalized_query": 'db.car_makers.aggregate([{"$lookup": {"from": "countries", "as": "c"}}])'}
    assert graded_relevance(superset, case) == 2


def test_graded_relevance_without_gold_collections_falls_back_to_one():
    case = {"database": "car_1", "gold_collections": []}
    exemplar = {"database": "car_1", "normalized_query": "db.car_makers.find({})"}
    assert graded_relevance(exemplar, case) == 1
