# Tests for the two pieces of logic every published number in this project
# depends on: results_match() (what counts as a correct query) and normalize()
# (what the model's raw output is rewritten into before it is executed).
#
# There were no tests before this. The scoring code had been extended four
# times -- key-order independence, the null-_id strip, the key-tolerant
# fallback tier, the BSON conversion -- each time by reasoning about the
# change rather than pinning the behavior, and each extension made the
# comparison MORE permissive. This file pins what the comparison currently
# accepts AND, just as importantly, what it lets through: the four
# false-positive classes at the bottom are marked as characterization tests,
# not desired behavior. They exist so that anyone who tightens the scorer sees
# exactly which assertions flip.
#
#   python -m pytest tests/ -q

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from execute_queries import (  # noqa: E402
    check_query_is_safe,
    is_empty,
    materialize_result,
    results_match,
    to_json_safe,
)
from normalize import normalize  # noqa: E402


# ---------------------------------------------------------------------------
# results_match -- things it SHOULD accept
# ---------------------------------------------------------------------------

def test_identical_lists_match():
    assert results_match([{"a": 1}], [{"a": 1}])


def test_row_order_does_not_matter():
    # Mongo guarantees no document order without an explicit $sort, so two
    # runs of the same correct query can legitimately differ in row order.
    assert results_match([{"a": 1}, {"a": 2}], [{"a": 2}, {"a": 1}])


def test_key_order_does_not_matter():
    # Atlas emits _id first, mongomock does not. repr()-based comparison
    # silently broke on this; _canonical() sorts keys.
    assert results_match([{"_id": 1, "n": 2}], [{"n": 2, "_id": 1}])


def test_key_name_tolerant_fallback():
    # Finding 3: the model computed the right value under a different alias.
    assert results_match([{"max": 640}], [{"max_charge_amount": 640}])


def test_null_id_is_stripped_in_the_tolerant_tier():
    # {_id: null} is a $group artifact, not data.
    assert results_match([{"_id": None, "total": 7}], [{"total": 7}])


def test_scalar_equality():
    assert results_match(5, 5)
    assert not results_match(5, 6)


# ---------------------------------------------------------------------------
# results_match -- things it SHOULD reject
# ---------------------------------------------------------------------------

def test_different_values_do_not_match():
    assert not results_match([{"a": 1}], [{"a": 2}])


def test_non_null_id_still_participates():
    # A non-null _id carries real data (e.g. the $group key); stripping it
    # would mask genuinely different rows.
    assert not results_match([{"_id": "cat", "n": 1}], [{"_id": "dog", "n": 1}])


def test_extra_row_does_not_match():
    assert not results_match([{"a": 1}, {"a": 2}], [{"a": 1}])


def test_unprojected_id_breaks_the_match():
    # Characterization of a known FALSE NEGATIVE: 4 RAG cases fail only
    # because the model didn't project _id away. Recorded, not endorsed.
    assert not results_match([{"_id": "65f0", "name": "A"}], [{"name": "A"}])


# ---------------------------------------------------------------------------
# Known false-positive classes
#
# These assert that a WRONG query currently scores as correct. They are
# characterization tests -- they document the scorer's blind spots so a
# future tightening is visible as a deliberate assertion flip rather than a
# surprise. Each corresponds to a confirmed case in
# evaluation/audit_false_positives.py.
# ---------------------------------------------------------------------------

def test_fp_empty_matches_empty():
    # spider-flight_2-75: gold filters SourceAirport and returns 0; the model
    # filters DestAirport and also returns 0.
    assert results_match(0, 0)
    assert results_match([], [])


def test_fp_coincident_scalar():
    # spider-college_2-46: counting all departments and counting departments
    # that offer courses happen to give the same number.
    assert results_match(11, 11)


def test_fp_wrong_order_is_invisible():
    # spider-wine_1-30 / spider-apartment_rentals-29: the question asks for an
    # ORDER, the model produces the right set in the wrong order, and the
    # comparison is order-insensitive by design.
    ranked = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    reversed_ranking = [{"name": "C"}, {"name": "B"}, {"name": "A"}]
    assert results_match(reversed_ranking, ranked)


def test_fp_key_tolerant_tier_ignores_column_meaning():
    # The tolerant tier compares VALUES only. Two rows carrying the same
    # numbers under semantically different columns match.
    assert results_match([{"wins": 3, "losses": 5}], [{"losses": 3, "wins": 5}])


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ([], True), (0, True), (0.0, True), ({}, True), ("", True),
    ([{"a": 1}], False), (1, False), (-1, False), ([0], False),
])
def test_is_empty(value, expected):
    assert is_empty(value) is expected


# ---------------------------------------------------------------------------
# normalize -- the Mongo-shell/JS dialect fixes
# ---------------------------------------------------------------------------

def test_bare_null_becomes_none():
    assert normalize('db.x.find({"a": null})') == "db.x.find({'a': None})"


def test_quoted_null_string_is_left_alone():
    # "null" as a VALUE is real data in this dataset (car_1.cars_data stores
    # missing horsepower as the string "null"); only the bare token is rewritten.
    assert '"null"' in normalize('db.x.find({"a": "null"})').replace("'", '"')


def test_camelcase_method_is_renamed():
    assert normalize("db.x.countDocuments({})") == "db.x.count_documents({})"


def test_zero_arg_count_documents_gets_a_filter():
    # PyMongo's count_documents() requires an explicit filter; the shell's
    # countDocuments() does not. Renaming alone crashed on case cs-e2.
    assert normalize("db.x.countDocuments()") == "db.x.count_documents({})"


def test_outer_list_wrapper_is_unwrapped():
    assert normalize("list(db.x.find({}))") == "db.x.find({})"


def test_unparseable_input_is_left_as_is():
    # Better a clean downstream parse failure than a silently mangled query.
    assert normalize("db.x.find({") == "db.x.find({"


# ---------------------------------------------------------------------------
# check_query_is_safe -- the AST allowlist guarding eval()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "db.singer.find({})",
    "db.singer.count_documents({'a': 1})",
    "db.singer.aggregate([{'$match': {'a': 1}}, {'$group': {'_id': '$b'}}])",
    "len(list(db.singer.find({})))",
])
def test_safe_queries_pass(query):
    ok, reason = check_query_is_safe(query)
    assert ok, reason


@pytest.mark.parametrize("query,fragment", [
    ("db.x.drop()", "method not allowed"),
    ("db.x.delete_many({})", "method not allowed"),
    ("db.x.aggregate([{'$out': 'stolen'}])", "$out"),
    ("db.x.aggregate([{'$merge': {'into': 'y'}}])", "$merge"),
    ("db.x.aggregate([{'$function': {'body': 'f'}}])", "$function"),
    ("db.x.aggregate([{'$indexStats': {}}])", "unsafe pipeline stage"),
    ("__import__('os')", "unexpected name"),
    ("db.x.find({", "syntax error"),
])
def test_unsafe_queries_are_rejected(query, fragment):
    ok, reason = check_query_is_safe(query)
    assert not ok
    assert fragment in reason


@pytest.mark.parametrize("query", [
    # $where inside a find() FILTER is not an aggregate() pipeline and is only
    # a literal dict to the outer AST allowlist, so it slipped through both
    # checks until FORBIDDEN_OPERATORS was added. Regression test.
    "db.x.find({'$where': '1==1'})",
    "db.x.find({'a': 1, '$where': 'this.b > 2'})",
    "db.x.aggregate([{'$match': {'$where': 'true'}}])",
    # ...and nested inside a $lookup sub-pipeline, the deepest place it hides.
    "db.x.aggregate([{'$lookup': {'from': 'y', 'pipeline': [{'$match': {'$where': 'true'}}]}}])",
])
def test_server_side_javascript_is_rejected_at_any_depth(query):
    ok, reason = check_query_is_safe(query)
    assert not ok, f"{query!r} was allowed"
    assert "$where" in reason


# ---------------------------------------------------------------------------
# to_json_safe -- BSON values that would otherwise fail to serialize
# ---------------------------------------------------------------------------

def test_to_json_safe_converts_nested_objectid():
    from bson import ObjectId
    oid = ObjectId()
    converted = []
    out = to_json_safe({"rows": [{"_id": oid}]}, converted)
    assert out == {"rows": [{"_id": str(oid)}]}
    assert converted == ["ObjectId"]


def test_to_json_safe_leaves_plain_values_alone():
    assert to_json_safe({"a": 1, "b": [1, "x", None]}) == {"a": 1, "b": [1, "x", None]}


# ---------------------------------------------------------------------------
# materialize_result -- turning what safe_eval_query() returned into the shape
# the scorer compares.
#
# The three call sites (run_model, execute_gold.py, the demo UI) all used to
# open-code `if not isinstance(result, list): result = list(result)`, which is
# right for a Cursor and destructive for a dict. find_one() returns a dict, so
# list() handed back its KEYS and threw the values away; find_one() with no
# match returns None, where list() raised and the case was recorded FAIL.
# Measured before the fix: 27 find_one records in the committed execution
# results, 0 with execution_accuracy True.
# ---------------------------------------------------------------------------

def test_find_one_match_is_wrapped_not_iterated():
    # THE bug: list({"_id": 1, "Theme": "Sci-Fi"}) == ["_id", "Theme"]
    assert materialize_result({"_id": 1, "Theme": "Sci-Fi"}) == [{"_id": 1, "Theme": "Sci-Fi"}]


def test_find_one_no_match_is_empty_not_an_error():
    assert materialize_result(None) == []


def test_a_correct_find_one_can_now_score_correct():
    gold = [{"Theme": "Sci-Fi"}]
    assert results_match(materialize_result({"Theme": "Sci-Fi"}), gold)


def test_cursor_is_still_materialized():
    class FakeCursor:
        def __iter__(self):
            return iter([{"a": 1}, {"a": 2}])
    assert materialize_result(FakeCursor()) == [{"a": 1}, {"a": 2}]


@pytest.mark.parametrize("value", [5, 1.5, "x", True, [{"a": 1}], []])
def test_other_shapes_pass_through_unchanged(value):
    assert materialize_result(value) == value
