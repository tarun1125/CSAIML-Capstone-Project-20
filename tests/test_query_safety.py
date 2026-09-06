# Tests for check_query_is_safe() -- the only thing standing between a
# model-generated string and eval() against the live Atlas cluster.
#
# tests/test_scoring.py pins what the SCORER accepts. This file pins what the
# SAFETY GUARD rejects, which is a different question with a different failure
# mode: a scorer bug produces a wrong number, a guard bug produces a write
# against the cluster.
#
# The guard's history is a series of holes closed one at a time -- first the
# outer AST allowlist, then Finding 7's pipeline-stage check, then
# FORBIDDEN_OPERATORS walking every literal dict at any depth. Each round
# closed the payloads that had been thought of. This file exists so the next
# round starts from "which of these assertions fails" rather than from scratch.
#
# The organizing principle behind the current guard is FAIL CLOSED: if the
# checker cannot statically prove a dict key is a safe literal string, it
# rejects the query rather than passing it through unread. Every payload in
# TestOperatorNameIsNotWrittenLiterally below defeated a checker that read
# only literal keys, and each was verified to hand a real $out / $merge /
# $where to the driver.
#
#   python -m pytest tests/ -q

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from execute_queries import check_query_is_safe  # noqa: E402


def assert_rejected(query: str):
    ok, reason = check_query_is_safe(query)
    assert not ok, f"SECURITY: query was ALLOWED but must be rejected: {query}"
    return reason


def assert_allowed(query: str):
    ok, reason = check_query_is_safe(query)
    assert ok, f"legitimate query was rejected ({reason}): {query}"


# ---------------------------------------------------------------------------
# Queries the pipeline actually produces -- these must keep working.
#
# Verified against all 5,975 unique query strings in this repo's result and
# reference files: the fail-closed rules below reject none of them.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    'db.singer.find({}, {"_id": 0})',
    'db.c.find({"Year": {"$gte": 1977}}, {"_id": 0, "Model": 1}).sort({"Year": 1}).limit(10)',
    'db.c.aggregate([{"$match": {"x": 1}}, {"$group": {"_id": "$k", "n": {"$sum": 1}}}])',
    'db.c.aggregate([{"$lookup": {"from": "other", "localField": "a", "foreignField": "b", "as": "j"}}])',
    'db.c.count_documents({})',
    'db.c.distinct("Maker")',
    'len(list(db.c.find({}, {"_id": 0})))',
    'db["model_list.json"].find({}, {"_id": 0})',
    # $unionWith / $setWindowFields are read-only and appear in real gold
    # queries (dk-c2, spider-college_1-74, spider-bike_1-53 ...).
    'db.c.aggregate([{"$unionWith": {"coll": "other"}}])',
    'db.c.aggregate([{"$setWindowFields": {"output": {"avg": {"$avg": "$g"}}}}])',
])
def test_real_queries_still_allowed(query):
    assert_allowed(query)


# ---------------------------------------------------------------------------
# Writes and server-side JavaScript, written literally.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    'db.c.aggregate([{"$out": "pwned"}])',
    'db.c.aggregate([{"$merge": {"into": "pwned"}}])',
    'db.c.find({"$where": "function(){ return true; }"})',
    'db.c.aggregate([{"$match": {"$where": "function(){ return true; }"}}])',
    'db.c.aggregate([{"$addFields": {"x": {"$function": {"body": "f", "args": [], "lang": "js"}}}}])',
    'db.c.aggregate([{"$group": {"_id": None, "v": {"$accumulator": {"init": "f", "lang": "js"}}}}])',
    'db.c.aggregate([{"$facet": {"a": [{"$out": "pwned"}]}}])',
    'db.c.drop()',
    'db.c.insert_one({"x": 1})',
    'db.c.delete_many({})',
])
def test_writes_and_server_side_js_rejected(query):
    assert_rejected(query)


# ---------------------------------------------------------------------------
# The pipeline argument is not always a literal list.
#
# _check_pipeline_stages() only walks `aggregate()` calls whose FIRST
# POSITIONAL arg is an ast.List. Each of these evades that path; they are
# caught now only because FORBIDDEN_OPERATORS walks every dict in the tree
# independently of where it sits. If someone ever narrows that walk back to
# the pipeline check, these are the tests that fail.
#
# `pipeline=` is not exotic -- it is the real PyMongo keyword argument name.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    'db.c.aggregate(pipeline=[{"$out": "pwned"}])',
    'db.c.aggregate(*[[{"$out": "pwned"}]])',
    'db.c.aggregate([{"$match": {}}] + [{"$out": "pwned"}])',
    'db.c.aggregate(({"$out": "pwned"},))',
    'db.c.aggregate(list([{"$out": "pwned"}]))',
])
def test_non_literal_pipeline_argument_rejected(query):
    assert_rejected(query)


# ---------------------------------------------------------------------------
# The operator name is not written literally.
#
# This is the class the fail-closed rule exists for. Python computes the key
# at runtime, so a checker that reads only ast.Constant keys sees nothing
# forbidden and passes the query straight to eval(). Every payload here was
# confirmed to reach the driver as a real $out / $merge / $where before the
# guard was changed to reject keys it cannot statically read.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    'db.c.aggregate([{"$o" + "ut": "pwned"}])',                      # BinOp key
    'db.c.aggregate([{f"$merge": {"into": "pwned"}}])',              # JoinedStr key
    'db.c.aggregate([{"".join(["$", "out"]): "pwned"}])',            # method-call key
    'db.c.aggregate([{"$OUT".lower(): "pwned"}])',                   # method-call key
    'db.c.aggregate([{"%sout" % "$": "pwned"}])',                    # %-format key
    'db.c.find({"$wh" + "ere": "function(){ return true; }"})',      # BinOp key, find() filter
    'db.c.aggregate([dict([("$out", "pwned")])])',                   # no ast.Dict node at all
    'db.c.find(dict([("$where", "function(){ return true; }")]))',   # no ast.Dict node at all
    'db.c.aggregate([{**{"$out": "pwned"}}])',                       # ** unpacking, key is None
])
def test_operator_name_not_written_literally_rejected(query):
    assert_rejected(query)


# ---------------------------------------------------------------------------
# Escaping to arbitrary Python. These were all already blocked; the tests keep
# them blocked.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    'db.c.find().__class__',
    'db.c.find(__import__("os").system("id"))',
    '__import__("os").system("id")',
    'connect()',
    'MongoClient("mongodb://evil/")',
    'open("/etc/passwd").read()',
    'db.c.find({"x": [i for i in range(10)]})',
    'eval("1+1")',
])
def test_python_escape_rejected(query):
    assert_rejected(query)


def test_syntax_error_is_rejected_not_raised():
    ok, reason = check_query_is_safe("db.c.find({")
    assert not ok
    assert "syntax error" in reason
