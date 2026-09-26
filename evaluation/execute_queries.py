# Run model-generated queries against the real Atlas databases and score them
# for real, instead of just checking whether eval() raised an exception.
#
# Day-1 fixes applied here (see docs/BACKLOG.md #1-#5 for the original bug
# reports):
#   - routes each case to its OWN database via case["database"], instead of
#     always querying "concert_singer"
#   - compares the executed result against a gold result (data/gold_results.json,
#     produced by execute_gold.py) instead of treating "didn't throw" as PASS
#   - runs a minimal AST allowlist check before eval() -- rejects anything
#     that isn't a plain db.<collection>.<method>(...) call chain
#   - runs both models (Qwen + ChatGPT) in one pass instead of editing
#     INPUT/OUTPUT by hand and re-running
#
# Still deliberately simple: one file, no classes, no package -- matches the
# rest of the repo. Run with: python evaluation/execute_queries.py

import ast
import json
import sys
from datetime import date, datetime
from pathlib import Path

from bson import ObjectId
from pymongo import MongoClient  # still used as a type hint below (run_model)

# 2026-08-28: load_env_file/connect used to be defined here AND, separately
# and divergently, in atlas_verify_and_load.py -- a real DRY gap flagged by
# this project's own code-smell audit (the two connect()s had quietly grown
# different behavior: only one had a connection timeout + password-masked
# logging). Both now import the single shared implementation from
# atlas_env.py at the repo root instead. connect/load_env_file are
# re-exported under their original names so every existing
# `from execute_queries import connect` call site (fine_tuning/score_*.py,
# rag/score_rag.py, dump_atlas_to_local.py, evaluation/execute_gold.py)
# keeps working unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_env import connect, load_env_file  # noqa: E402  reuse, don't reimplement


# ---------------------------------------------------------------------------
# Minimal safety guard
#
# Not a full sandboxed module with tests -- just enough to stop a
# model-generated string from doing anything other than reading via
# db.<collection>.<method>(...). See docs/BACKLOG.md #5 for the fuller
# version this is trimmed from.
# ---------------------------------------------------------------------------

ALLOWED_METHODS = {
    "find", "find_one", "aggregate", "count_documents",
    "distinct", "sort", "limit", "skip",
    # Finding 8: safe read-only Python builtins that Claude uses in
    # some generated queries (e.g. len(list(db.x.find(...))), sorted(...)).
    # 6+ Claude-arm rejections were caused by their absence.
    "len", "sorted", "list", "dict",
}

# Finding 7: aggregation pipeline stages that are safe (read-only).
# Any stage key not in this set inside an aggregate() pipeline will be
# rejected by check_query_is_safe() -- this closes the gap where $merge,
# $out, $where, $function etc. could pass through the outer AST check
# uncaught because they're just literal dicts, not disallowed names.
SAFE_PIPELINE_STAGES = {
    "$match", "$project", "$group", "$sort", "$limit", "$skip",
    "$unwind", "$lookup", "$addFields", "$replaceRoot", "$count",
    "$set", "$unset", "$bucket", "$bucketAuto", "$facet",
    "$sortByCount", "$sample", "$redact", "$replaceWith",
    # Type conversion operators appearing as stage-level keys in some
    # gold queries (e.g. inside $addFields expressions)
    "$toInt", "$toDouble", "$toString", "$convert",
    "$toLong", "$toDecimal", "$toBool", "$toDate", "$toObjectId",
    "$cond", "$switch", "$ifNull",
    # 2026-09-05: both are read-only and both appear in GOLD queries that are
    # stored in data/gold_results.json with status PASS -- so they executed
    # once, but were not on this list, which meant execute_gold.py could no
    # longer regenerate 7 of its own gold results (dk-c2, spider-bike_1-1,
    # spider-bike_1-53, spider-college_1-74, spider-dog_kennels-69,
    # spider-formula_1-4, spider-sakila_1-15). Found by re-running
    # check_query_is_safe() over every stored query. $unionWith reads a second
    # collection (it is the only way to express SQL's UNION in an aggregation
    # pipeline); $setWindowFields computes window functions. Neither writes.
    # A $unionWith sub-pipeline is not walked by _check_pipeline_stages(), but
    # _check_forbidden_operators() above walks every dict at every depth, so a
    # $out/$merge/$where hidden inside one is still rejected.
    "$unionWith", "$setWindowFields",
}


# Operators that execute server-side JavaScript or otherwise escape the
# read-only contract, wherever they appear. _check_pipeline_stages() below
# only inspects the top-level stage keys of an aggregate() pipeline, so a
# $where nested inside a find() FILTER -- db.x.find({"$where": "..."}) --
# went straight through it: not an aggregate() call, and just a literal dict
# as far as the outer AST allowlist is concerned. Found by tests/test_scoring.py
# (2026-09-05). No query in any arm's results uses one of these, so closing
# the hole changes no published number; it closes it before one does.
FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$merge", "$out"}


def _check_forbidden_operators(tree) -> tuple[bool, str]:
    """Reject $where/$function/$accumulator/$merge/$out ANYWHERE in the query
    -- at any nesting depth, inside a find() filter or an aggregate() pipeline
    or a $lookup sub-pipeline alike.

    FAILS CLOSED. The first version of this check read only keys that were
    literal string Constants inside literal ast.Dict nodes, and skipped
    anything else. That is a hole, not a limitation, because the operator name
    never has to be written as a literal:

        db.c.aggregate([{"$o" + "ut": "pwned"}])      # BinOp key
        db.c.aggregate([{f"$merge": {...}}])          # JoinedStr key
        db.c.aggregate([dict([("$out", "pwned")])])   # no ast.Dict at all
        db.c.find({"$wh" + "ere": "function(){...}"}) # same trick, find() filter

    All four were verified to pass the old check and hand a real $out/$merge/
    $where to the driver. Python evaluates the key at runtime, so a checker
    that only understands literals cannot see them.

    Rather than trying to constant-fold arbitrary expressions -- an endless
    game against str.join, .replace, %-formatting, chr() and so on -- this
    refuses anything it cannot statically prove is a safe literal:

      * a dict key that is not a literal string Constant  -> reject
      * ** unpacking in a dict (key is None)              -> reject
      * a dict(...) constructor call                      -> reject

    Verified free: across all 20,014 query strings in this repo's result and
    reference files, zero use a non-literal dict key and zero call dict(),
    so this rejects nothing that has ever legitimately run."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            # dict(...) builds a mapping whose keys never appear as ast.Dict
            # keys, so no key-based check can see them.
            return False, "dict() constructor not allowed (operator keys cannot be checked statically)"

        if not isinstance(node, ast.Dict):
            continue

        for key_node in node.keys:
            if key_node is None:
                return False, "dict ** unpacking not allowed (keys cannot be checked statically)"
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                return False, (
                    "non-literal dict key not allowed "
                    f"({type(key_node).__name__} -- keys cannot be checked statically)"
                )
            if key_node.value in FORBIDDEN_OPERATORS:
                return False, f"method not allowed: forbidden operator {key_node.value}"
    return True, ""


def _check_pipeline_stages(tree) -> tuple[bool, str]:
    """Finding 7: inspect aggregate() pipeline contents for dangerous stages.
    Walks the AST looking for aggregate() calls whose first positional arg
    is a list of dicts, then checks every dict-key against SAFE_PIPELINE_STAGES.
    Rejects $merge, $out, $where, $function, $accumulator etc. that could
    write data or execute arbitrary JS against the live Atlas cluster."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "aggregate"
                and node.args):
            continue
        pipeline_arg = node.args[0]
        if not isinstance(pipeline_arg, ast.List):
            continue
        for stage_node in pipeline_arg.elts:
            if not isinstance(stage_node, ast.Dict):
                continue
            for key_node in stage_node.keys:
                if key_node is None:
                    continue  # **-unpacking, reject
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    stage_key = key_node.value
                    if stage_key.startswith("$") and stage_key not in SAFE_PIPELINE_STAGES:
                        return False, f"unsafe pipeline stage: {stage_key}"
    return True, ""


def check_query_is_safe(query: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"dunder access: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in ALLOWED_METHODS:
                return False, f"method not allowed: {node.func.attr}"
        if isinstance(node, ast.Name) and node.id not in {
            "db", "None", "True", "False",
            # Finding 8: safe builtins the model may wrap results in
            "len", "sorted", "list", "dict",
        }:
            return False, f"unexpected name: {node.id}"

    # Forbidden operators, at any depth (find() filters included)
    ok, reason = _check_forbidden_operators(tree)
    if not ok:
        return False, reason

    # Finding 7: pipeline-stage content inspection
    ok, reason = _check_pipeline_stages(tree)
    if not ok:
        return False, reason

    return True, ""


def safe_eval_query(query: str, db):
    ok, reason = check_query_is_safe(query)
    if not ok:
        raise ValueError(f"REJECTED by safety check ({reason})")
    return eval(query)  # guarded by check_query_is_safe() above


# ---------------------------------------------------------------------------
# JSON-safety for real Mongo results
#
# A query that runs successfully and returns real documents comes back with
# BSON-native types json.dump() can't handle on its own -- most commonly
# ObjectId on every document's own "_id" (none of the seed data sets a
# custom _id, so Atlas auto-assigns one on import; any find()/aggregate()
# that doesn't explicitly project _id away will carry one). Converting here,
# right where the result is captured, means a case that genuinely succeeded
# stays scored as a genuine success -- it doesn't get thrown away as a fake
# FAIL just because the value needs a string form to write to disk.
# ---------------------------------------------------------------------------

def materialize_result(result):
    """Turn whatever safe_eval_query() returned into the list-or-scalar shape
    the scorer compares, WITHOUT destroying it.

    The three call sites (run_model below, execute_gold.py, and the demo UI's
    execute_against_atlas) all used to open-code this as:

        if isinstance(result, (int, float, str, bool)): pass
        elif not isinstance(result, list):              result = list(result)

    That is correct for a Cursor and wrong for a dict, which is exactly what
    find_one() returns:

        list({"_id": 1, "Theme": "Sci-Fi"})  ==  ["_id", "Theme"]

    -- the field NAMES, with every value silently discarded. And when
    find_one() matches nothing it returns None, where list(None) raises
    TypeError and the case is recorded as a hard FAIL.

    Measured across this repo's committed execution results before the fix:
    27 records whose query used find_one -- 6 stored as bare key-lists, 18
    recorded as 'NoneType object is not iterable' failures, and
    execution_accuracy True on exactly zero of them. find_one is in
    ALLOWED_METHODS and normalize.py actively rewrites findOne -> find_one,
    so the pipeline was deliberately producing queries that could not score.

    No GOLD query uses find_one (0 of 1,517 in data/reference_queries.json),
    so no stored gold result was ever affected by this -- only model
    predictions, and only ever downward."""
    if isinstance(result, (int, float, str, bool)):
        return result
    if result is None:
        # find_one() with no match. The empty ANSWER, not an error.
        return []
    if isinstance(result, dict):
        # find_one() with a match -- one document. Wrap it; never iterate it.
        return [result]
    if isinstance(result, list):
        return result
    return list(result)  # Cursor / CommandCursor / other iterable


def to_json_safe(value, _converted=None):
    if _converted is None:
        _converted = []
    if isinstance(value, ObjectId):
        _converted.append("ObjectId")
        return str(value)
    if isinstance(value, (datetime, date)):
        _converted.append(type(value).__name__)
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_json_safe(v, _converted) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v, _converted) for v in value]
    return value


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _canonical(x):
    """Recursively turn dicts/lists into a form that's both comparable and
    sortable, and -- critically -- independent of dict key order. repr()
    isn't: {'_id': 1, 'n': 2} and {'n': 2, '_id': 1} are equal dicts but
    give different repr strings, which silently broke sorting/comparison
    whenever two engines (or two differently-ordered $group specs) emit the
    same values with different key order -- e.g. Atlas puts _id first,
    mongomock doesn't."""
    if isinstance(x, dict):
        return tuple(sorted((k, _canonical(v)) for k, v in x.items()))
    if isinstance(x, list):
        return tuple(_canonical(i) for i in x)
    return x


def _strip_null_id(d: dict) -> dict:
    """Return a copy of dict d with '_id' removed ONLY when its value
    is None (null). A null _id is a MongoDB aggregation artifact (e.g.
    $group: {_id: null}), not real data, so stripping it is safe. A
    non-null _id (e.g. _id: 'cat', _id: 11) carries real data from
    the query result and must participate in the comparison -- stripping
    it would mask genuinely different rows."""
    if d.get("_id") is None and "_id" in d:
        return {k: v for k, v in d.items() if k != "_id"}
    return d


def _values_canonical(d: dict) -> tuple:
    """Like _canonical but IGNORES keys -- returns only the sorted values,
    each individually canonicalized. Used for key-name-tolerant matching:
    the model outputs {'max': 640} where gold expects {'max_charge_amount':
    640} -- functionally identical, different alias."""
    return tuple(sorted(_canonical(v) for v in d.values()))


def results_match(a, b) -> bool:
    """Order-insensitive comparison for list results, direct equality for
    scalars (counts, etc). Falls back to plain equality if items aren't
    sortable (e.g. mixed dict shapes).

    Finding 3 -- two-tier matching for lists of dicts:
      1. Exact match (keys and values must both match)
      2. Key-tolerant fallback: if exact match fails, strip '_id' from
         both sides and compare by values only. Catches the 4 confirmed
         FT cases where the model computes the correct answer under a
         slightly different field alias. Conservative: same row count,
         same per-row value set, just tolerates different key names.
         Logged when triggered so the scoring change is visible."""
    if isinstance(a, list) and isinstance(b, list):
        try:
            if sorted(map(_canonical, a)) == sorted(map(_canonical, b)):
                return True
        except TypeError:
            if a == b:
                return True

        # --- Tier 2: key-name-tolerant fallback (Finding 3) ---
        if (a and b and len(a) == len(b)
                and all(isinstance(x, dict) for x in a)
                and all(isinstance(x, dict) for x in b)):
            try:
                a_stripped = [_strip_null_id(d) for d in a]
                b_stripped = [_strip_null_id(d) for d in b]
                a_vals = sorted(_values_canonical(d) for d in a_stripped)
                b_vals = sorted(_values_canonical(d) for d in b_stripped)
                if a_vals == b_vals:
                    print("[results_match] KEY-TOLERANT match: values identical, "
                          "key names differ (Finding 3 -- output key-name "
                          "normalization)")
                    return True
            except TypeError:
                pass

        return False
    return a == b


def is_empty(result) -> bool:
    if isinstance(result, (int, float)):
        return result == 0
    return not result


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(client: MongoClient, model_name: str, input_file: Path,
              output_file: Path, gold_results: dict) -> list[dict]:
    with input_file.open(encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for item in cases:
        case_id = str(item["id"])
        database = item.get("database")
        query = item["normalized_query"]

        print("=" * 80)
        print(f"[{model_name}] id={case_id} db={database}")

        record = {
            "id": item["id"],
            "question": item.get("question"),
            "database": database,
            "query": query,
        }

        if not database:
            print(f"[{model_name}] id={case_id} SKIPPED: no database field on this case")
            record["status"] = "SKIPPED"
            record["reason"] = "no database field"
            results.append(record)
            continue

        db = client[database]

        try:
            result = safe_eval_query(query, db)
            result = materialize_result(result)

            converted = []
            result = to_json_safe(result, converted)
            if converted:
                print(f"[{model_name}] id={case_id} converted {len(converted)} "
                      f"BSON value(s) to JSON-safe form for storage: "
                      f"{sorted(set(converted))}")

            record["status"] = "PASS"           # ran without error (old meaning of PASS)
            record["result"] = result
            record["non_empty_rate"] = not is_empty(result)

            gold = gold_results.get(case_id)
            if gold is None:
                record["execution_accuracy"] = None
                print(f"[{model_name}] id={case_id} ran OK, but no gold result for this id "
                      f"-- run execute_gold.py or check reference_queries.json")
            else:
                record["execution_accuracy"] = results_match(result, gold["result"])
                print(f"[{model_name}] id={case_id} ran OK -> "
                      f"non_empty={record['non_empty_rate']} "
                      f"execution_accuracy={record['execution_accuracy']}")

        except Exception as e:
            record["status"] = "FAIL"
            record["error"] = str(e)
            record["non_empty_rate"] = False
            record["execution_accuracy"] = False
            print(f"[{model_name}] id={case_id} FAILED: {e}")

        results.append(record)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    parsed = sum(r.get("status") == "PASS" for r in results)
    non_empty = sum(r.get("non_empty_rate") is True for r in results)
    correct = sum(r.get("execution_accuracy") is True for r in results)

    print(f"\n[{model_name}] SUMMARY  parse/run OK: {parsed}/{total}  "
          f"non_empty: {non_empty}/{total}  execution_accuracy: {correct}/{total}")
    print(f"[{model_name}] saved -> {output_file}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    gold_file = data_dir / "gold_results.json"
    if not gold_file.exists():
        raise RuntimeError(
            "data/gold_results.json not found. Run evaluation/execute_gold.py "
            "first -- execution_accuracy has nothing to compare against without it."
        )

    with gold_file.open(encoding="utf-8") as f:
        gold_raw = json.load(f)
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[main] loaded {len(gold_results)}/{len(gold_raw)} usable gold results "
          f"from {gold_file}")

    client = connect()

    RUNS = [
        ("Qwen2.5", data_dir / "qwen_normalized.json", data_dir / "qwen_execution_results.json"),
        # GPT arm replaced with Claude (data/claude_normalized.json, generated
        # by Claude directly since a live GPT arm isn't available -- labeled
        # honestly as "Claude" throughout, not disguised as GPT/ChatGPT).
        ("Claude", data_dir / "claude_normalized.json", data_dir / "claude_execution_results.json"),
        # Fine-tuned (LoRA, rank 16) arm, added 2026-08-22. Generated by
        # fine_tuning/generate_predictions.py directly from the trained
        # adapter (mlx_lm, no fuse/GGUF/Ollama step), then normalized the
        # same way as the other two arms. SKIPPED cleanly below if this
        # file doesn't exist yet.
        ("Fine-tuned", data_dir / "finetuned_normalized.json", data_dir / "finetuned_execution_results.json"),
    ]

    for model_name, input_file, output_file in RUNS:
        if not input_file.exists():
            print(f"[main] [{model_name}] SKIPPED: {input_file} not found")
            continue
        run_model(client, model_name, input_file, output_file, gold_results)
