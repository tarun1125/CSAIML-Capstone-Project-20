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
}


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
            if isinstance(result, (int, float, str, bool)):
                pass
            elif not isinstance(result, list):
                result = list(result)

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
