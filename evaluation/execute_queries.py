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
from pathlib import Path

from pymongo import MongoClient
from pymongo.server_api import ServerApi


# ---------------------------------------------------------------------------
# Env / connection
# ---------------------------------------------------------------------------

def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def connect() -> MongoClient:
    project_root = Path(__file__).resolve().parents[1]
    env_values = load_env_file(project_root / "atlas-credentials.env")
    uri = env_values.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not found in atlas-credentials.env")

    client = MongoClient(uri, server_api=ServerApi("1"))
    client.admin.command("ping")
    print("[connect] Pinged Atlas, connection OK")
    return client


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
}


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
        if isinstance(node, ast.Name) and node.id not in {"db", "None", "True", "False"}:
            return False, f"unexpected name: {node.id}"

    return True, ""


def safe_eval_query(query: str, db):
    ok, reason = check_query_is_safe(query)
    if not ok:
        raise ValueError(f"REJECTED by safety check ({reason})")
    return eval(query)  # guarded by check_query_is_safe() above


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


def results_match(a, b) -> bool:
    """Order-insensitive comparison for list results, direct equality for
    scalars (counts, etc). Falls back to plain equality if items aren't
    sortable (e.g. mixed dict shapes)."""
    if isinstance(a, list) and isinstance(b, list):
        try:
            return sorted(map(_canonical, a)) == sorted(map(_canonical, b))
        except TypeError:
            return a == b
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
        ("ChatGPT", data_dir / "gpt_normalized.json", data_dir / "gpt_execution_results.json"),
    ]

    for model_name, input_file, output_file in RUNS:
        if not input_file.exists():
            print(f"[main] [{model_name}] SKIPPED: {input_file} not found")
            continue
        run_model(client, model_name, input_file, output_file, gold_results)
