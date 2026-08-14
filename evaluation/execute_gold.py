
import json
from pathlib import Path

from execute_queries import connect, safe_eval_query, is_empty, to_json_safe


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    input_file = data_dir / "reference_queries.json"
    output_file = data_dir / "gold_results.json"

    with input_file.open(encoding="utf-8") as f:
        cases = json.load(f)

    client = connect()

    results = []
    empty_ids = []
    failed_ids = []

    for item in cases:
        case_id = str(item["id"])
        database = item.get("database")
        query = item["normalized_query"]

        print("=" * 80)
        print(f"[gold] id={case_id} db={database}")

        record = {
            "id": item["id"],
            "question": item.get("question"),
            "database": database,
            "query": query,
        }

        if not database:
            print(f"[gold] id={case_id} SKIPPED: no database field -- "
                  f"add one in data/reference_queries.json")
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
                print(f"[gold] id={case_id} converted {len(converted)} BSON value(s) "
                      f"to JSON-safe form for storage: {sorted(set(converted))}")

            record["status"] = "PASS"
            record["result"] = result

            if is_empty(result):
                empty_ids.append(case_id)
                print(f"[gold] id={case_id} *** EMPTY RESULT -- this case is "
                      f"broken (wrong schema/collection/field), not the model ***")
            else:
                print(f"[gold] id={case_id} OK")

        except Exception as e:
            failed_ids.append(case_id)
            record["status"] = "FAIL"
            record["error"] = str(e)
            print(f"[gold] id={case_id} *** FAILED TO EXECUTE: {e} ***")

        results.append(record)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok = len(cases) - len(empty_ids) - len(failed_ids)
    print(f"\n[gold] SUMMARY  {ok}/{len(cases)} cases returned a non-empty result")
    print(f"[gold] saved -> {output_file}")

    if empty_ids:
        print(f"\n[gold] WARNING: {len(empty_ids)} case(s) returned EMPTY and cannot "
              f"be trusted as ground truth: {empty_ids}")
    if failed_ids:
        print(f"[gold] WARNING: {len(failed_ids)} case(s) FAILED to execute: {failed_ids}")
    if empty_ids or failed_ids:
        print("[gold] Fix these before trusting execution_accuracy numbers for these ids.")


if __name__ == "__main__":
    main()
