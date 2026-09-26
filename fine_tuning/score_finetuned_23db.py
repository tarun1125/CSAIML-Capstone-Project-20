# Scores the 23-db-trained fine-tuned adapter's full-304 run against gold.
# Reuses connect()/run_model() from evaluation/execute_queries.py unchanged
# (same pattern rag/score_rag.py and fine_tuning/score_finetuned_304.py both
# already use). Unlike score_finetuned_304.py (the 6-db adapter's
# generalization-test scorer), there's no in-scope/out-of-scope split
# needed here -- this adapter was trained on all 23 databases, so all 304
# results are one fair, directly comparable population. Reports a single
# accuracy number, same style as rag/score_rag.py's own output, so it slots
# directly next to the baseline/RAG/6-db-fine-tuned numbers for the final
# 4-way (or now 5-arm, counting both fine-tuned variants) comparison.
#
# Needs a real Atlas connection -- run this locally, not through a
# sandboxed tool environment with no route to Atlas.
#
# Usage (after generate_predictions_23db.py -> normalize.py have produced
# data/finetuned_full304_23db_normalized.json):
#   python fine_tuning/score_finetuned_23db.py
#   python fine_tuning/score_finetuned_23db.py --normalized-path data/finetuned_full304_23db_normalized.json

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from execute_queries import connect, run_model  # noqa: E402  reuse, don't reimplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-path", default=str(ROOT / "data" / "finetuned_full304_23db_normalized.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "finetuned_full304_23db_execution_results.json"))
    args = parser.parse_args()

    normalized_path = Path(args.normalized_path)
    output_path = Path(args.output)

    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    print(f"[score_finetuned_23db] loaded {len(normalized)} predictions from {normalized_path}")

    gold_file = ROOT / "data" / "gold_results.json"
    if not gold_file.exists():
        raise RuntimeError("data/gold_results.json not found. Run evaluation/execute_gold.py first.")
    gold_raw = json.loads(gold_file.read_text(encoding="utf-8"))
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[score_finetuned_23db] loaded {len(gold_results)}/{len(gold_raw)} usable gold results")

    client = connect()

    print("\n" + "=" * 80)
    print("[score_finetuned_23db] Scoring Fine-tuned(23db-trained, full-304) against gold...")
    print("=" * 80)
    results = run_model(client, "Fine-tuned(23db-trained)", normalized_path, output_path, gold_results)

    total = len(results)
    correct = sum(r.get("execution_accuracy") is True for r in results)
    pct = correct / total * 100 if total else 0.0

    print("\n" + "=" * 80)
    print("[score_finetuned_23db] FINAL RESULT")
    print("=" * 80)
    print(f"  Fine-tuned (trained on all 23 dbs), full 304-case test set: {correct}/{total} ({pct:.1f}%)")

    # Read baseline/RAG comparison numbers live from their own scored CSV
    # rather than hardcoding them here -- this line went stale once before
    # (kept citing RAG 136/304 after the FK/no-FK A/B re-run produced
    # 141-142/304) and silently misled anyone reading the console output.
    mlx_csv = ROOT / "rag" / "outputs" / "rag_vs_baseline_scores_mlx.csv"
    if mlx_csv.exists():
        import csv as _csv
        rows = {r["Arm"]: r for r in _csv.DictReader(mlx_csv.open())}
        b = rows.get("Qwen-Baseline(test-slice)")
        r = rows.get("Qwen-RAG")
        b_str = f"baseline {b['Correct']}/{b['Total']} ({float(b['Accuracy']):.1f}%)" if b else "baseline n/a"
        r_str = f"RAG {r['Correct']}/{r['Total']} ({float(r['Accuracy']):.1f}%)" if r else "RAG n/a"
        print(f"  Compare against: {b_str}, {r_str}, "
              "and the 6-db adapter's own in-scope/generalization split from score_finetuned_304.py.")
    else:
        print("  Compare against: baseline/RAG scores not found at "
              f"{mlx_csv} -- run rag/score_rag.py first for a live comparison.")

    summary_path = ROOT / "fine_tuning" / "outputs" / "finetuned_23db_scores.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Arm,Correct,Total,Accuracy\n")
        f.write(f"Fine-tuned(23db-trained-full304),{correct},{total},{pct:.2f}\n")
    print(f"[score_finetuned_23db] saved -> {summary_path}")


if __name__ == "__main__":
    main()
