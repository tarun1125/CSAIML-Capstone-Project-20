# Scores the fine-tuned arm's full-304 run (fine_tuning/generate_predictions_304.py
# -> normalize.py) against gold, the same way rag/score_rag.py scores
# baseline/RAG: reuses connect()/run_model() from evaluation/execute_queries.py
# unchanged (no second scoring implementation to maintain or trust).
#
# Why this is its OWN small script rather than an entry added to
# evaluation/execute_queries.py's RUNS list: that list produces one pooled
# accuracy number per arm, but this run's 304 predictions are NOT one
# homogeneous population -- 80 are in-scope (the adapter's own trained
# databases, the fair comparison point) and 224 are an out-of-distribution
# generalization test (see generate_predictions_304.py's module docstring).
# Pooling them into a single RUNS-list number would silently blend two
# different measurements into one misleading headline figure -- exactly
# what this project has been careful never to do (same discipline as
# score_rag.py's own test-slice re-slicing). This script scores everything
# once against gold (one Atlas pass, same as run_model always does), then
# reports the split three ways: in-scope only, out-of-scope-generalization
# only, and (clearly labeled, never as if it were one number) both.
#
# Needs a real Atlas connection -- run this locally, not through a
# sandboxed tool environment with no route to Atlas.
#
# Usage (after generate_predictions_304.py -> normalize.py have produced
# data/finetuned_full304_normalized.json):
#   python fine_tuning/score_finetuned_304.py
#   python fine_tuning/score_finetuned_304.py --normalized-path data/finetuned_full304_normalized.json

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from execute_queries import connect, run_model  # noqa: E402  reuse, don't reimplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-path", default=str(ROOT / "data" / "finetuned_full304_normalized.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "finetuned_full304_execution_results.json"))
    args = parser.parse_args()

    normalized_path = Path(args.normalized_path)
    output_path = Path(args.output)

    # in_scope was stamped onto every record by generate_predictions_304.py
    # and carried through unchanged by normalize.py (it only ever touches
    # normalized_query). run_model() below builds its own output records
    # from scratch and does NOT copy this field through, so keep our own
    # id -> in_scope map here to re-attach it after scoring.
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    in_scope_by_id = {str(r["id"]): r["in_scope"] for r in normalized}
    n_in_scope = sum(in_scope_by_id.values())
    print(f"[score_finetuned_304] loaded {len(normalized)} predictions from {normalized_path} "
          f"({n_in_scope} in-scope, {len(normalized) - n_in_scope} generalization-test)")

    gold_file = ROOT / "data" / "gold_results.json"
    if not gold_file.exists():
        raise RuntimeError(
            "data/gold_results.json not found. Run evaluation/execute_gold.py first."
        )
    gold_raw = json.loads(gold_file.read_text(encoding="utf-8"))
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[score_finetuned_304] loaded {len(gold_results)}/{len(gold_raw)} usable gold results")

    client = connect()

    print("\n" + "=" * 80)
    print(f"[score_finetuned_304] Scoring Fine-tuned(full-304) against gold...")
    print("=" * 80)
    results = run_model(client, "Fine-tuned(full-304)", normalized_path, output_path, gold_results)

    def accuracy(rows):
        total = len(rows)
        correct = sum(r.get("execution_accuracy") is True for r in rows)
        return correct, total, (correct / total * 100 if total else 0.0)

    in_scope_rows = [r for r in results if in_scope_by_id.get(str(r["id"])) is True]
    out_scope_rows = [r for r in results if in_scope_by_id.get(str(r["id"])) is False]

    all_c, all_t, all_p = accuracy(results)
    in_c, in_t, in_p = accuracy(in_scope_rows)
    out_c, out_t, out_p = accuracy(out_scope_rows)

    print("\n" + "=" * 80)
    print("[score_finetuned_304] FINAL COMPARISON -- three numbers, NOT one:")
    print("=" * 80)
    print(f"  In-scope (fair eval, adapter's 6 trained dbs, incl. all 61 original holdout ids):"
          f"  {in_c}/{in_t} ({in_p:.1f}%)")
    print(f"  Out-of-scope (generalization test, 17 unseen dbs -- OOD, not a fair headline number):"
          f"  {out_c}/{out_t} ({out_p:.1f}%)")
    print(f"  All 304 pooled (reference only -- DO NOT quote this alone, see module docstring):"
          f"  {all_c}/{all_t} ({all_p:.1f}%)")

    summary_path = ROOT / "fine_tuning" / "outputs" / "finetuned_304_scores.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Split,Correct,Total,Accuracy\n")
        f.write(f"InScope,{in_c},{in_t},{in_p:.2f}\n")
        f.write(f"OutOfScopeGeneralization,{out_c},{out_t},{out_p:.2f}\n")
        f.write(f"AllPooled_DO_NOT_QUOTE_ALONE,{all_c},{all_t},{all_p:.2f}\n")
    print(f"[score_finetuned_304] saved -> {summary_path}")


if __name__ == "__main__":
    main()
