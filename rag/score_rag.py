# Scores the RAG arm against gold, AND -- this is the part that matters for
# a fair comparison -- re-scores the ORIGINAL baseline Qwen run restricted
# to the SAME 21 held-out test ids RAG was evaluated on.
#
# Why re-slice the baseline at all: outputs/execution_scores.csv already
# has a Qwen number (6.61%), but that was computed over all 121 cases.
# 100 of those 121 are now the few-shot pool RAG retrieves from -- comparing
# RAG's 21-case score against a 121-case baseline number would be comparing
# different test sets, not measuring what retrieval added. This script
# produces the one number that actually answers "did retrieval help":
# baseline-on-the-test-slice vs RAG-on-the-test-slice, same 21 ids, same
# gold, same scoring code (imports safe_eval_query / to_json_safe /
# results_match / run_model from evaluation/execute_queries.py unchanged --
# no second scoring implementation to maintain or trust).
#
# Needs a real Atlas connection -- run this locally (same as
# execute_gold.py / execute_queries.py), not through a sandboxed tool
# environment with no route to Atlas.
#
# Usage (after normalize.py has produced data/qwen_rag_normalized.json):
#   python rag/score_rag.py

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from execute_queries import connect, run_model  # noqa: E402  reuse, don't reimplement


def main():
    data_dir = ROOT / "data"

    test_cases = json.loads((data_dir / "rag_test.json").read_text(encoding="utf-8"))
    test_ids = {str(c["id"]) for c in test_cases}
    print(f"[score_rag] scoring against {len(test_ids)} held-out test ids: {sorted(test_ids)}")

    rag_normalized_file = data_dir / "qwen_rag_normalized.json"
    if not rag_normalized_file.exists():
        raise RuntimeError(
            f"{rag_normalized_file} not found. Run, in order: "
            f"1) the Colab RAG inference notebook (produces data/qwen_rag_results.json) "
            f"2) python normalize.py data/qwen_rag_results.json data/qwen_rag_normalized.json"
        )

    # Slice the ORIGINAL (121-case) baseline normalized predictions down to
    # just the 21 test ids, so it's scored on the exact same slice as RAG.
    baseline_all = json.loads((data_dir / "qwen_normalized.json").read_text(encoding="utf-8"))
    baseline_slice = [c for c in baseline_all if str(c["id"]) in test_ids]
    if len(baseline_slice) != len(test_ids):
        missing = test_ids - {str(c["id"]) for c in baseline_slice}
        print(f"[score_rag] WARNING: {len(missing)} test id(s) not found in "
              f"qwen_normalized.json: {sorted(missing)}")
    baseline_slice_file = data_dir / "qwen_baseline_testslice_normalized.json"
    baseline_slice_file.write_text(json.dumps(baseline_slice, indent=2), encoding="utf-8")
    print(f"[score_rag] sliced baseline: {len(baseline_slice)}/{len(baseline_all)} "
          f"predictions kept -> {baseline_slice_file}")

    gold_raw = json.loads((data_dir / "gold_results.json").read_text(encoding="utf-8"))
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[score_rag] loaded {len(gold_results)}/{len(gold_raw)} usable gold results")

    client = connect()

    print("\n" + "=" * 80)
    print("[score_rag] Scoring BASELINE on the 21-case test slice...")
    print("=" * 80)
    baseline_results = run_model(
        client, "Qwen-Baseline(test-slice)",
        baseline_slice_file, data_dir / "qwen_baseline_testslice_execution_results.json",
        gold_results,
    )

    print("\n" + "=" * 80)
    print("[score_rag] Scoring RAG on the 21-case test slice...")
    print("=" * 80)
    rag_results = run_model(
        client, "Qwen-RAG",
        rag_normalized_file, data_dir / "qwen_rag_execution_results.json",
        gold_results,
    )

    def accuracy(results):
        total = len(results)
        correct = sum(r.get("execution_accuracy") is True for r in results)
        return correct, total, (correct / total * 100 if total else 0.0)

    b_correct, b_total, b_pct = accuracy(baseline_results)
    r_correct, r_total, r_pct = accuracy(rag_results)

    # Also surface the retrieval-quality diagnostic build_prompts.py already
    # computed, so a reader sees "did retrieval find the right database" next
    # to "did the final answer come out correct" -- these can diverge (right
    # database, wrong query; or by luck, wrong database, still-correct
    # query on a simple case) and conflating them would hide which stage
    # of the pipeline needs work if the RAG number disappoints.
    prompts = json.loads((data_dir / "rag_prompts.json").read_text(encoding="utf-8"))
    db_match = sum(p["database_match"] for p in prompts)

    print("\n" + "=" * 80)
    print("[score_rag] FINAL COMPARISON (same 21 test ids, same gold, same scoring code)")
    print("=" * 80)
    print(f"  Database retrieval accuracy : {db_match}/{len(prompts)} "
          f"({db_match/len(prompts)*100:.1f}%)  -- diagnostic, not the headline number")
    print(f"  Baseline (test-slice)       : {b_correct}/{b_total} ({b_pct:.1f}%)")
    print(f"  RAG                         : {r_correct}/{r_total} ({r_pct:.1f}%)")
    print(f"  Delta                       : {r_pct - b_pct:+.1f} percentage points")
    print(f"\n  NOTE: n=21 -- standard error at p~=0.5 is ~11 points, so treat this "
          f"as a directional signal, not a precise measurement.")

    summary_path = ROOT / "outputs" / "rag_vs_baseline_scores.csv"
    summary_path.parent.mkdir(exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Arm,Correct,Total,Accuracy\n")
        f.write(f"Qwen-Baseline(test-slice),{b_correct},{b_total},{b_pct:.2f}\n")
        f.write(f"Qwen-RAG,{r_correct},{r_total},{r_pct:.2f}\n")
        f.write(f"DatabaseRetrievalDiagnostic,{db_match},{len(prompts)},"
                f"{db_match/len(prompts)*100:.2f}\n")
    print(f"[score_rag] saved -> {summary_path}")


if __name__ == "__main__":
    main()
