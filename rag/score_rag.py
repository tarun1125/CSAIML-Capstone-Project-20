# Scores the RAG arm against gold, AND -- this is the part that matters for
# a fair comparison -- re-scores the ORIGINAL baseline Qwen run restricted
# to the SAME held-out test ids RAG was evaluated on (however many that is
# for the current split -- rag/build_split.py's TEST_FRACTION decides that,
# not this script).
#
# Why re-slice the baseline at all: outputs/execution_scores.csv already
# has a Qwen number, but that was computed over the FULL golden set --
# including the cases that are now the few-shot pool RAG retrieves from.
# Comparing RAG's test-slice score against a full-set baseline number would
# be comparing different test sets, not measuring what retrieval added.
# This script produces the one number that actually answers "did retrieval
# help": baseline-on-the-test-slice vs RAG-on-the-test-slice, same ids, same
# gold, same scoring code (imports safe_eval_query / to_json_safe /
# results_match / run_model from evaluation/execute_queries.py unchanged --
# no second scoring implementation to maintain or trust).
#
# Needs a real Atlas connection -- run this locally (same as
# execute_gold.py / execute_queries.py), not through a sandboxed tool
# environment with no route to Atlas.
#
# Usage (after normalize.py has produced rag/data/qwen_rag_normalized.json):
#   python rag/score_rag.py          # K=10 (default), matches build_prompts.py's
#                                     # own unsuffixed-by-default convention --
#                                     # reads rag_prompts.json / qwen_rag_normalized.json,
#                                     # writes rag_vs_baseline_scores.csv
#   python rag/score_rag.py 3        # K=3 sweep -- reads rag_prompts_k3.json /
#                                     # qwen_rag_normalized_k3.json, writes
#                                     # rag_vs_baseline_scores_k3.csv
#   python rag/score_rag.py 5        # same, for K=5
# The K suffix only touches RAG-specific filenames (prompts/normalized/
# execution-results/summary) -- the baseline test-slice re-score doesn't
# depend on K at all (same 61/304 ids regardless of how many few-shot
# examples RAG used), so it's recomputed fresh each run rather than K-suffixed.
#
# MLX-unification addition (see generate_baseline_mlx.py /
# rag/generate_rag_mlx.py at the repo root/rag/ -- Tarun's call to stop
# comparing arms generated on different serving stacks): two more OPTIONAL
# positional args let this same scoring code run against the new
# MLX-generated files instead of the historical Colab/HF ones, without
# touching the default (zero-arg) behavior at all --
#   python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_normalized.json
# When either override is given, this run's own output files get an
# "_mlx" marker (baseline_slice/execution-results/summary CSV) so it never
# silently overwrites the historical Colab-era files sitting at the
# original, un-suffixed names -- both runs' results stay on disk side by
# side for comparison.

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from execute_queries import connect, run_model  # noqa: E402  reuse, don't reimplement

TOP_K_ARG = sys.argv[1] if len(sys.argv) > 1 else None
BASELINE_FILE_ARG = sys.argv[2] if len(sys.argv) > 2 else None
RAG_NORMALIZED_FILE_ARG = sys.argv[3] if len(sys.argv) > 3 else None
SUFFIX = "" if TOP_K_ARG in (None, "10") else f"_k{TOP_K_ARG}"
RUN_MARKER = "_mlx" if (BASELINE_FILE_ARG or RAG_NORMALIZED_FILE_ARG) else ""


def main():
    data_dir = ROOT / "data"          # shared with baseline -- unchanged, stays at top level
    rag_data_dir = ROOT / "rag" / "data"  # RAG-specific artifacts
    rag_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[score_rag] K arg={TOP_K_ARG!r} -> file suffix={SUFFIX!r} "
          f"(empty suffix = the original K=10 filenames); "
          f"baseline override={BASELINE_FILE_ARG!r}, rag-normalized override={RAG_NORMALIZED_FILE_ARG!r}, "
          f"run marker={RUN_MARKER!r}")

    test_cases = json.loads((rag_data_dir / "rag_test.json").read_text(encoding="utf-8"))
    test_ids = {str(c["id"]) for c in test_cases}
    print(f"[score_rag] scoring against {len(test_ids)} held-out test ids: {sorted(test_ids)}")

    rag_normalized_file = Path(RAG_NORMALIZED_FILE_ARG) if RAG_NORMALIZED_FILE_ARG else rag_data_dir / f"qwen_rag_normalized{SUFFIX}.json"
    rag_prompts_file = rag_data_dir / f"rag_prompts{SUFFIX}.json"
    if not rag_normalized_file.exists():
        raise RuntimeError(
            f"{rag_normalized_file} not found. Run, in order: "
            f"1) python rag/build_prompts.py{' ' + TOP_K_ARG if TOP_K_ARG else ''} "
            f"(produces {rag_prompts_file.name}) "
            f"2) generation (rag/generate_rag_mlx.py, or the Colab RAG inference "
            f"notebook), pointed at {rag_prompts_file.name} "
            f"3) python normalize.py <results file> {rag_normalized_file}"
        )

    # Baseline: either the historical full-golden-set file, sliced down to
    # the current test ids (original behavior, unchanged), or -- when
    # BASELINE_FILE_ARG is given -- a file that's already exactly this
    # test slice (e.g. generate_baseline_mlx.py's output), in which case
    # the "slice" below is a no-op filter that just confirms full coverage.
    baseline_source = Path(BASELINE_FILE_ARG) if BASELINE_FILE_ARG else data_dir / "qwen_normalized.json"
    baseline_all = json.loads(baseline_source.read_text(encoding="utf-8"))
    baseline_slice = [c for c in baseline_all if str(c["id"]) in test_ids]
    if len(baseline_slice) != len(test_ids):
        missing = test_ids - {str(c["id"]) for c in baseline_slice}
        print(f"[score_rag] WARNING: {len(missing)} test id(s) not found in "
              f"{baseline_source.name}: {sorted(missing)}")
    baseline_slice_file = rag_data_dir / f"qwen_baseline_testslice_normalized{RUN_MARKER}.json"
    baseline_slice_file.write_text(json.dumps(baseline_slice, indent=2), encoding="utf-8")
    print(f"[score_rag] sliced baseline: {len(baseline_slice)}/{len(baseline_all)} "
          f"predictions kept (source: {baseline_source.name}) -> {baseline_slice_file}")

    gold_raw = json.loads((data_dir / "gold_results.json").read_text(encoding="utf-8"))
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[score_rag] loaded {len(gold_results)}/{len(gold_raw)} usable gold results")

    client = connect()

    n_test = len(test_ids)

    print("\n" + "=" * 80)
    print(f"[score_rag] Scoring BASELINE on the {n_test}-case test slice...")
    print("=" * 80)
    baseline_results = run_model(
        client, "Qwen-Baseline(test-slice)",
        baseline_slice_file, rag_data_dir / f"qwen_baseline_testslice_execution_results{RUN_MARKER}.json",
        gold_results,
    )

    print("\n" + "=" * 80)
    print(f"[score_rag] Scoring RAG on the {n_test}-case test slice...")
    print("=" * 80)
    rag_results = run_model(
        client, "Qwen-RAG",
        rag_normalized_file, rag_data_dir / f"qwen_rag_execution_results{SUFFIX}{RUN_MARKER}.json",
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
    prompts = json.loads(rag_prompts_file.read_text(encoding="utf-8"))
    db_match = sum(p["database_match"] for p in prompts)

    # Worst-case (p=0.5) standard error on a proportion, computed for whatever
    # n this run's test slice actually has -- not a number carried over from
    # an earlier, smaller split. SE = sqrt(p(1-p)/n); at p=0.5 that's the
    # widest a binomial's SE gets, so this is a conservative (upper-bound)
    # caveat, not an exact CI.
    se_pct = (0.5 * 0.5 / n_test) ** 0.5 * 100 if n_test else 0.0

    print("\n" + "=" * 80)
    print(f"[score_rag] FINAL COMPARISON  K={TOP_K_ARG or 10}  "
          f"(same {n_test} test ids, same gold, same scoring code)")
    print("=" * 80)
    print(f"  Database retrieval accuracy : {db_match}/{len(prompts)} "
          f"({db_match/len(prompts)*100:.1f}%)  -- diagnostic, not the headline number")
    print(f"  Baseline (test-slice)       : {b_correct}/{b_total} ({b_pct:.1f}%)")
    print(f"  RAG                         : {r_correct}/{r_total} ({r_pct:.1f}%)")
    print(f"  Delta                       : {r_pct - b_pct:+.1f} percentage points")
    print(f"\n  NOTE: n={n_test} -- worst-case standard error at p~=0.5 is ~{se_pct:.0f} points, "
          f"so treat this as a directional signal, not a precise measurement.")

    summary_path = ROOT / "rag" / "outputs" / f"rag_vs_baseline_scores{SUFFIX}{RUN_MARKER}.csv"
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
