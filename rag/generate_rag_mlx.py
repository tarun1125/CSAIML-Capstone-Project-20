# RAG generation for the 304-case test slice, via MLX-LM -- the RAG half of
# the MLX-unification decision (see generate_baseline_mlx.py's module
# docstring at the repo root for the full "why" -- not repeated here).
#
# Unlike the baseline script, this one needs ZERO prompt construction: each
# of the 304 cases in rag/data/rag_prompts.json already carries its own
# fully-assembled system_prompt (retrieved few-shot examples + the
# retrieved database's schema), built by rag/build_prompts.py. That build
# step is CPU-only (FAISS + local schema lookups), unaffected by which
# stack does the actual Qwen generation, so it does not need rebuilding --
# this script just reads its output and generates.
#
# Usage:
#   python rag/generate_rag_mlx.py
#   python rag/generate_rag_mlx.py --prompts-path rag/data/rag_prompts.json \
#       --output rag/data/qwen_rag_mlx_results.json
#
# Resumable, same convention as generate_baseline_mlx.py: if --output
# already exists, ids already present are skipped and generation picks up
# where it left off, with a full checkpoint write after every case.
#
# Output feeds:
#   python normalize.py rag/data/qwen_rag_mlx_results.json rag/data/qwen_rag_mlx_normalized.json
#   python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_normalized.json

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fine_tuning"))
from spot_check import STOP_MARKERS, clean  # noqa: E402  (reuse, don't re-derive)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.generate_rag_mlx")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"  # same model, no adapter -- RAG is a base-model arm


def load_checkpoint(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in existing}
    except (json.JSONDecodeError, KeyError):
        log.warning("Could not read existing %s as a valid checkpoint -- starting fresh.", output_path)
        return {}


def save_checkpoint(output_path: Path, results_by_id: dict, case_order: list):
    ordered = [results_by_id[c["id"]] for c in case_order if c["id"] in results_by_id]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-path", default=str(REPO_ROOT / "rag" / "data" / "rag_prompts.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "rag" / "data" / "qwen_rag_mlx_results.json"))
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    prompts_path = Path(args.prompts_path)
    output_path = Path(args.output)
    cases = json.loads(prompts_path.read_text(encoding="utf-8"))
    log.info("Loaded %d RAG prompts from %s (each carries its own retrieved system_prompt)", len(cases), prompts_path)

    db_match = sum(1 for c in cases if c.get("database_match"))
    log.info("Retrieval diagnostic (unchanged from build_prompts.py's own run): "
              "database_match %d/%d (%.1f%%) -- generation quality is a separate question from this.",
              db_match, len(cases), db_match / len(cases) * 100 if cases else 0.0)

    done = load_checkpoint(output_path)
    if done:
        log.info("Resuming: %d/%d cases already have a saved result in %s -- skipping those.",
                  len(done), len(cases), output_path)

    log.info("Stop markers in use for generation cleanup: %s (works around mlx-lm issue #973)", STOP_MARKERS)
    log.info("Loading %s (no adapter -- RAG is a base-model arm) ...", MODEL)
    model, tokenizer = load(MODEL)
    log.info("Model loaded. Beginning generation over %d cases (%d remaining).", len(cases), len(cases) - len(done))

    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue

        messages = [
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user", "content": case["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        case_start = time.monotonic()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.monotonic() - case_start
        generated = clean(raw)

        log.info(
            "[%d/%d] id=%s gold_db=%s pred_db=%s match=%s (%.1fs) -> %s",
            i, len(cases), case["id"], case["gold_database"], case["predicted_database"],
            case["database_match"], elapsed, generated[:80].replace("\n", " "),
        )

        done[case["id"]] = {
            "id": case["id"],
            "question": case["question"],
            "database": case["gold_database"],
            "predicted_database": case["predicted_database"],
            "database_match": case["database_match"],
            "complexity": case.get("complexity"),
            "generated_query": generated,
        }
        save_checkpoint(output_path, done, cases)  # write progress after EVERY case

    total_elapsed = time.monotonic() - run_start
    log.info("Generation complete: %d/%d cases in %s -> %.1fs total this run",
              len(done), len(cases), output_path, total_elapsed)
    log.info(
        "Next: python normalize.py %s rag/data/qwen_rag_mlx_normalized.json",
        output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
    )
    log.info("Then: python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_normalized.json")


if __name__ == "__main__":
    main()
