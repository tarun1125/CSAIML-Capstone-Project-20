# Full-scale generation run for the fine-tuned arm -- the real 61-case
# counterpart to spot_check.py's n=3/n=10 sanity checks.
#
# 2026-08-22 decision: this replaces the originally planned fuse -> GGUF ->
# Ollama -> generate path. spot_check.py already proved mlx_lm can load the
# adapter and generate directly, unfused, with zero extra tooling -- fusing
# was only ever a prerequisite for GGUF conversion, and GGUF/Ollama were
# only ever needed to *serve* the model somewhere else. Since scoring only
# needs generated text, not a servable model, this script just does the
# same thing spot_check.py does, at n=61, with no fuse/GGUF/Ollama step in
# between and no chat-template-mismatch risk. (Fusing + GGUF/Ollama import
# is still available later, standalone, if a demo wants the model runnable
# via `ollama run` -- that's fully decoupled from getting this score.)
#
# Output feeds the SAME two scripts the baseline/RAG arms already use,
# unmodified:
#   python normalize.py data/finetuned_results.json data/finetuned_normalized.json
#   python evaluation/execute_queries.py   (after adding the "Fine-tuned" row
#                                            to its RUNS list -- see that file)
#
# Usage:
#   python fine_tuning/generate_predictions.py
#   python fine_tuning/generate_predictions.py --adapter-path fine_tuning/adapters_smoketest
#
# This script never eval()s anything itself -- it only generates and
# string-cleans text. The actual safety gate (AST allowlist before eval) is
# evaluation/execute_queries.py's job, same as it already is for the other
# two arms; duplicating that check here would just be a second place for it
# to drift out of sync.

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from prepare_data import SYSTEM_PROMPT  # noqa: E402
from spot_check import STOP_MARKERS, clean  # noqa: E402  (reuse, don't re-derive)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.generate_predictions")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default=str(ROOT / "adapters"), help="Path to the trained LoRA adapter")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--holdout-path", default=str(ROOT / "data" / "holdout_eval_cases.json"),
        help="61-case holdout set (never used for training)",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "data" / "finetuned_results.json"),
        help="Where to write results, in the same shape as data/qwen2.5-coder_results.json",
    )
    args = parser.parse_args()

    holdout_path = Path(args.holdout_path)
    cases = json.loads(holdout_path.read_text(encoding="utf-8"))
    log.info("Loaded %d holdout cases from %s", len(cases), holdout_path)
    log.info("Stop markers in use for generation cleanup: %s (works around mlx-lm issue #973)", STOP_MARKERS)

    log.info("Loading %s with adapter %s ...", MODEL, args.adapter_path)
    model, tokenizer = load(MODEL, adapter_path=args.adapter_path)
    log.info("Model + adapter loaded. Beginning generation over %d cases.", len(cases))

    results = []
    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        case_start = time.monotonic()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.monotonic() - case_start
        generated = clean(raw)

        log.info(
            "[%d/%d] id=%s db=%s (%.1fs) -> %s",
            i, len(cases), case["id"], case["database"], elapsed, generated[:80].replace("\n", " "),
        )

        results.append({
            "id": case["id"],
            "question": case["question"],
            "database": case["database"],
            "complexity": case.get("complexity"),
            "generated_query": generated,
        })

    total_elapsed = time.monotonic() - run_start
    log.info("Generation complete: %d cases in %.1fs (%.1fs/case avg)", len(results), total_elapsed, total_elapsed / len(results))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved -> %s", out_path)
    log.info(
        "Next: python normalize.py %s %s",
        out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path,
        (out_path.parent / (out_path.stem.replace("_results", "_normalized") + ".json")).relative_to(REPO_ROOT),
    )


if __name__ == "__main__":
    main()
