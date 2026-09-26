# Quick sanity check of the trained LoRA adapter BEFORE investing in the
# fuse -> GGUF -> Ollama pipeline (see fine_tuning/README.md "Next stage").
#
# Runs the fine-tuned model on a handful of the 61 TRUE held-out cases
# (fine_tuning/data/holdout_eval_cases.json) and prints the generated query
# next to the gold query, so you can eyeball whether the adapter is
# producing plausible PyMongo -- not a scored metric, just "does this look
# sane before we spend 20+ minutes on GGUF conversion."
#
# Deliberately reuses SYSTEM_PROMPT from prepare_data.py (import, not a
# copy-paste duplicate) so the prompt this script sends the model is
# byte-identical to what every training example used -- a second, drifted
# copy of a 3.7k-character constant is exactly the kind of DRY violation
# already flagged elsewhere in this repo's code-smell review.
#
# Usage:
#   python fine_tuning/spot_check.py                # first 3 holdout cases
#   python fine_tuning/spot_check.py --n 5           # first 5
#   python fine_tuning/spot_check.py --adapter-path fine_tuning/adapters_smoketest
#
# Results are always written to fine_tuning/spot_check_output.<timestamp>.json
# (overwritten per run is NOT done -- each run gets its own timestamped file,
# so an earlier run's results are never silently lost) as well as printed.

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare_data import SYSTEM_PROMPT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.spot_check")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

# Generation post-processing and confidence capture now live in
# fine_tuning/generation_utils.py -- they are used by all three arms and by
# the demo UI, so a CLI sanity-check script was the wrong home for them.
# Re-exported here under their original names so every existing
# `from spot_check import clean` / `STOP_MARKERS` / `generate_with_logprobs`
# call site keeps working unchanged.
from generation_utils import (  # noqa: F401  re-export for backwards compatibility
    STOP_MARKERS,
    clean,
    clean_with_logprobs,
    confidence_fields,
    generate_with_logprobs,
    kept_char_mask,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Number of holdout cases to spot-check (default 3)")
    parser.add_argument("--adapter-path", default=str(ROOT / "adapters"), help="Path to the trained LoRA adapter")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    holdout_path = ROOT / "data" / "holdout_eval_cases.json"
    all_holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    cases = all_holdout[: args.n]
    log.info("Spot-checking %d of %d held-out cases against adapter at %s", len(cases), len(all_holdout), args.adapter_path)

    log.info("Loading %s with adapter %s ...", MODEL, args.adapter_path)
    model, tokenizer = load(MODEL, adapter_path=args.adapter_path)
    log.info("Model + adapter loaded.")

    results = []
    for i, case in enumerate(cases, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        log.info("[%d/%d] [%s] generating...", i, len(cases), case["id"])
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        generated = clean(raw)

        print(f"\n{'=' * 80}")
        print(f"[{case['id']}] ({case['database']}, {case.get('complexity')})")
        print(f"Q: {case['question']}")
        print(f"\nGOLD:      {case['normalized_query']}")
        print(f"GENERATED: {generated}")

        results.append({
            "id": case["id"],
            "database": case["database"],
            "complexity": case.get("complexity"),
            "question": case["question"],
            "gold": case["normalized_query"],
            "generated": generated,
            "raw_generated": raw,  # unstripped, in case the STOP_MARKERS truncation itself needs auditing later
        })

    print(f"\n{'=' * 80}")

    out_path = ROOT / f"spot_check_output.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "adapter_path": args.adapter_path,
        "n": len(cases),
        "results": results,
    }, indent=2), encoding="utf-8")
    log.info("Saved results -> %s", out_path)
    log.info("Done. This is a visual sanity check only -- NOT a scored metric. "
              "The real number comes from running the fused/GGUF/Ollama model "
              "through evaluation/execute_queries.py against all 61 cases.")


if __name__ == "__main__":
    main()
