# Generation for the SECOND fine-tuned adapter -- the one trained on all 23
# databases via BATCHED schema prompts (fine_tuning/adapters_23db, see
# fine_tuning/prepare_data_23db.py and lora_config_23db.yaml), run over the
# full 304-case rag/data/rag_test.json. Deliberately separate from both
# fine_tuning/generate_predictions.py (the original 61-case, 6-db-adapter
# reproducer) and fine_tuning/generate_predictions_304.py (the 6-db
# adapter's generalization test over all 304).
#
# CONSISTENCY, READ BEFORE EDITING: this script does NOT recompute the
# database batching or reassemble any schema prompt text. It reads the
# batch definitions AND the exact, already-assembled system-prompt text
# for each batch straight out of fine_tuning/data_23db/split_manifest.json
# -- the file prepare_data_23db.py wrote at training time. This guarantees
# byte-identical prompts between training and generation regardless of any
# later drift in rag_test.json / rag_fewshot_pool.json / rag_prompts.json
# -- reading the literal recorded text removes the "two scripts computing
# the same thing independently" risk entirely, rather than just relying on
# both scripts happening to still agree.
#
# Usage:
#   python fine_tuning/generate_predictions_23db.py
#   python fine_tuning/generate_predictions_23db.py --adapter-path fine_tuning/adapters_23db
#
# Resumable, same convention as every other MLX generation script this
# project uses: checkpoint written after every case.
#
# Output feeds:
#   python normalize.py data/finetuned_full304_23db_results.json data/finetuned_full304_23db_normalized.json
# then:
#   python fine_tuning/score_finetuned_23db.py

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
from spot_check import STOP_MARKERS, clean  # noqa: E402  reuse, don't re-derive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.generate_predictions_23db")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"


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


def load_batch_prompts(manifest_path: Path) -> dict:
    """Reads fine_tuning/data_23db/split_manifest.json and returns
    {database_name: exact_training_system_prompt_text}. Fails loudly (not
    with a guessed fallback) if the manifest is missing -- that means
    prepare_data_23db.py hasn't been run yet, a real precondition, not a
    place to silently fall back to something else."""
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} not found. Run fine_tuning/prepare_data_23db.py first -- this "
            f"script reads the exact trained schema-batch prompts from that file, it does not "
            f"reassemble them independently (see this script's own module docstring for why)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_to_prompt = {}
    for batch in manifest["batches"]:
        for db in batch["databases"]:
            db_to_prompt[db] = batch["system_prompt"]
    log.info("Loaded %d batch prompt(s) covering %d database(s) from %s (batch_size=%s)",
              len(manifest["batches"]), len(db_to_prompt), manifest_path, manifest.get("batch_size"))
    return db_to_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default=str(ROOT / "adapters_23db"), help="Path to the 23-db-trained LoRA adapter")
    parser.add_argument("--test-path", default=str(REPO_ROOT / "rag" / "data" / "rag_test.json"))
    parser.add_argument("--manifest-path", default=str(ROOT / "data_23db" / "split_manifest.json"),
                         help="Written by prepare_data_23db.py -- source of the exact trained batch prompts")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "finetuned_full304_23db_results.json"))
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise RuntimeError(
            f"{adapter_path} not found. Train it first: "
            f"python fine_tuning/prepare_data_23db.py, then "
            f"mlx_lm.lora --config fine_tuning/lora_config_23db.yaml (see that file's own header "
            f"for the recommended smoke-test-first sequence)."
        )

    test_path = Path(args.test_path)
    output_path = Path(args.output)
    cases = json.loads(test_path.read_text(encoding="utf-8"))
    log.info("Loaded %d test-slice cases from %s", len(cases), test_path)

    db_to_prompt = load_batch_prompts(Path(args.manifest_path))
    missing_dbs = {c["database"] for c in cases} - set(db_to_prompt)
    if missing_dbs:
        raise RuntimeError(
            f"{sorted(missing_dbs)} appear in {test_path} but have NO batch prompt in the training "
            f"manifest -- rag_test.json has changed since prepare_data_23db.py last ran. Re-run "
            f"prepare_data_23db.py (and retrain) before generating, rather than guessing a prompt "
            f"for a database the adapter never actually trained on with this schema."
        )

    done = load_checkpoint(output_path)
    if done:
        log.info("Resuming: %d/%d cases already have a saved result in %s -- skipping those.",
                  len(done), len(cases), output_path)

    log.info("Stop markers in use for generation cleanup: %s (works around mlx-lm issue #973)", STOP_MARKERS)
    log.info("Loading %s with 23-db adapter %s ...", MODEL, adapter_path)
    model, tokenizer = load(MODEL, adapter_path=str(adapter_path))
    log.info("Model + adapter loaded. Beginning generation over %d cases (%d remaining).", len(cases), len(cases) - len(done))

    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue

        system_prompt = db_to_prompt[case["database"]]  # this case's own batch, matching training exactly
        messages = [
            {"role": "system", "content": system_prompt},
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

        done[case["id"]] = {
            "id": case["id"],
            "question": case["question"],
            "database": case["database"],
            "complexity": case.get("complexity"),
            "generated_query": generated,
        }
        save_checkpoint(output_path, done, cases)  # write progress after EVERY case

    total_elapsed = time.monotonic() - run_start
    log.info("Generation complete: %d/%d cases in %s -> %.1fs total this run", len(done), len(cases), output_path, total_elapsed)
    log.info("Next: python normalize.py %s data/finetuned_full304_23db_normalized.json",
              output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path)
    log.info("Then: python fine_tuning/score_finetuned_23db.py")


if __name__ == "__main__":
    main()
