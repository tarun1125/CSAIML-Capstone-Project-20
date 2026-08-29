# Fine-tuning generation extended from the original 61-case holdout to the
# full 304-case rag/data/rag_test.json -- the fine-tuning half of finishing
# the MLX-unified 3-arm comparison (see generate_baseline_mlx.py /
# rag/generate_rag_mlx.py at the repo root/rag/, and the project status
# doc's "First real MLX-unified 304-case run" entry for why this matters
# now: baseline and RAG were just re-measured on all 304 and RAG's number
# moved a lot once the serving-stack confound was removed -- fine-tuning
# needs the same full-304 treatment before any conclusion comparing the
# three arms is trustworthy).
#
# Deliberately a SEPARATE script from fine_tuning/generate_predictions.py,
# not a rewrite of it -- that script is the original, already-referenced
# 61-case reproducer (SYSTEM_PROMPT verbatim from training, no scope
# splitting needed since all 61 ids are in the 6 trained databases). This
# script adds two things that script was never built for: covering ids
# outside those 6 databases, and resuming a long run.
#
# WHY TWO DIFFERENT SYSTEM PROMPTS, NOT ONE:
# The adapter was LoRA-trained with ONE fixed prompt baked into every
# training example -- prepare_data.SYSTEM_PROMPT, hand-authored, 6
# databases, 27 collections (see fine_tuning/prepare_data.py). For the 80
# of 304 test ids whose database IS one of those 6, this script uses that
# exact verbatim prompt -- same as training, same as the original 20/61
# result, no confound.
#
# For the other 224 ids (17 databases the adapter never saw during
# training), using the SAME narrow 6-db prompt would be actively wrong --
# it doesn't even mention their database. But swapping to some ad-hoc new
# prompt format would introduce a SECOND, different confound (is a miss
# because the database is unfamiliar, or because the adapter's baked-in
# behavior doesn't recognize a differently-shaped prompt at all?). The
# least-confounded choice available: reuse generate_baseline_mlx.py's own
# PROMPT_HEADER/PROMPT_RULES/schema-assembly (import it directly, don't
# reimplement) to build ONE expanded prompt covering all 17 new databases,
# in the exact same header/rules wording and "Schema (N databases, M
# collections):" structure the training prompt already uses -- the closest
# a broader prompt can get to "the same kind of prompt, more schema" rather
# than a differently-styled one.
#
# THIS IS AN OUT-OF-DISTRIBUTION GENERALIZATION TEST FOR THOSE 224 IDS, NOT
# A FAIR EVAL OF THE ADAPTER'S TRAINED SCOPE -- say so explicitly wherever
# these numbers get reported. The 80 in-scope ids are the fair comparison
# point; the other 224 answer a different, also-interesting question ("how
# far does a narrow LoRA adapter's behavior transfer to schemas and
# databases it never saw"), and the two should never be silently pooled
# into one blended accuracy number without saying which is which.
#
# Usage:
#   python fine_tuning/generate_predictions_304.py
#   python fine_tuning/generate_predictions_304.py --adapter-path fine_tuning/adapters
#
# Resumable, same convention as generate_baseline_mlx.py / generate_rag_mlx.py:
# checkpoint written after every case.
#
# Output feeds:
#   python normalize.py data/finetuned_full304_results.json data/finetuned_full304_normalized.json
# then evaluation/execute_queries.py (or a small ad-hoc scoring pass, since
# this needs the in-scope/out-of-scope split evaluation/execute_queries.py's
# single RUNS-list entry doesn't give you by default).
#
# Side benefit, not the main point: the 80 in-scope ids include all 61 of
# the original holdout, generated fresh from the SAME adapter and SAME
# prompt as the original 20/61 (32.8%) result. If MLX/Metal generation is
# run-to-run consistent (an open question flagged in the project status
# doc), those 61 ids' accuracy here should land at or very near 32.8%
# again -- this run answers that consistency question for free, no
# separate re-run needed.

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
from prepare_data import SYSTEM_PROMPT  # noqa: E402  verbatim in-scope (6-db) prompt
from spot_check import STOP_MARKERS, clean  # noqa: E402  reuse, don't re-derive

sys.path.insert(0, str(REPO_ROOT))
from generate_baseline_mlx import (  # noqa: E402  reuse the baseline arm's own schema assembly
    PROMPT_HEADER, PROMPT_RULES, KNOWN_STRING_TYPED_NUMERIC_NOTE, build_schema_block,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.generate_predictions_304")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

# The 6 databases the adapter was actually trained on (verbatim from
# fine_tuning/prepare_data.py's own SYSTEM_PROMPT header comment / training
# data). Any test id outside this set is out-of-distribution for this
# adapter, whatever database it happens to be in.
ORIGINAL_6 = {"car_1", "concert_singer", "dog_kennels", "network_1", "pets_1", "world_1"}


def build_generalization_prompt(rag_prompts_path: Path, out_of_scope_databases: set) -> tuple[str, dict]:
    """One shared, expanded prompt for every out-of-scope database, built the
    same way generate_baseline_mlx.py builds its own SYSTEM_PROMPT -- same
    header/rules wording, same 'Schema (N databases, M collections):'
    framing, just a different (wider) set of databases in scope. Returns the
    prompt text plus the same coverage dict build_schema_block reports."""
    schema_text, coverage = build_schema_block(rag_prompts_path, out_of_scope_databases)
    n_colls = schema_text.count("\n- ")
    prompt = (
        PROMPT_HEADER
        + f"Schema ({len(coverage['covered'])} databases, {n_colls} collection(s) "
        + "-- generalization test: these databases were NOT in this adapter's training data):\n\n"
        + schema_text
        + "\n"
        + KNOWN_STRING_TYPED_NUMERIC_NOTE
        + PROMPT_RULES
    )
    return prompt, coverage


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
    parser.add_argument("--adapter-path", default=str(ROOT / "adapters"), help="Path to the trained LoRA adapter")
    parser.add_argument("--test-path", default=str(REPO_ROOT / "rag" / "data" / "rag_test.json"),
                         help="Full 304-case test slice (default), not the original 61-only holdout")
    parser.add_argument("--rag-prompts-path", default=str(REPO_ROOT / "rag" / "data" / "rag_prompts.json"),
                         help="Source of real schema for out-of-scope databases (see module docstring)")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "finetuned_full304_results.json"))
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    test_path = Path(args.test_path)
    output_path = Path(args.output)
    cases = json.loads(test_path.read_text(encoding="utf-8"))
    log.info("Loaded %d test-slice cases from %s", len(cases), test_path)

    in_scope_ids = {c["id"] for c in cases if c["database"] in ORIGINAL_6}
    out_of_scope_databases = {c["database"] for c in cases} - ORIGINAL_6
    log.info("Scope split: %d/%d ids are in the adapter's trained databases (fair eval), "
              "%d/%d are NOT (generalization test) -- see module docstring, do not pool these blindly.",
              len(in_scope_ids), len(cases), len(cases) - len(in_scope_ids), len(cases))

    generalization_prompt, coverage = build_generalization_prompt(Path(args.rag_prompts_path), out_of_scope_databases)
    missing = set(coverage["missing"])
    if missing:
        affected = [c for c in cases if c["database"] in missing]
        log.warning("Generalization prompt has NO schema for %s -- %d case(s) will run with their own "
                    "database entirely absent from the prompt: %s",
                    sorted(missing), len(affected), [c["id"] for c in affected])
    log.info("Generalization prompt assembled: %d character(s), %d/%d out-of-scope database(s) covered",
              len(generalization_prompt), len(coverage["covered"]), len(coverage["covered"]) + len(missing))

    done = load_checkpoint(output_path)
    if done:
        log.info("Resuming: %d/%d cases already have a saved result in %s -- skipping those.",
                  len(done), len(cases), output_path)

    log.info("Stop markers in use for generation cleanup: %s (works around mlx-lm issue #973)", STOP_MARKERS)
    log.info("Loading %s with adapter %s ...", MODEL, args.adapter_path)
    model, tokenizer = load(MODEL, adapter_path=args.adapter_path)
    log.info("Model + adapter loaded. Beginning generation over %d cases (%d remaining).", len(cases), len(cases) - len(done))

    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue

        in_scope = case["id"] in in_scope_ids
        system_prompt = SYSTEM_PROMPT if in_scope else generalization_prompt
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
            "[%d/%d] id=%s db=%s scope=%s (%.1fs) -> %s",
            i, len(cases), case["id"], case["database"],
            "in-scope" if in_scope else "GENERALIZATION", elapsed, generated[:80].replace("\n", " "),
        )

        done[case["id"]] = {
            "id": case["id"],
            "question": case["question"],
            "database": case["database"],
            "complexity": case.get("complexity"),
            "in_scope": in_scope,
            "generated_query": generated,
        }
        save_checkpoint(output_path, done, cases)  # write progress after EVERY case

    total_elapsed = time.monotonic() - run_start
    log.info("Generation complete: %d/%d cases in %s -> %.1fs total this run", len(done), len(cases), output_path, total_elapsed)
    log.info("Next: python normalize.py %s data/finetuned_full304_normalized.json",
              output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path)


if __name__ == "__main__":
    main()
