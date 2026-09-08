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
# Self-consistency (k samples at T > 0, confidence = fraction agreeing with
# the modal answer -- see run_samples() for why this is worth having next to
# mean logprob). ALWAYS write this to its own --output: a crash mid-sampling
# must not be able to damage the canonical greedy results file.
#   python rag/generate_rag_mlx.py --samples 5 --temp 0.7 \
#       --output rag/data/qwen_rag_mlx_selfconsistency_k5.json
#
# Resumable, same convention as generate_baseline_mlx.py: if --output
# already exists, ids already present are skipped and generation picks up
# where it left off, with a full checkpoint write after every case.
#
# CONFIDENCE (added 2026-09-05): each record now also carries mean_logprob /
# sum_logprob / token_logprobs / n_gen_tokens. Nothing in this repo used to
# emit a probability anywhere, so there was no confidence signal to calibrate
# against -- no ECE, no reliability diagram, no temperature scaling possible
# from these artifacts. Generation itself is UNCHANGED: mlx_lm 0.31.3's
# `generate()` is literally `"".join(r.text for r in stream_generate(...))`,
# so streaming to capture per-step logprobs runs the identical decode path
# and produces identical text (see spot_check.generate_with_logprobs).
#
# The logprobs are trimmed to the span clean() keeps, NOT recorded over the
# whole raw generation. This matters more than it looks: because of mlx-lm
# issue #973 the model keeps sampling past <|im_end|> to max_tokens, so a raw
# generation is typically a short correct query followed by a long discarded
# tail -- averaging over that tail would produce a "confidence" describing
# text that was thrown away. See spot_check.clean_with_logprobs.
#
# Output feeds:
#   python normalize.py rag/data/qwen_rag_mlx_results.json rag/data/qwen_rag_mlx_normalized.json
#   python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_normalized.json

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fine_tuning"))
from generation_utils import (  # noqa: E402  (reuse, don't re-derive)
    STOP_MARKERS,
    confidence_fields,
    generate_with_logprobs,
)

from run_manifest import read_manifest, write_manifest  # noqa: E402
from normalize import normalize  # noqa: E402  the canonical equality key -- see run_samples()

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


def run_samples(model, tokenizer, prompt, max_tokens: int, n_samples: int, temp: float) -> dict:
    """Generates `n_samples` completions and folds them into one record.

    SELF-CONSISTENCY, and why it needs its own equality key: mean logprob is
    the standard confidence signal and it is usually badly calibrated -- the
    model is just as fluent when it is wrong. The alternative signal is
    agreement: sample k times at T > 0 and ask what fraction of the samples
    landed on the same answer. That only works if "the same answer" is
    decided semantically, so this votes on normalize.py's AST-canonical form
    (which strips the outer list(), fixes Mongo-shell dialect, and re-unparses
    so whitespace and quoting cannot split a vote) rather than on raw strings.
    Two byte-different generations of the same query must count as agreement,
    or the agreement number is measuring formatting.

    ALL k generations are persisted, not just the winner -- an aggregate
    agreement fraction cannot be re-derived into a different bucketing later,
    and anything computing calibration downstream needs to see the spread it
    is calibrating."""
    sampler = make_sampler(temp=temp) if temp > 0 else None
    kwargs = {"sampler": sampler} if sampler is not None else {}

    samples = []
    for _ in range(n_samples):
        text, kept_lps, diag, _raw = generate_with_logprobs(
            model, tokenizer, prompt, max_tokens=max_tokens, **kwargs
        )
        conf = confidence_fields(kept_lps)
        samples.append({
            "query": text,
            "normalized_query": normalize(text) if text else "",
            "mean_logprob": conf["mean_logprob"],
            "sum_logprob": conf["sum_logprob"],
            "n_gen_tokens": conf["n_gen_tokens"],
            "logprob_trim_method": diag["logprob_trim_method"],
        })

    counts = Counter(s["normalized_query"] for s in samples)
    modal_key, modal_count = counts.most_common(1)[0]
    # Tie-break, and choice of representative within the winning group, by
    # highest mean logprob -- Counter.most_common's order is insertion order
    # on ties, which would silently make the result depend on sampling order.
    tied = [k for k, c in counts.items() if c == modal_count]
    if len(tied) > 1:
        def best_lp(key):
            return max((s["mean_logprob"] for s in samples
                        if s["normalized_query"] == key and s["mean_logprob"] is not None),
                       default=float("-inf"))
        modal_key = max(tied, key=best_lp)
    winner = max(
        (s for s in samples if s["normalized_query"] == modal_key),
        key=lambda s: (s["mean_logprob"] if s["mean_logprob"] is not None else float("-inf")),
    )

    return {
        # Kept at the top level under the same names the k=1 path uses, so
        # normalize.py / score_rag.py / execute_queries.py read this file with
        # zero changes -- the winner IS the arm's answer.
        "generated_query": winner["query"],
        "mean_logprob": winner["mean_logprob"],
        "sum_logprob": winner["sum_logprob"],
        "n_gen_tokens": winner["n_gen_tokens"],
        # The self-consistency signal itself.
        "self_consistency": counts[modal_key] / len(samples),
        "modal_count": counts[modal_key],
        "n_distinct_answers": len(counts),
        "n_samples": len(samples),
        "sample_temp": temp,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-path", default=str(REPO_ROOT / "rag" / "data" / "rag_prompts.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "rag" / "data" / "qwen_rag_mlx_results.json"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--samples", type=int, default=1,
                        help="Self-consistency: generate k samples per case and let confidence be "
                             "the fraction agreeing with the modal answer. k=1 (default) is the "
                             "existing greedy behavior, unchanged.")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Sampling temperature. Must be > 0 with --samples > 1: k greedy "
                             "samples are k identical samples and measure nothing.")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.samples > 1 and args.temp <= 0:
        parser.error(
            "--samples > 1 with --temp 0 would generate k identical greedy completions and report "
            "a self-consistency of 1.0 for every case -- a meaningless number that looks like a "
            "result. Pass --temp 0.7 (or similar) for a real self-consistency run."
        )
    if args.samples == 1 and args.temp > 0:
        log.warning("--temp %.2f with --samples 1: this is a single STOCHASTIC generation, not the "
                    "greedy baseline. Intentional?", args.temp)

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
        if args.samples > 1:
            gen_block = run_samples(model, tokenizer, prompt, args.max_tokens, args.samples, args.temp)
            elapsed = time.monotonic() - case_start
            log.info(
                "[%d/%d] id=%s gold_db=%s pred_db=%s match=%s (%.1fs) "
                "self_consistency=%d/%d (%d distinct) mean_lp=%s -> %s",
                i, len(cases), case["id"], case["gold_database"], case["predicted_database"],
                case["database_match"], elapsed,
                gen_block["modal_count"], gen_block["n_samples"], gen_block["n_distinct_answers"],
                f"{gen_block['mean_logprob']:.3f}" if gen_block["mean_logprob"] is not None else "n/a",
                gen_block["generated_query"][:70].replace("\n", " "),
            )
        else:
            generated, kept_lps, lp_diag, _raw = generate_with_logprobs(
                model, tokenizer, prompt, max_tokens=args.max_tokens
            )
            elapsed = time.monotonic() - case_start
            conf = confidence_fields(kept_lps)
            gen_block = {**conf, **lp_diag, "generated_query": generated}
            log.info(
                "[%d/%d] id=%s gold_db=%s pred_db=%s match=%s (%.1fs) mean_lp=%s (%d/%d tok kept) -> %s",
                i, len(cases), case["id"], case["gold_database"], case["predicted_database"],
                case["database_match"], elapsed,
                f"{conf['mean_logprob']:.3f}" if conf["mean_logprob"] is not None else "n/a",
                lp_diag["n_gen_tokens_kept"], lp_diag["n_gen_tokens_raw"],
                generated[:80].replace("\n", " "),
            )

        done[case["id"]] = {
            "id": case["id"],
            "question": case["question"],
            "database": case["gold_database"],
            "predicted_database": case["predicted_database"],
            "database_match": case["database_match"],
            "complexity": case.get("complexity"),
            **gen_block,
        }
        save_checkpoint(output_path, done, cases)  # write progress after EVERY case

    total_elapsed = time.monotonic() - run_start
    log.info("Generation complete: %d/%d cases in %s -> %.1fs total this run",
              len(done), len(cases), output_path, total_elapsed)

    # Carry the prompt build's own provenance (TOP_K, FK on/off, which index)
    # through into this run's manifest, so a results file records the WHOLE
    # chain that produced it rather than only the generation half.
    prompt_manifest = read_manifest(prompts_path) or {}
    manifest_file = write_manifest(
        output_path,
        script=__file__,
        arm="rag",
        model=MODEL,
        adapter=None,
        max_tokens=args.max_tokens,
        temp=args.temp,
        samples=args.samples,
        decoding=("greedy" if args.samples == 1 and args.temp == 0 else f"sampling T={args.temp}"),
        self_consistency_mode=args.samples > 1,
        n_cases=len(done),
        prompts_file=str(prompts_path.relative_to(REPO_ROOT)) if prompts_path.is_relative_to(REPO_ROOT) else str(prompts_path),
        prompts_manifest=prompt_manifest or None,
        top_k=prompt_manifest.get("top_k"),
        include_fk=prompt_manifest.get("include_fk"),
        records_logprobs=True,
    )
    log.info("Wrote run manifest -> %s", manifest_file)
    log.info(
        "Next: python normalize.py %s rag/data/qwen_rag_mlx_normalized.json",
        output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
    )
    log.info("Then: python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_normalized.json")


if __name__ == "__main__":
    main()
