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

from mlx_lm import generate, load, stream_generate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare_data import SYSTEM_PROMPT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.spot_check")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

# 2026-08-21: mlx-lm's TokenizerWrapper only checks a single eos_token_id,
# but Qwen2.5-Instruct models declare MULTIPLE valid end-of-turn tokens
# (confirmed: github.com/ml-explore/mlx-lm issue #973, open/unfixed as of
# the installed 0.31.3). Generation doesn't recognize <|im_end|> as a stop
# signal, so it keeps sampling garbage until max_tokens. The model itself
# is NOT confused -- it correctly emits <|im_end|> right after the real
# answer every time; mlx-lm just doesn't act on it. Fixed here at the
# string level (truncate at the first occurrence) rather than patching the
# library, since this is a known upstream bug, not something to work around
# by editing installed package internals.
STOP_MARKERS = ["<|im_end|>", "<|endoftext|>"]


def clean(raw: str) -> str:
    # Same ```python fence stripping the baseline notebook applies (cell 5)
    # -- Qwen sometimes wraps output in markdown despite being told not to.
    text = raw.replace("```python", "").replace("```", "")
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Per-token confidence, trimmed to the span clean() actually keeps.
#
# WHY THIS IS NOT JUST `sum(logprobs)/len(logprobs)`: because of mlx-lm issue
# #973 (see STOP_MARKERS above), generation does not stop at <|im_end|> -- it
# keeps sampling to max_tokens, so a typical raw generation is a short correct
# query followed by a long tail of garbage that clean() throws away. Averaging
# logprobs over the whole raw generation would therefore score mostly tokens
# that are not in the answer. A confidence number computed over text you
# discarded is not a confidence number for the text you kept, and any
# downstream calibration metric (ECE, reliability diagram, temperature
# scaling) built on it would be quietly, unfixably wrong.
#
# So: reproduce clean()'s edits at the CHARACTER level, then keep exactly the
# tokens that contributed at least one surviving character.
# ---------------------------------------------------------------------------


def _replace_all(raw: str, idxs: list[int], needle: str) -> list[int]:
    """One `str.replace(needle, "")` step, tracked back to raw indices.

    `idxs` are the indices of `raw` still surviving, in order; they spell the
    string the next edit operates on. This has to work on THAT string, not on
    `raw`, because clean() chains `.replace("```python","").replace("```","")`
    and each replace re-scans the previous one's OUTPUT -- a backtick run like
    '```python`````python' collapses differently under chained replaces than
    under two independent passes over the original. Matches str.replace's own
    rule: leftmost, non-overlapping, one left-to-right sweep."""
    s = "".join(raw[i] for i in idxs)
    out: list[int] = []
    pos = 0
    while True:
        j = s.find(needle, pos)
        if j == -1:
            out.extend(idxs[pos:])
            return out
        out.extend(idxs[pos:j])
        pos = j + len(needle)


def kept_char_mask(raw: str) -> list[bool]:
    """Per-character: does this character of `raw` survive clean()?

    Mirrors clean()'s three edits, in the same order and with the same
    chaining semantics -- ```python then ``` fence removal, truncation at the
    first stop marker, then .strip(). Verified against clean() itself by
    clean_with_logprobs below, which falls back to a conservative trim rather
    than trusting a mask that does not reproduce clean()'s own output."""
    idxs = list(range(len(raw)))

    # 1. Markdown fences, chained exactly as clean() chains them.
    idxs = _replace_all(raw, idxs, "```python")
    idxs = _replace_all(raw, idxs, "```")

    # 2. Truncate at the earliest stop marker. clean() searches for markers
    #    AFTER fence removal, so this searches the post-fence string too.
    s = "".join(raw[i] for i in idxs)
    cut = len(s)
    for marker in STOP_MARKERS:
        i = s.find(marker)
        if i != -1:
            cut = min(cut, i)
    idxs = idxs[:cut]

    # 3. .strip() -- leading/trailing whitespace of the CONCATENATED
    #    survivors, which is why this walks surviving indices rather than
    #    the raw string's own ends.
    start, end = 0, len(idxs)
    while start < end and raw[idxs[start]].isspace():
        start += 1
    while end > start and raw[idxs[end - 1]].isspace():
        end -= 1
    idxs = idxs[start:end]

    keep = [False] * len(raw)
    for i in idxs:
        keep[i] = True
    return keep


def clean_with_logprobs(pieces: list[str], token_logprobs: list[float]) -> tuple[str, list[float], dict]:
    """`pieces` and `token_logprobs` are the per-step text and chosen-token
    logprob from mlx_lm.stream_generate, in generation order.

    Returns (cleaned_text, kept_logprobs, diagnostics). cleaned_text is
    byte-identical to clean("".join(pieces)) -- this function changes nothing
    about the text, it only decides which logprobs belong to it."""
    raw = "".join(pieces)
    text = clean(raw)

    keep = kept_char_mask(raw)
    reconstructed = "".join(c for c, k in zip(raw, keep) if k)

    if reconstructed == text:
        kept, offset = [], 0
        for piece, lp in zip(pieces, token_logprobs):
            end = offset + len(piece)
            if any(keep[offset:end]):
                kept.append(lp)
            offset = end
        method = "char_mask"
    else:
        # The mask disagreed with clean() -- refuse to guess. Fall back to the
        # coarse but always-safe trim (everything up to the first stop marker)
        # and say so in the record, so a downstream consumer can see which
        # cases used which rule instead of silently averaging the wrong span.
        cut = len(raw)
        for marker in STOP_MARKERS:
            i = raw.find(marker)
            if i != -1:
                cut = min(cut, i)
        kept, offset = [], 0
        for piece, lp in zip(pieces, token_logprobs):
            if offset >= cut:
                break
            kept.append(lp)
            offset += len(piece)
        method = "stop_marker_fallback"

    diagnostics = {
        "n_gen_tokens_raw": len(token_logprobs),
        "n_gen_tokens_kept": len(kept),
        "logprob_trim_method": method,
    }
    return text, kept, diagnostics


def generate_with_logprobs(model, tokenizer, prompt, max_tokens: int, **kwargs):
    """Drop-in replacement for `generate(...)` that also returns the chosen
    token's log probability at every step.

    Text-identical to `generate()` BY CONSTRUCTION, not by luck: in the
    installed mlx_lm 0.31.3, `generate()` is literally
    `"".join(r.text for r in stream_generate(...))` -- it runs this exact
    loop and throws the distribution away. Same kwargs, same sampler, same
    decode path, so swapping to this changes what is RECORDED, never what is
    generated. (The repo's own byte-identity check against the pre-change
    results file is still worth running; this just says what it should find.)

    `resp.logprobs` is the full log-probability vector over the vocabulary at
    that step, so the chosen token's own logprob is `resp.logprobs[resp.token]`
    -- confirmed against mlx_lm 0.31.3's GenerationResponse docstring
    ("token (int): The next token" / "logprobs (mx.array): A vector of log
    probabilities").

    Returns (cleaned_text, kept_logprobs, diagnostics, raw_text)."""
    pieces: list[str] = []
    tok_lps: list[float] = []
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, **kwargs):
        pieces.append(resp.text)
        tok_lps.append(float(resp.logprobs[resp.token]))
    text, kept, diagnostics = clean_with_logprobs(pieces, tok_lps)
    return text, kept, diagnostics, "".join(pieces)


def confidence_fields(kept_logprobs: list[float]) -> dict:
    """The persisted confidence block. mean_logprob is the length-normalized
    signal calibration works from; sum_logprob is kept because it is the
    actual sequence log-likelihood and the two rank candidates differently
    (sum penalizes long answers, mean does not)."""
    n = len(kept_logprobs)
    return {
        "token_logprobs": kept_logprobs,
        "mean_logprob": (sum(kept_logprobs) / n) if n else None,
        "sum_logprob": sum(kept_logprobs) if n else None,
        "n_gen_tokens": n,
    }


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
