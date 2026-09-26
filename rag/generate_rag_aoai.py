# RAG generation through Azure OpenAI -- arm G1 of docs/AZURE-PLAN.md, Phase 4.
#
# The generator changes; NOTHING else does. Same prompts file (A0's
# rag/data/rag_prompts.json by default), same system/user split as
# rag/generate_rag_mlx.py, same clean() post-processing, same scoring path
# afterwards (rag/score_rerank_arms.py). Retrieval is held fixed, so G1 vs A0
# answers exactly one question: what does a larger hosted model buy here?
#
#   python rag/generate_rag_aoai.py --output rag/data/aoai_gpt4o_k10_results.json
#   python rag/generate_rag_aoai.py --limit 50 --output rag/data/aoai_gpt4o_k10_det_a.json   # determinism
#
# Reads AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_API_KEY
# from the environment or the gitignored azure.env. Uses Azure's v1 endpoint
# (<endpoint>/openai/v1/), which takes no api-version string.
#
# DECODING: temperature 0 and a fixed seed. A hosted model at temperature 0 is
# NOT guaranteed deterministic (batching, hardware, backend updates), which is
# why the plan runs a determinism check and reports the flip rate -- the same
# way the README reports the oracle's. system_fingerprint is recorded per case
# so a backend change mid-run is visible after the fact.
#
# RESUMABLE like generate_rag_mlx.py, with one fix that script lacks: each
# record carries the prompts file it came from, and resuming against a
# different prompts file is refused. Otherwise two arms could silently mix in
# one output file.
#
# COST: every record keeps the API's own token counts. The dollar figure in the
# manifest is an ESTIMATE from --price-in / --price-out (USD per 1M tokens) --
# check them against the Azure pricing page on the day; the token counts are
# the ground truth.

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fine_tuning"))
from atlas_env import load_env_file  # noqa: E402
from generation_utils import clean  # noqa: E402  the canonical post-processing, MLX-free since ef8f032
from run_manifest import read_manifest, write_manifest  # noqa: E402

log = logging.getLogger("rag.generate_rag_aoai")


def load_settings() -> dict:
    import os
    values = {}
    env_file = REPO_ROOT / "azure.env"
    if env_file.exists():
        values = load_env_file(env_file)
    get = lambda k: os.environ.get(k) or values.get(k)  # noqa: E731
    cfg = {k: get(k) for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY")}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(f"missing settings: {missing} (environment or azure.env)")
    return cfg


def load_checkpoint(output_path: Path, prompts_rel: str) -> dict:
    if not output_path.exists():
        return {}
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    sources = {r.get("prompts_file") for r in existing}
    if sources != {prompts_rel}:
        raise SystemExit(
            f"{output_path} was generated from {sorted(map(str, sources))}, not {prompts_rel}. "
            "Refusing to resume into it -- use a fresh --output."
        )
    return {r["id"]: r for r in existing}


def save_checkpoint(output_path: Path, done: dict, cases: list) -> None:
    ordered = [done[c["id"]] for c in cases if c["id"] in done]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-path", default=str(REPO_ROOT / "rag" / "data" / "rag_prompts.json"))
    parser.add_argument("--output", required=True,
                        help="results file; required on purpose, so no run can land on another's file")
    parser.add_argument("--limit", type=int, default=None, help="first N cases only (determinism check)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--price-in", type=float, default=2.50, help="USD per 1M input tokens (estimate)")
    parser.add_argument("--price-out", type=float, default=10.00, help="USD per 1M output tokens (estimate)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from openai import BadRequestError, OpenAI  # noqa: PLC0415  only this arm needs the SDK

    cfg = load_settings()
    client = OpenAI(base_url=cfg["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
                    api_key=cfg["AZURE_OPENAI_API_KEY"], max_retries=6, timeout=60)

    prompts_path = Path(args.prompts_path).resolve()
    prompts_rel = str(prompts_path.relative_to(REPO_ROOT)) if prompts_path.is_relative_to(REPO_ROOT) else str(prompts_path)
    cases = json.loads(prompts_path.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]
    output_path = Path(args.output)
    done = load_checkpoint(output_path, prompts_rel)
    log.info("%d cases from %s, %d already done -> %s (deployment=%s, T=%s, seed=%s)",
             len(cases), prompts_rel, len(done), output_path, cfg["AZURE_OPENAI_DEPLOYMENT"],
             args.temperature, args.seed)

    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue
        t0 = time.monotonic()
        record = {
            "id": case["id"],
            "question": case["question"],
            "database": case["gold_database"],
            "predicted_database": case["predicted_database"],
            "database_match": case["database_match"],
            "complexity": case.get("complexity"),
            "prompts_file": prompts_rel,
        }
        try:
            r = client.chat.completions.create(
                model=cfg["AZURE_OPENAI_DEPLOYMENT"],
                messages=[{"role": "system", "content": case["system_prompt"]},
                          {"role": "user", "content": case["question"]}],
                temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
            )
            raw = r.choices[0].message.content or ""
            record.update({
                "generated_query": clean(raw),
                "raw_output": raw,
                "finish_reason": r.choices[0].finish_reason,
                "model": r.model,
                "system_fingerprint": r.system_fingerprint,
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            })
        except BadRequestError as exc:
            # Content-filter refusals and the like. Recorded, not fatal: an empty
            # query scores as a failure, which is what it is.
            record.update({"generated_query": "", "raw_output": "", "error": str(exc)[:500],
                           "prompt_tokens": 0, "completion_tokens": 0})
            log.warning("[%d/%d] id=%s request rejected: %s", i, len(cases), case["id"], str(exc)[:160])
        record["latency_s"] = round(time.monotonic() - t0, 3)
        done[case["id"]] = record
        save_checkpoint(output_path, done, cases)
        log.info("[%d/%d] id=%s (%.1fs, %s/%s tok) -> %s", i, len(cases), case["id"], record["latency_s"],
                 record.get("prompt_tokens"), record.get("completion_tokens"),
                 record["generated_query"][:80].replace("\n", " "))

    results = [done[c["id"]] for c in cases if c["id"] in done]
    tok_in = sum(r.get("prompt_tokens") or 0 for r in results)
    tok_out = sum(r.get("completion_tokens") or 0 for r in results)
    est = (tok_in * args.price_in + tok_out * args.price_out) / 1e6
    latencies = sorted(r["latency_s"] for r in results)
    log.info("done: %d cases, %d in / %d out tokens, est. $%.3f, %.1fs this run",
             len(results), tok_in, tok_out, est, time.monotonic() - run_start)

    prompt_manifest = read_manifest(prompts_path) or {}
    manifest = write_manifest(
        output_path, script=__file__, arm="rag-aoai", generator="azure-openai",
        deployment=cfg["AZURE_OPENAI_DEPLOYMENT"],
        model_versions=sorted({r.get("model") for r in results if r.get("model")}),
        system_fingerprints=sorted({r.get("system_fingerprint") for r in results if r.get("system_fingerprint")}),
        endpoint=cfg["AZURE_OPENAI_ENDPOINT"], api="openai/v1",
        temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens,
        n_cases=len(results), n_errors=sum("error" in r for r in results),
        prompt_tokens=tok_in, completion_tokens=tok_out,
        est_cost_usd=round(est, 4), price_in_per_m=args.price_in, price_out_per_m=args.price_out,
        latency_p50_s=latencies[len(latencies) // 2] if latencies else None,
        latency_p95_s=latencies[int(len(latencies) * 0.95)] if latencies else None,
        prompts_file=prompts_rel, prompts_manifest=prompt_manifest or None,
        top_k=prompt_manifest.get("top_k"), include_fk=prompt_manifest.get("include_fk"),
    )
    log.info("manifest -> %s", manifest)
    log.info("Next: add '<label>': '%s' to ARMS in rag/score_rerank_arms.py, then: "
             "python rag/score_rerank_arms.py <label>", output_path.name)


if __name__ == "__main__":
    sys.exit(main())
