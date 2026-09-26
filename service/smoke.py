# Phase 6 measurements against the LIVE endpoint (docs/AZURE-PLAN.md).
#
#   python service/smoke.py smoke --n 50       # first 50 held-out cases, pre-declared slice
#   python service/smoke.py cold --runs 3      # first request after scale-to-zero, 3 times
#
# URL and API key come from Azure (az CLI, names from azure.env), so the key is
# read from Key Vault into memory and never written anywhere.
#
# SMOKE: replays rag/data/rag_test.json[:n] in order -- a slice fixed in the plan
# before any run, not questions picked afterwards. Requests are spaced to stay
# under the service's own rate limit, otherwise this measures the limiter.
# Records, per case: client wall time, the server's per-stage latency_ms, the
# query, and results_match(rows, gold) -- the benchmark's own comparison --
# so the smoke run is also a small SERVED accuracy check. A case whose rows
# were capped by MAX_ROWS is recorded as not comparable rather than wrong.
#
# COLD: polls until the app has zero replicas (scale-to-zero after its idle
# cooldown), then times one request. Each run can take several minutes of
# waiting, most of it the cooldown.
#
# Output: results/azure_smoke.json / results/azure_cold_start.json, each with a
# manifest (image, revision, deployment settings).

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from atlas_env import load_env_file  # noqa: E402
from execute_queries import results_match  # noqa: E402  the benchmark's comparison
from run_manifest import write_manifest  # noqa: E402

RESULTS = REPO_ROOT / "results"


def az(*args) -> str:
    return subprocess.run(["az", *args, "-o", "tsv"], capture_output=True, text=True, check=True).stdout.strip()


def target() -> dict:
    env = load_env_file(REPO_ROOT / "azure.env")
    rg, app, kv = env["RG"], env["APP"], env["KV"]
    return {
        "rg": rg, "app": app,
        "url": "https://" + az("containerapp", "show", "-g", rg, "-n", app,
                               "--query", "properties.configuration.ingress.fqdn"),
        "key": az("keyvault", "secret", "show", "--vault-name", kv, "-n", "service-api-key", "--query", "value"),
        "revision": az("containerapp", "show", "-g", rg, "-n", app, "--query", "properties.latestRevisionName"),
        "image": az("containerapp", "show", "-g", rg, "-n", app,
                    "--query", "properties.template.containers[0].image"),
        "env_settings": az("containerapp", "show", "-g", rg, "-n", app, "--query",
                           "properties.template.containers[0].env[?!secretRef].[name,value]"),
    }


def ask(t: dict, question: str, timeout: float = 240) -> tuple[float, dict]:
    req = urllib.request.Request(
        t["url"] + "/query", method="POST",
        data=json.dumps({"question": question}).encode(),
        headers={"x-api-key": t["key"], "content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def smoke(t: dict, n: int, spacing: float) -> None:
    cases = json.loads((REPO_ROOT / "rag" / "data" / "rag_test.json").read_text(encoding="utf-8"))[:n]
    gold = {str(r["id"]): r for r in json.loads((REPO_ROOT / "data" / "gold_results.json").read_text())
            if r.get("status") == "PASS"}
    rows = []
    for i, case in enumerate(cases, 1):
        t_next = time.monotonic() + spacing
        try:
            wall, r = ask(t, case["question"])
            g = gold.get(str(case["id"]))
            if not r.get("safe") or r.get("error"):
                correct = False
            elif r.get("truncated"):
                correct = None  # rows capped by MAX_ROWS: not comparable to the full gold result
            else:
                correct = results_match(r.get("rows"), g["result"]) if g else None
            rows.append({"id": case["id"], "gold_database": case["database"], "database": r["database"],
                         "database_match": r["database"] == case["database"], "safe": r["safe"],
                         "error": r.get("error"), "truncated": r.get("truncated"), "correct": correct,
                         "query": r.get("query"), "client_s": round(wall, 3), "latency_ms": r["latency_ms"],
                         "tokens": r.get("tokens"), "model": r.get("model")})
            print(f"[{i}/{len(cases)}] {str(case['id']):28s} {wall:5.2f}s db_ok={rows[-1]['database_match']} "
                  f"safe={r['safe']} correct={correct}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": case["id"], "failed_request": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"[{i}/{len(cases)}] {case['id']} REQUEST FAILED: {exc}", flush=True)
        time.sleep(max(0.0, t_next - time.monotonic()))

    ok = [r for r in rows if "failed_request" not in r]
    stages = ["embed", "retrieve", "generate", "guard", "execute", "total"]
    summary = {
        "n_cases": len(rows), "n_ok_requests": len(ok),
        "client_s": {"p50": pct([r["client_s"] for r in ok], 50), "p95": pct([r["client_s"] for r in ok], 95),
                     "mean": round(statistics.mean(r["client_s"] for r in ok), 3) if ok else None},
        "server_ms": {s: {"p50": pct([r["latency_ms"][s] for r in ok if s in r["latency_ms"]], 50),
                          "p95": pct([r["latency_ms"][s] for r in ok if s in r["latency_ms"]], 95)}
                      for s in stages},
        "database_match": sum(r["database_match"] for r in ok),
        "safe": sum(r["safe"] for r in ok),
        "correct": sum(r["correct"] is True for r in ok),
        "not_comparable_truncated": sum(r["correct"] is None for r in ok),
        "prompt_tokens": sum((r.get("tokens") or {}).get("prompt_tokens") or 0 for r in ok),
        "completion_tokens": sum((r.get("tokens") or {}).get("completion_tokens") or 0 for r in ok),
    }
    print(json.dumps(summary, indent=2))
    out = RESULTS / "azure_smoke.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    write_manifest(out, script=__file__, stage="Phase 6 smoke benchmark (live endpoint)",
                   url=t["url"], revision=t["revision"], image=t["image"], settings=t["env_settings"],
                   slice="rag/data/rag_test.json[:%d]" % n, spacing_s=spacing)
    print("wrote", out)


def cold(t: dict, runs: int, poll_s: float, max_wait_s: float) -> None:
    results = []
    for i in range(1, runs + 1):
        waited, t0 = 0.0, time.monotonic()
        while True:
            n = az("containerapp", "replica", "list", "-g", t["rg"], "-n", t["app"], "--query", "length(@)")
            if n in ("", "0"):
                break
            if time.monotonic() - t0 > max_wait_s:
                raise SystemExit(f"still {n} replica(s) after {max_wait_s}s -- is something calling the app?")
            time.sleep(poll_s)
        waited = round(time.monotonic() - t0, 1)
        wall, r = ask(t, "How many singers do we have?")
        results.append({"run": i, "waited_for_zero_s": waited, "client_s": round(wall, 2),
                        "server_total_ms": r["latency_ms"]["total"], "latency_ms": r["latency_ms"]})
        print(f"cold run {i}: {wall:.1f}s end to end (server pipeline {r['latency_ms']['total']/1000:.1f}s; "
              f"waited {waited:.0f}s for zero replicas)", flush=True)
    warm_wall, _ = ask(t, "How many singers do we have?")
    out = RESULTS / "azure_cold_start.json"
    payload = {"runs": results, "warm_follow_up_s": round(warm_wall, 2),
               "cold_s": {"min": min(r["client_s"] for r in results), "max": max(r["client_s"] for r in results),
                          "median": statistics.median(r["client_s"] for r in results)}}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest(out, script=__file__, stage="Phase 6 cold start (scale 0 -> 1)",
                   url=t["url"], revision=t["revision"], image=t["image"])
    print(json.dumps(payload["cold_s"]), "warm:", payload["warm_follow_up_s"], "\nwrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--n", type=int, default=50)
    s.add_argument("--spacing", type=float, default=6.5, help="seconds between requests (limit is 10/min)")
    c = sub.add_parser("cold"); c.add_argument("--runs", type=int, default=3)
    c.add_argument("--poll", type=float, default=30); c.add_argument("--max-wait", type=float, default=1800)
    args = ap.parse_args()
    t = target()
    print(f"target {t['url']} revision={t['revision']} image={t['image']}", flush=True)
    smoke(t, args.n, args.spacing) if args.mode == "smoke" else cold(t, args.runs, args.poll, args.max_wait)


if __name__ == "__main__":
    main()
