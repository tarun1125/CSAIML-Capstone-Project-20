# Execution scoring for the reranking experiment's §5 arms.
#
# Normalizes each arm's raw generations and runs them against the live Atlas
# cluster through evaluation/execute_queries.run_model -- the SAME scoring path
# score_rag.py uses, imported rather than reimplemented. A second scoring
# implementation is the one thing that could make this experiment's central
# comparison meaningless: if the arms were scored even slightly differently, the
# execution-accuracy column would be measuring the scorer, not the retrieval.
#
# Needs a real Atlas connection -- run locally, same as execute_gold.py.
#
# Usage:
#   python rag/score_rerank_arms.py             # every arm that has generations
#   python rag/score_rerank_arms.py k5 rerankQ_k5
#
# Output per arm: rag/data/qwen_rag_<arm>_normalized.json and
# rag/data/qwen_rag_<arm>_execution_results.json. eval_retrieval.py
# --with-execution reads the latter.

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from execute_queries import connect, run_model  # noqa: E402  reuse, don't reimplement

RAG_DATA = REPO_ROOT / "rag" / "data"
DATA = REPO_ROOT / "data"

# label -> raw generations file. The K=10 bi-encoder arm is deliberately absent:
# it already exists as rag/data/qwen_rag_execution_results_mlx.json, the
# canonical run this whole experiment is measured against. Regenerating it would
# risk moving the baseline underneath the comparison.
ARMS = {
    "k5": "qwen_rag_mlx_k5_results.json",
    "rerankQ_k5": "qwen_rag_mlx_rerankQ_k5_results.json",
    "rerankQQ_k5": "qwen_rag_mlx_rerankQQ_k5_results.json",
    "rerankQ_k10": "qwen_rag_mlx_rerankQ_k10_results.json",
}


def main():
    wanted = sys.argv[1:] or list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {list(ARMS)}")

    gold_raw = json.loads((DATA / "gold_results.json").read_text(encoding="utf-8"))
    gold_results = {str(r["id"]): r for r in gold_raw if r.get("status") == "PASS"}
    print(f"[score_rerank_arms] {len(gold_results)}/{len(gold_raw)} usable gold results")

    client = connect()
    summary = {}

    for arm in wanted:
        raw = RAG_DATA / ARMS[arm]
        if not raw.exists():
            print(f"[score_rerank_arms] SKIP {arm}: {raw.name} not generated yet")
            continue
        n_raw = len(json.loads(raw.read_text(encoding="utf-8")))
        if n_raw != 304:
            # Generation is resumable and checkpoints after every case, so a
            # partial file is a normal mid-run state, not corruption. Scoring it
            # anyway would silently produce an accuracy over a different n than
            # the other arms -- which is exactly the comparison this experiment
            # is trying to make, so refuse.
            print(f"[score_rerank_arms] SKIP {arm}: {n_raw}/304 cases generated so far")
            continue

        normalized = RAG_DATA / f"qwen_rag_{arm}_normalized.json"
        print(f"\n[score_rerank_arms] normalize {raw.name} -> {normalized.name}")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "normalize.py"), str(raw), str(normalized)],
            check=True, capture_output=True,
        )

        out = RAG_DATA / f"qwen_rag_{arm}_execution_results.json"
        print(f"[score_rerank_arms] executing arm {arm} against Atlas...")
        results = run_model(client, f"Qwen-RAG[{arm}]", normalized, out, gold_results)
        correct = sum(r.get("execution_accuracy") is True for r in results)
        summary[arm] = (correct, len(results))
        print(f"[score_rerank_arms] {arm}: {correct}/{len(results)} = "
              f"{correct / len(results) * 100:.2f}%")

    print("\n" + "=" * 60)
    for arm, (correct, total) in summary.items():
        print(f"  {arm:<16} {correct:>3}/{total} = {correct / total * 100:5.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
