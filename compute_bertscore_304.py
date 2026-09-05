"""
BERTScore on the current, unified-MLX-LM 304-case test set (baseline / RAG-FK / fine-tuned-1000iter).
Mirrors the methodology in mongodb_nl_to_sql_1.ipynb cell 9 exactly (lang="en", rescale_with_baseline=True).
"""
import json
from pathlib import Path
import pandas as pd
from bert_score import score

ROOT = Path("/home/claude/metrics_304/CSAIML-Capstone-Project-20")
REFERENCE = ROOT / "data/reference_queries.json"
OUTPUT = Path("/home/claude/metrics_304/bertscore_scores_304.csv")

ARMS = {
    "Baseline": ROOT / "data/qwen_baseline_mlx_testslice_normalized.json",
    "RAG (K=10, FK)": ROOT / "rag/data/qwen_rag_mlx_fk_normalized.json",
    "Fine-tuned (23-db, 1000iter)": ROOT / "data/finetuned_full304_23db_1000iter_normalized.json",
}


def load_reference(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(x["id"]): x["normalized_query"] for x in data}


def load_model(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(row["id"]): row["normalized_query"] for row in data}


def evaluate(model_name, predictions, reference):
    shared_ids = sorted(set(reference) & set(predictions))
    refs = [reference[qid] for qid in shared_ids]
    preds = [predictions[qid] for qid in shared_ids]

    print(f"\nEvaluating {model_name}... (n={len(shared_ids)})")
    P, R, F1 = score(preds, refs, lang="en", rescale_with_baseline=True)

    rows = []
    for i, qid in enumerate(shared_ids):
        rows.append({
            "id": qid,
            "arm": model_name,
            "precision": round(P[i].item(), 4),
            "recall": round(R[i].item(), 4),
            "f1": round(F1[i].item(), 4),
        })
    print(f"Average F1 ({model_name}) = {F1.mean().item():.4f}")
    return rows


reference = load_reference(REFERENCE)
rows = []
for name, path in ARMS.items():
    pred = load_model(path)
    rows.extend(evaluate(name, pred, reference))

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print("\nSaved:", OUTPUT)
print(df.groupby("arm")[["precision", "recall", "f1"]].mean().round(4))
