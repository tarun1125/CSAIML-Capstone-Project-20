"""Final cross-arm comparison, unified MLX-LM stack for the current numbers.

Reads directly from already-scored CSV/JSON outputs (never hardcodes numbers)
so these charts can never drift from what score_rag.py / score_finetuned_304.py /
score_finetuned_23db.py actually reported. Matches the "Academic Ink & Crimson"
palette already established in capstone_presentation.pptx / capstone_report.docx
(1B2836 ink navy, E7EAED panel grey, 9AA5AD grey = baseline, 3D5A73 blue = RAG,
7A2E2E crimson = fine-tuned) rather than introducing a new palette here.

Two figures:
  1. final_arm_comparison_n304.png -- the headline chart: baseline / RAG /
     fine-tuned(23-db retrained, the arm's best real number) on the full
     304-case test set.
  2. old_run_vs_current_run.png -- the requested "why did the story change"
     chart. This is a genuinely DIFFERENT comparison from re-slicing the
     current run down to the old 61 ids: it plots the ORIGINAL 61-case run
     exactly as it was measured at the time (baseline/RAG on the original
     Colab/HF fp16 stack -- rag/outputs/rag_vs_baseline_scores.csv, the
     pre-MLX file; fine-tuned unchanged, it was always MLX --
     data/finetuned_execution_results.json) against THIS current 304-case
     run (baseline/RAG/fine-tuned all on the unified MLX-LM stack). So the
     left-hand bars are literally the old headline numbers this project
     originally reported; the right-hand bars are today's numbers. Any
     movement in baseline/RAG reflects the stack switch *and* the larger
     test set together, not a same-stack re-slice -- the footnote says so
     explicitly so this is never misread as apples-to-apples.

Deliberately dropped from the previous version of this script: the in-scope
/ out-of-scope 6-db-adapter breakdown chart (superseded -- every fine-tuning
number now shown is the 23-db-retrained adapter, in-scope everywhere) and a
same-stack-only 61-vs-304 recut (that answers a different question -- "does
scale alone explain it" -- not what was asked here: old reported numbers vs
current reported numbers).
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

INK = "#1B2836"
MUTED = "#5B6570"
GREY_BASELINE = "#9AA5AD"
GREY_BASELINE_LIGHT = "#C7CDD2"
BLUE_RAG = "#3D5A73"
BLUE_RAG_LIGHT = "#8CA5B8"
CRIMSON = "#7A2E2E"
CRIMSON_LIGHT = "#C79494"
PANEL = "#E7EAED"


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- Load real, already-scored CURRENT full-304 numbers (unified MLX-LM) ---
mlx_rows = {r["Arm"]: r for r in read_csv_rows(ROOT / "rag" / "outputs" / "rag_vs_baseline_scores_mlx.csv")}
ft23_rows = {r["Arm"]: r for r in read_csv_rows(ROOT / "fine_tuning" / "outputs" / "finetuned_23db_scores.csv")}

baseline_acc_304 = float(mlx_rows["Qwen-Baseline(test-slice)"]["Accuracy"])
baseline_n_304 = mlx_rows["Qwen-Baseline(test-slice)"]["Correct"] + "/" + mlx_rows["Qwen-Baseline(test-slice)"]["Total"]

rag_acc_304 = float(mlx_rows["Qwen-RAG"]["Accuracy"])
rag_n_304 = mlx_rows["Qwen-RAG"]["Correct"] + "/" + mlx_rows["Qwen-RAG"]["Total"]

ft23_key = [k for k in ft23_rows if k.startswith("Fine-tuned(23db")][0]
ft_acc_304 = float(ft23_rows[ft23_key]["Accuracy"])
ft_n_304 = ft23_rows[ft23_key]["Correct"] + "/" + ft23_rows[ft23_key]["Total"]

# --- Load the ORIGINAL 61-case numbers, exactly as first reported ----------
# Baseline/RAG: the pre-MLX file (original Colab/HF fp16 stack). Fine-tuned:
# unchanged -- it was already MLX-LM from day one, so there is no "old stack"
# version of it to separately load.
old_rows = {r["Arm"]: r for r in read_csv_rows(ROOT / "rag" / "outputs" / "rag_vs_baseline_scores.csv")}
baseline_acc_61 = float(old_rows["Qwen-Baseline(test-slice)"]["Accuracy"])
baseline_n61_str = old_rows["Qwen-Baseline(test-slice)"]["Correct"] + "/" + old_rows["Qwen-Baseline(test-slice)"]["Total"]
rag_acc_61 = float(old_rows["Qwen-RAG"]["Accuracy"])
rag_n61_str = old_rows["Qwen-RAG"]["Correct"] + "/" + old_rows["Qwen-RAG"]["Total"]

ft_records = json.load(open(ROOT / "data" / "finetuned_execution_results.json"))
ft_records = ft_records["results"] if isinstance(ft_records, dict) and "results" in ft_records else ft_records
ft_c61 = sum(bool(r["execution_accuracy"]) for r in ft_records)
ft_n61 = len(ft_records)
ft_acc_61 = 100 * ft_c61 / ft_n61
ft_n61_str = f"{ft_c61}/{ft_n61}"

print(f"OLD (n=61, as originally reported) -- baseline {baseline_n61_str} ({baseline_acc_61}%), "
      f"RAG {rag_n61_str} ({rag_acc_61}%), fine-tuned {ft_n61_str} ({ft_acc_61:.2f}%)")
print(f"CURRENT (n=304, unified MLX-LM) -- baseline {baseline_n_304} ({baseline_acc_304}%), "
      f"RAG {rag_n_304} ({rag_acc_304}%), fine-tuned {ft_n_304} ({ft_acc_304}%)")

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# ============================================================================
# Figure 1: headline three-arm comparison, full 304, unified MLX-LM stack
# ============================================================================
labels = ["Baseline\n(zero-shot)", "RAG\n(K=10)", "Fine-tuned\n(23-db retrained)"]
values = [baseline_acc_304, rag_acc_304, ft_acc_304]
counts = [baseline_n_304, rag_n_304, ft_n_304]
colors = [GREY_BASELINE, BLUE_RAG, CRIMSON]

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(labels, values, color=colors, width=0.55)
for bar, v, c in zip(bars, values, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%\n({c})",
            ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
ax.set_ylabel("Execution accuracy (%)", fontsize=11)
ax.set_ylim(0, max(values) * 1.35)
ax.set_title("Final Comparison — Full 304-Case Test Set (Unified MLX-LM Stack)",
             fontsize=13, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax.set_axisbelow(True)
fig.text(0.5, 0.01,
          "All three arms measured on the identical 304-case test set, on the same MLX-LM inference stack on Apple Silicon —\n"
          "no cross-stack confound. RAG retrieves real few-shot examples at inference time; fine-tuning bakes patterns into weights.",
          ha="center", fontsize=8.5, color=MUTED, style="italic")
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(FIGURES / "final_arm_comparison_n304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'final_arm_comparison_n304.png'}")

# ============================================================================
# Figure 2: the ORIGINAL 61-case run (as first reported) vs THIS current
# 304-case run. Color = arm identity (fixed hue); fill weight + hatch =
# which run (secondary encoding, not a new hue).
# ============================================================================
arm_labels = ["Baseline\n(zero-shot)", "RAG\n(K=10)", "Fine-tuned"]
acc_old = [baseline_acc_61, rag_acc_61, ft_acc_61]
acc_new = [baseline_acc_304, rag_acc_304, ft_acc_304]
n_old = [baseline_n61_str, rag_n61_str, ft_n61_str]
n_new = [baseline_n_304, rag_n_304, ft_n_304]
dark_colors = [GREY_BASELINE, BLUE_RAG, CRIMSON]
light_colors = [GREY_BASELINE_LIGHT, BLUE_RAG_LIGHT, CRIMSON_LIGHT]

x = np.arange(3)
w = 0.32
fig, ax = plt.subplots(figsize=(8.5, 5.8))
bars_old = ax.bar(x - w / 2, acc_old, width=w, color=dark_colors,
                   label="Original 61-case run (as first reported)")
bars_new = ax.bar(x + w / 2, acc_new, width=w, color=light_colors, hatch="///", edgecolor=INK,
                   linewidth=0.4, label="Current 304-case run (unified MLX-LM)")

for bar, v, c in zip(bars_old, acc_old, n_old):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.1f}%\n({c})",
            ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
for bar, v, c in zip(bars_new, acc_new, n_new):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.1f}%\n({c})",
            ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(arm_labels, fontsize=11)
ax.set_ylabel("Execution accuracy (%)", fontsize=11)
ax.set_ylim(0, max(acc_old + acc_new) * 1.32)
ax.set_title("Original 61-Case Run vs. Current 304-Case Run", fontsize=13, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.text(0.5, 0.01,
          "Left bars are this project's original numbers exactly as first measured (baseline/RAG on the old Colab/HF fp16 stack;\n"
          "fine-tuning was always MLX-LM). Right bars are today's numbers, all three arms unified on MLX-LM, on the larger 304-case set.\n"
          "Fine-tuning beat RAG originally in part because RAG was measured on a noisier stack — see the stack-unification finding.",
          ha="center", fontsize=8.5, color=MUTED, style="italic")
plt.tight_layout(rect=[0, 0.11, 1, 1])
plt.savefig(FIGURES / "old_run_vs_current_run.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'old_run_vs_current_run.png'}")
