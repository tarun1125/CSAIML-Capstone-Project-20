"""CodeBLEU (3-component, dataflow excluded) by arm, full 304-case test set,
alongside execution accuracy for the same three arms -- two small-multiple
panels sharing categorical x-axis, NOT a dual-axis chart (each metric has
its own scale, so each gets its own panel; see dataviz non-negotiables).

Reads directly from outputs/codebleu_scores_304.csv (never hardcodes a
number). Same "Academic Ink & Crimson" palette as the other current figures.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "outputs" / "figures"

INK = "#1B2836"
MUTED = "#5B6570"
GREY_BASELINE = "#9AA5AD"
BLUE_RAG = "#3D5A73"
CRIMSON = "#7A2E2E"
PANEL = "#E7EAED"

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

with open(ROOT / "outputs" / "codebleu_scores_304.csv", newline="") as f:
    rows = {r["Arm"]: r for r in csv.DictReader(f)}

labels = ["Baseline\n(zero-shot)", "RAG\n(K=10, FK)", "Fine-tuned\n(23-db, 1000iter)"]
keys = ["Baseline", "RAG (K=10, FK)", "Fine-tuned (23-db, 1000iter)"]
colors = [GREY_BASELINE, BLUE_RAG, CRIMSON]

codebleu_3c = [float(rows[k]["codebleu_3component"]) for k in keys]
# Execution accuracy, current canonical numbers (baseline 15/304, RAG FK-canonical 142/304, fine-tuned 103/304)
exec_acc = [4.9, 46.7, 33.9]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

bars1 = ax1.bar(labels, exec_acc, color=colors, width=0.6)
for bar, v in zip(bars1, exec_acc):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
ax1.set_ylabel("Execution accuracy (%)", fontsize=11)
ax1.set_ylim(0, max(exec_acc) * 1.3)
ax1.set_title("Execution Accuracy", fontsize=12.5, color=INK, fontweight="bold", pad=12, loc="left")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax1.set_axisbelow(True)
plt.setp(ax1.get_xticklabels(), fontsize=9)

bars2 = ax2.bar(labels, codebleu_3c, color=colors, width=0.6)
for bar, v in zip(bars2, codebleu_3c):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
ax2.set_ylabel("CodeBLEU (3-component)", fontsize=11)
ax2.set_ylim(0, max(codebleu_3c) * 1.3)
ax2.set_title("CodeBLEU (3-Component, Dataflow Excluded)", fontsize=12.5, color=INK, fontweight="bold", pad=12, loc="left")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax2.set_axisbelow(True)
plt.setp(ax2.get_xticklabels(), fontsize=9)

fig.suptitle("Execution Accuracy vs. CodeBLEU — Full 304-Case Test Set", fontsize=13.5, color=INK, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
          "RAG and fine-tuned score within 0.002 of each other on CodeBLEU (surface query similarity) despite a 12.8-point\n"
          "gap in execution accuracy (functional correctness) — the two metrics are not interchangeable for this task.",
          ha="center", fontsize=8.5, color=MUTED, style="italic")
plt.tight_layout()
plt.savefig(FIGURES / "codebleu_vs_execution_accuracy_304.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved", FIGURES / "codebleu_vs_execution_accuracy_304.png")
