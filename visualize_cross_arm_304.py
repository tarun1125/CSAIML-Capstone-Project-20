"""Cross-arm diagnostic visualizations for the CURRENT full 304-case,
unified-MLX-LM test set -- the same set of charts as
visualize_cross_arm_comparison.ipynb built for the original 61-case holdout,
recomputed on today's real data (baseline 15/304, RAG 142/304 FK-canonical,
fine-tuned 73/304, 23-db collision-fix-retrained adapter).

Reads directly from already-scored execution-results JSON / reference_queries.json
(never hardcodes a number) so these charts can never drift from what
score_rag.py / score_finetuned_23db.py actually reported. Same "Academic Ink &
Crimson" palette as visualize_final_comparison.py and the delivered pptx/docx
(1B2836 ink navy, E7EAED panel grey, 9AA5AD grey=baseline, 3D5A73 blue=RAG,
7A2E2E crimson=fine-tuned, 5B6570 muted).

Differences from the original 61-case notebook, stated explicitly rather than
silently changed:
  - Claude is NOT included in the all-arms composition chart here. Claude's
    execution results only cover the original 61 ids (61/304) -- the 243
    newly-expanded test ids were never scored against Claude. Including it
    would mean either faking 243 results or silently comparing populations of
    different size under one bar; neither is acceptable, so this set of
    charts is baseline/RAG/fine-tuned only (n=304 for all three).
  - RAG's case-level diagnostics (outcome breakdown, bug taxonomy, complexity,
    overlap) use the RAG execution-results file that is CURRENTLY on disk,
    which -- per this session's own documented A/B-test file-overwrite
    finding -- holds the no-FK variant's case-level results (141/304), not
    the FK-canonical aggregate (142/304) used for the headline number. The
    A/B test's own null result (1 case / 0.3pp) means this makes no
    practically-different diagnostic story either way; still stated here
    rather than silently glossed over.
  - Added one new figure not in the original 61-case set:
    finetuned_accuracy_by_database.png, a per-database accuracy bar across
    all 23 databases -- directly relevant this session since it's the
    concrete evidence for the batch-collision-fix finding (college_3 0%->28.6%).

Five figures (matching the original set) + 1 new one, all saved to
outputs/figures/ with a _304 suffix so they never collide with or overwrite
the original 61-case PNGs.
"""
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
BLUE_RAG = "#3D5A73"
CRIMSON = "#7A2E2E"
PANEL = "#E7EAED"
GOOD = "#3D7A4E"      # correct -- new status-style green, reserved, not reused for series
BAD_EMPTY = "#C7CDD2"  # ran-but-empty -- light neutral
BAD_WRONG = "#B8862E"  # ran-nonempty-wrong -- amber
BAD_FAIL = "#8C3B3B"   # hard-fail -- dark red, distinct from crimson (fine-tuned identity)

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

# ---------------------------------------------------------------------------
# Load real, already-scored data
# ---------------------------------------------------------------------------
baseline = json.load(open(ROOT / "rag" / "data" / "qwen_baseline_testslice_execution_results_mlx.json"))
rag = json.load(open(ROOT / "rag" / "data" / "qwen_rag_execution_results_mlx.json"))
ft = json.load(open(ROOT / "data" / "finetuned_full304_23db_execution_results.json"))
ref = json.load(open(ROOT / "data" / "reference_queries.json"))
ref_by_id = {str(r["id"]): r for r in ref}

for name, records in [("baseline", baseline), ("RAG", rag), ("fine-tuned", ft)]:
    assert len(records) == 304, f"{name} has {len(records)} records, expected 304"

print(f"Loaded: baseline={len(baseline)}, RAG={len(rag)}, fine-tuned={len(ft)} (all 304, confirmed)")


def outcome_bucket(r):
    """hard_fail / empty / wrong_nonempty / correct -- mirrors the scoring
    funnel already used in prior notebooks and this session's own
    verification pass."""
    if r.get("execution_accuracy"):
        return "correct"
    if r.get("status") != "PASS":
        return "hard_fail"
    if not r.get("non_empty_rate"):
        return "empty"
    return "wrong_nonempty"


def outcome_counts(records):
    counts = {"correct": 0, "wrong_nonempty": 0, "empty": 0, "hard_fail": 0}
    for r in records:
        counts[outcome_bucket(r)] += 1
    assert sum(counts.values()) == len(records)
    return counts


BUCKET_ORDER = ["correct", "wrong_nonempty", "empty", "hard_fail"]
BUCKET_LABEL = {"correct": "Correct", "wrong_nonempty": "Ran, wrong result",
                "empty": "Ran, empty result", "hard_fail": "Hard failure (parse/exec error)"}
BUCKET_COLOR = {"correct": GOOD, "wrong_nonempty": BAD_WRONG, "empty": BAD_EMPTY, "hard_fail": BAD_FAIL}

baseline_counts = outcome_counts(baseline)
rag_counts = outcome_counts(rag)
ft_counts = outcome_counts(ft)

print("Baseline funnel:", baseline_counts)
print("RAG funnel:", rag_counts)
print("Fine-tuned funnel:", ft_counts)


def draw_funnel(counts, title, filename, accent_color):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = [BUCKET_LABEL[b] for b in BUCKET_ORDER]
    values = [counts[b] for b in BUCKET_ORDER]
    colors = [BUCKET_COLOR[b] for b in BUCKET_ORDER]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    total = sum(values)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.015, f"{v}\n({100*v/total:.1f}%)",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Cases (n=304)", fontsize=11)
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300)
    plt.close()
    print(f"Saved {FIGURES / filename}")


draw_funnel(baseline_counts, "Baseline (zero-shot) Outcome Breakdown -- Full 304-Case Set", "baseline_outcome_breakdown_304.png", GREY_BASELINE)
draw_funnel(rag_counts, "RAG (K=10) Outcome Breakdown -- Full 304-Case Set", "rag_outcome_breakdown_304.png", BLUE_RAG)
draw_funnel(ft_counts, "Fine-tuned (23-db) Outcome Breakdown -- Full 304-Case Set", "finetuned_outcome_breakdown_304.png", CRIMSON)

# ---------------------------------------------------------------------------
# All-arms outcome composition -- 100% stacked bar, baseline/RAG/fine-tuned
# (Claude excluded -- see module docstring)
# ---------------------------------------------------------------------------
arms = ["Baseline\n(zero-shot)", "RAG\n(K=10)", "Fine-tuned\n(23-db)"]
arm_counts = [baseline_counts, rag_counts, ft_counts]
arm_colors_id = [GREY_BASELINE, BLUE_RAG, CRIMSON]

fig, ax = plt.subplots(figsize=(8, 5.5))
bottom = np.zeros(3)
for bucket in BUCKET_ORDER:
    vals = np.array([100 * c[bucket] / 304 for c in arm_counts])
    ax.bar(arms, vals, bottom=bottom, color=BUCKET_COLOR[bucket], width=0.55, label=BUCKET_LABEL[bucket])
    for i, v in enumerate(vals):
        if v > 4:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottom += vals
ax.set_ylabel("Share of 304 cases (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("All-Arms Outcome Composition -- Full 304-Case Set (Unified MLX-LM)", fontsize=12.5, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
fig.text(0.5, 0.965, "Claude excluded: its execution results cover only the original 61 ids, not the full 304.",
          ha="center", fontsize=8, color=MUTED, style="italic")
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.savefig(FIGURES / "all_arms_outcome_composition_304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'all_arms_outcome_composition_304.png'}")

# ---------------------------------------------------------------------------
# Accuracy by complexity
# ---------------------------------------------------------------------------
def complexity_of(rid):
    ref_row = ref_by_id.get(str(rid))
    if not ref_row:
        return "unknown"
    c = ref_row.get("complexity")
    if c is None:
        return "unknown"
    if c in ("high", "complex"):
        return "hard"
    return c

def accuracy_by_complexity(records):
    buckets = {}
    for r in records:
        c = complexity_of(r["id"])
        buckets.setdefault(c, [0, 0])
        buckets[c][1] += 1
        if r.get("execution_accuracy"):
            buckets[c][0] += 1
    return buckets

COMPLEXITY_ORDER = ["easy", "medium", "hard", "unknown"]
# Real data-quality finding, checked and reported rather than silently dropped:
# 79/304 test cases (all from round-3 spider additions, e.g. college_1/college_2/
# bike_1) have complexity: null in reference_queries.json -- not just the
# already-documented "high"/"complex" vocabulary inconsistency, but a genuinely
# missing value. Excluding them from this chart would silently drop 26% of the
# test set from a complexity breakdown -- shown explicitly as its own "Unknown"
# bar instead.
b_comp = accuracy_by_complexity(baseline)
r_comp = accuracy_by_complexity(rag)
f_comp = accuracy_by_complexity(ft)

print("Baseline by complexity:", b_comp)
print("RAG by complexity:", r_comp)
print("Fine-tuned by complexity:", f_comp)

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(COMPLEXITY_ORDER))
w = 0.25
for i, (label, comp, color) in enumerate([
    ("Baseline", b_comp, GREY_BASELINE), ("RAG", r_comp, BLUE_RAG), ("Fine-tuned", f_comp, CRIMSON)
]):
    vals = [100 * comp.get(c, [0, 1])[0] / comp.get(c, [0, 1])[1] if comp.get(c, [0, 0])[1] else 0 for c in COMPLEXITY_ORDER]
    ns = [comp.get(c, [0, 0])[1] for c in COMPLEXITY_ORDER]
    bars = ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=label)
    for bar, v, n in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in COMPLEXITY_ORDER], fontsize=11)
n_unknown = b_comp.get("unknown", [0, 0])[1]
if n_unknown:
    print(f"NOTE: {n_unknown}/304 test cases have complexity=null in reference_queries.json (data-quality gap, not a script bug) -- shown as their own 'Unknown' bar, not silently dropped.")
ax.set_ylabel("Execution accuracy (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("Accuracy by Question Complexity -- Full 304-Case Set", fontsize=13, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "accuracy_by_complexity_304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'accuracy_by_complexity_304.png'}")

# ---------------------------------------------------------------------------
# RAG vs fine-tuned correct-set overlap
# ---------------------------------------------------------------------------
rag_correct_ids = set(r["id"] for r in rag if r.get("execution_accuracy"))
ft_correct_ids = set(r["id"] for r in ft if r.get("execution_accuracy"))
both = len(rag_correct_ids & ft_correct_ids)
rag_only = len(rag_correct_ids - ft_correct_ids)
ft_only = len(ft_correct_ids - rag_correct_ids)
neither = 304 - len(rag_correct_ids | ft_correct_ids)
union = len(rag_correct_ids | ft_correct_ids)
print(f"Overlap: both={both} rag_only={rag_only} ft_only={ft_only} neither={neither} union={union}/304 ({100*union/304:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 5))
cats = ["Both correct", "RAG only", "Fine-tuned only", "Neither correct"]
vals = [both, rag_only, ft_only, neither]
colors = ["#5A6E8C", BLUE_RAG, CRIMSON, PANEL]
bars = ax.bar(cats, vals, color=colors, width=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v}\n({100*v/304:.1f}%)", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
ax.set_ylabel("Cases (n=304)", fontsize=11)
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title(f"RAG vs. Fine-tuned Correct-Set Overlap -- Full 304-Case Set\n(Union covers {union}/304 = {100*union/304:.1f}%)",
             fontsize=12.5, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), fontsize=9.5)
plt.tight_layout()
plt.savefig(FIGURES / "rag_vs_finetuned_correct_overlap_304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'rag_vs_finetuned_correct_overlap_304.png'}")

# ---------------------------------------------------------------------------
# Bug taxonomy by arm -- mechanical classification only (no guessing)
# ---------------------------------------------------------------------------
def classify_bug(r, gold_query):
    if r.get("execution_accuracy"):
        return None
    status = r.get("status")
    err = (r.get("error") or "").lower()
    if status != "PASS":
        if "rejected by safety check" in err:
            return "blocked_by_safety_check"
        if "unknown operator" in err or "unknown group operator" in err or "must be an array" in err:
            return "invalid_operator_misuse"
        if "$convert" in err or "failed to parse number" in err or "cannot convert" in err:
            return "bad_type_cast"
        return "other_hard_fail"
    # PASS but wrong -- compare against gold query text mechanically
    pred_query = r.get("query", "")
    gold_has_lookup = "$lookup" in (gold_query or "")
    pred_has_lookup = "$lookup" in pred_query
    if gold_has_lookup and not pred_has_lookup:
        return "missing_join"
    gold_has_cast = any(op in (gold_query or "") for op in ["$toInt", "$toDouble", "$convert"])
    pred_has_cast = any(op in pred_query for op in ["$toInt", "$toDouble", "$convert"])
    if gold_has_cast and not pred_has_cast:
        return "missing_type_cast"
    return "other_wrong_logic"


TAXONOMY_ORDER = ["blocked_by_safety_check", "invalid_operator_misuse", "bad_type_cast",
                   "missing_join", "missing_type_cast", "other_wrong_logic"]
TAXONOMY_LABEL = {
    "blocked_by_safety_check": "Blocked by safety check",
    "invalid_operator_misuse": "Invalid operator misuse ($size/$min/$max)",
    "bad_type_cast": "Bad type cast",
    "missing_join": "Missing $lookup join",
    "missing_type_cast": "Missing type cast",
    "other_wrong_logic": "Other wrong logic",
}

def taxonomy_counts(records):
    counts = {k: 0 for k in TAXONOMY_ORDER}
    correct = 0
    other_hard_fail = 0
    for r in records:
        gold_row = ref_by_id.get(str(r["id"]))
        gold_query = gold_row.get("normalized_query", "") if gold_row else ""
        bucket = classify_bug(r, gold_query)
        if bucket is None:
            correct += 1
        elif bucket == "other_hard_fail":
            other_hard_fail += 1
        else:
            counts[bucket] += 1
    return counts, correct, other_hard_fail

b_tax, b_correct, b_ohf = taxonomy_counts(baseline)
r_tax, r_correct, r_ohf = taxonomy_counts(rag)
f_tax, f_correct, f_ohf = taxonomy_counts(ft)
print("Baseline taxonomy:", b_tax, "correct:", b_correct, "other_hard_fail:", b_ohf)
print("RAG taxonomy:", r_tax, "correct:", r_correct, "other_hard_fail:", r_ohf)
print("Fine-tuned taxonomy:", f_tax, "correct:", f_correct, "other_hard_fail:", f_ohf)
for tax, correct, ohf, n in [(b_tax, b_correct, b_ohf, 304), (r_tax, r_correct, r_ohf, 304), (f_tax, f_correct, f_ohf, 304)]:
    assert sum(tax.values()) + correct + ohf == n

fig, ax = plt.subplots(figsize=(9.5, 6))
arms3 = ["Baseline", "RAG", "Fine-tuned"]
tax_data = [b_tax, r_tax, f_tax]
tax_colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(TAXONOMY_ORDER)))
bottom = np.zeros(3)
for i, bucket in enumerate(TAXONOMY_ORDER):
    vals = np.array([t[bucket] for t in tax_data])
    ax.bar(arms3, vals, bottom=bottom, color=tax_colors[i], width=0.55, label=TAXONOMY_LABEL[bucket])
    bottom += vals
# also show "other hard fail" and "correct" as context bars stacked on top, muted
other_hf_vals = np.array([b_ohf, r_ohf, f_ohf])
ax.bar(arms3, other_hf_vals, bottom=bottom, color="#4A4A4A", width=0.55, label="Other hard failure")
bottom += other_hf_vals
correct_vals = np.array([b_correct, r_correct, f_correct])
ax.bar(arms3, correct_vals, bottom=bottom, color=GOOD, width=0.55, label="Correct (not a bug)")
ax.set_ylabel("Cases (n=304)", fontsize=11)
ax.set_title("Bug Taxonomy by Arm -- Full 304-Case Set (mechanical classification)", fontsize=12.5, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8)
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig(FIGURES / "bug_taxonomy_by_arm_304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'bug_taxonomy_by_arm_304.png'}")

# ---------------------------------------------------------------------------
# NEW this session: fine-tuned accuracy by database (evidence for the
# batch-collision-fix finding)
# ---------------------------------------------------------------------------
from collections import defaultdict
by_db = defaultdict(lambda: [0, 0])
for r in ft:
    by_db[r["database"]][1] += 1
    if r.get("execution_accuracy"):
        by_db[r["database"]][0] += 1

dbs_sorted = sorted(by_db.keys())
accs = [100 * by_db[d][0] / by_db[d][1] for d in dbs_sorted]
ns = [by_db[d][1] for d in dbs_sorted]
collision_dbs = {"college_1", "college_2", "college_3", "chinook_1", "store_1", "flight_2", "flight_4", "sakila_1"}
colors_db = [CRIMSON if d in collision_dbs else GREY_BASELINE for d in dbs_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(dbs_sorted, accs, color=colors_db, width=0.65)
for bar, v, n in zip(bars, accs, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_ylabel("Execution accuracy (%)", fontsize=11)
ax.set_ylim(0, max(accs) * 1.25)
ax.set_title("Fine-tuned (23-db) Accuracy by Database -- Full 304-Case Set", fontsize=13, color=INK, fontweight="bold", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PANEL, linewidth=1, zorder=0)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=CRIMSON, label="Former collision-group db (naming/domain siblings)"),
                Patch(facecolor=GREY_BASELINE, label="Other database")]
ax.legend(handles=legend_elems, loc="upper right", frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(FIGURES / "finetuned_accuracy_by_database_304.png", dpi=300)
plt.close()
print(f"Saved {FIGURES / 'finetuned_accuracy_by_database_304.png'}")

print("\nAll figures generated successfully.")
