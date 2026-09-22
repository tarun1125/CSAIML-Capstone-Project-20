# Canonical artifacts

Which file to read for each published number, and which same-named siblings are
superseded.

This exists because the results files are spread across `data/`, `rag/data/`,
`outputs/`, `rag/outputs/` and `fine_tuning/outputs/`, and several superseded
files sit unmarked next to the current ones under near-identical names —
`qwen_rag_mlx_results.json` vs `qwen_rag_mlx_fk_results.json` vs
`qwen_rag_mlx_nofk_results.json`, or four generations of `finetuned_full304*`.
Nothing in a filename says which is authoritative, so a reader picks by guess.
Nothing is deleted or moved here; the point is only to say which is which.

**Nothing in this file is a claim about what the numbers mean** — see the README
for that, and `outputs/false_positive_audit.csv` for how many of the "correct"
cases survive review.

---

## The three canonical arms (304-case held-out test set, unified MLX-LM stack)

| Arm | Published | Generation | Normalized | Case-level execution results | Aggregate |
| --- | ---: | --- | --- | --- | --- |
| Zero-shot baseline | 15/304 (4.9%) | `data/qwen_baseline_mlx_testslice_results.json` | `data/qwen_baseline_mlx_testslice_normalized.json` | `rag/data/qwen_baseline_testslice_execution_results_mlx.json` | `rag/outputs/rag_vs_baseline_scores_mlx.csv` |
| RAG (K=10, FK-included) | 142/304 (46.7%) | `rag/data/qwen_rag_mlx_fk_results.json` | `rag/data/qwen_rag_mlx_fk_normalized.json` | see the caveat below | `rag/outputs/rag_vs_baseline_scores_mlx.csv` |
| LoRA fine-tuned (23-db, 1000-iter) | 103/304 (33.9%) | `data/finetuned_full304_23db_1000iter_results.json` | `data/finetuned_full304_23db_1000iter_normalized.json` | `data/finetuned_full304_23db_1000iter_execution_results.json` | `fine_tuning/outputs/finetuned_23db_scores_1000iter.csv` |

Supporting inputs, all canonical:

| What | File |
| --- | --- |
| The 1,517-case reference corpus | `data/reference_queries.json` |
| Executed gold results (all 1,517) | `data/gold_results.json` |
| 304-case held-out test split | `rag/data/rag_test.json` |
| 1,213-case few-shot pool | `rag/data/rag_fewshot_pool.json` |
| RAG prompts at K=10, FK-included | `rag/data/rag_prompts.json` |
| Schema cards (23 dbs, 161 collections) | `rag/schema_cards.json` |
| Baseline confidence/logprob run | `data/qwen_baseline_mlx_testslice_results_lp.json` |
| RAG self-consistency, k=5, T=0.7 | `rag/data/qwen_rag_mlx_selfconsistency_k5.json` |
| Retrieval-ranking + reranking evaluation | `results/retrieval_eval.json` |
| Cross-encoder pair scores (both arms, cached) | `rag/data/rerank_scores.json` |

---

## The RAG case-level caveat

RAG's canonical **aggregate** figure is 142/304 (46.7%), scored from the
FK-included variant, in `rag/outputs/rag_vs_baseline_scores_mlx.csv` (identical
to `_mlx_fk.csv`).

The FK variant's **case-level** execution-results JSON was overwritten by the
later FK-vs-no-FK A/B run. The surviving case file is the no-FK variant,
`rag/data/qwen_rag_execution_results_mlx.json` at 141/304 — one case off the
aggregate, well inside noise. Every per-case analysis in this repo (outcome
mixes, complexity breakdowns, the correct-set overlap, the false-positive audit)
therefore reads the **141-case** file, while the headline accuracy quotes the
**142** aggregate. That 1-case discrepancy is documented rather than papered
over; regenerating the FK variant would close it.

---

## Superseded — do not read these for a current number

| File | Superseded by | Why |
| --- | --- | --- |
| `rag/data/qwen_rag_mlx_results.json` | `rag/data/qwen_rag_mlx_fk_results.json` | **Stale**: generated against an older prompt set, then committed beside newer prompts. Differs from a regeneration on 103/304 cases. Matches neither A/B variant. |
| `rag/data/qwen_rag_mlx_nofk_results.json` | — | The no-FK A/B arm. Valid as the B side of that experiment, not a headline number. |
| `rag/outputs/rag_vs_baseline_scores.csv`, `_k3`, `_k5` | `_mlx.csv` | The earlier 61-case Colab/HF-`transformers` runs. Superseded by the MLX-unified 304-case run. |
| `data/finetuned_full304_23db_results.json` (+ `_normalized`, `_execution_results`) | `*_1000iter_*` | The 200-iteration adapter, 73/304 (24.0%) — the epoch-parity bug. |
| `fine_tuning/outputs/finetuned_23db_scores.csv`, `_200iter.csv` | `_1000iter.csv` | Same. |
| `data/finetuned_full304_*.json` (no `_23db`) | `*_23db_1000iter_*` | The 6-database adapter evaluated on 304 cases. |
| `data/finetuned_results.json`, `_normalized`, `_execution_results` | `*_23db_1000iter_*` | The original 61-case fine-tuning run. |
| `data/qwen_baseline_newtestids_*.json` | `data/qwen_baseline_mlx_testslice_*` | Pre-MLX baseline. |
| `rag/data/qwen_baseline_testslice_execution_results.json` | `..._mlx.json` | Pre-MLX (Colab/HF) serving stack. |
| `outputs/execution_scores.csv` | — | Stage 1 only: Claude vs Qwen over the original 305-case corpus. Not comparable to any 304-case figure — different test set, different stack. |
| `data/qwen_execution_results.json`, `data/claude_execution_results.json` | — | Same Stage 1 run. |
| `outputs/codebleu_scores.csv`, `outputs/figures/bertscore.png` etc. (no `_304`) | `*_304.*` | Computed on the earlier 121-case subset. |
| `fine_tuning/outputs/finetuned_304_scores.csv` | — | The 6-db adapter's in-scope/out-of-scope generalization test. A different experiment, not a superseded version of the headline. |

---

## Figures

Current figures carry a `_304` suffix and were regenerated against the
1,000-iteration adapter. Any figure in `outputs/figures/` **without** `_304`
predates the epoch-parity retrain and shows the superseded 24.0% fine-tuned
number. `outputs/figures/_to_delete/` and `fine_tuning/outputs/_to_delete/` are
exactly what they say.

`outputs/figures/finetuned_23db_1000iter_loss_and_memory.png` has no generating
script in the repo — its source data (`fine_tuning/training_log_1000iter.txt`)
is committed, but the plotting script never was, so it is not regenerable from
a clean clone.

---

## Verification and audit outputs

| File | Produced by |
| --- | --- |
| `outputs/false_positive_audit.csv` | `python evaluation/audit_false_positives.py` |
| `outputs/sql_crosscheck.csv` | `python evaluation/crosscheck_sql_gold.py` |

Every generation and prompt-build step also writes a sibling
`<name>.manifest.json` recording model, adapter, decoding settings, library
versions, git SHA, and a SHA-256 of the results file — see `run_manifest.py`.
A manifest is the authority on how a specific file was produced; this document
is only the authority on which file to read.
