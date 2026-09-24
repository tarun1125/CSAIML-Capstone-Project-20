# Experiment — does hybrid (BM25 + vector) retrieval on Azure AI Search beat FAISS?

**Status:** pre-registration, signed off by Tarun on **2026-09-24**. Gate 1 not yet run.
Nothing in §4–§6 may change after Gate 1 has been run. If something turns out to be wrong, correct
it in place and mark it **[CORRECTED]** with the reason, as `EXPERIMENT-reranking.md` does.

---

## 1. The question

Does adding a **lexical (BM25) leg** to retrieval, fused with the existing MiniLM vector leg by
Reciprocal Rank Fusion, change (a) which exemplars and which database the prompt gets, and
(b) execution accuracy, with the generator, prompts, database policy and scoring held fixed?

Secondary: does Azure's managed semantic ranker (A3) repeat the reranking finding, where ranking
metrics moved and execution accuracy did not?

## 2. What already exists (facts)

| Piece | Where | Value |
|---|---|---|
| Exemplar pool | `rag/data/fewshot_metadata.json` | 1,213 exemplars, 23 databases |
| Held-out cases | `rag/data/rag_test.json` | 304 |
| A0 retrieval | FAISS `IndexFlatIP`, MiniLM on `question` only | exact cosine |
| A0 retrieval metrics | `results/retrieval_eval_faiss.json` | recall@5 0.9638 · MRR@10 0.9288 · nDCG@10 0.8928 · db vote@10 0.8487 (258/304) · db rank-1 0.9046 (275/304) |
| A0 execution (vote, FK on) | `rag/data/qwen_rag_fk_k10_execution_results.json` | **143/304** |
| A0 execution (rank-1, FK on) | `rag/data/qwen_rag_rank1_k10_execution_results.json` | 149/304 |
| Azure index | `srch-capstone-22b895` / `fewshot-exemplars` | same 1,213 docs, **FAISS's own vectors**, exhaustive KNN cosine, BM25 on `question` only (`en.microsoft`) |
| Oracle noise floor | README | ≈0.4% (1 in 277 flipped on a tie) |

## 3. Arms

| Arm | Retrieval | Role |
|---|---|---|
| A0 | FAISS | reference, already scored |
| A1 | Azure vector-only, exhaustive KNN | **positive control**: must reproduce A0's lists |
| A2 | Azure hybrid: BM25(`question`) + vector (k=50), RRF, top 10 | **the question** |
| A3 *(optional, Basic tier)* | A2 + semantic ranker | managed reranker |

**Held fixed in every arm:** the 304 cases, K = 10, FK on, `schema_cards`, `PROMPT_HEADER` /
`PROMPT_RULES` / `build_system_prompt`, Qwen2.5-Coder-1.5B bf16 on MLX, greedy decoding, max 300 tokens,
`clean()`, `normalize.py`, the execution oracle. The **database policy** is run both ways:
`vote` (primary, compared against the 143 run) and `rank1` (secondary, compared against the 149 run).

## 4. Gates and thresholds — set 2026-09-24, before any Azure retriever was evaluated

**Gate 1 (A1 parity).** A1's top-10 row list is compared with A0's on all 304 cases.

- **Pass threshold: ≥ 300 / 304 identical top-10 lists, and every non-identical case is an exact
  cosine tie** (`tie_at_divergence = true` in `parity_vs_faiss`). Both conditions are required.
- **If it fails:** stop. A2/A3 are not run or reported until the cause (vector upload, field
  mapping, metric, or ordering) is found and fixed, and Gate 1 then passes. *(This rule is from
  `docs/AZURE-PLAN.md`, Phase 3.)*

**Gate 2 (retrieval metrics), no pass/fail.** Report recall@{1,5,10}, MRR@10, nDCG@10, db vote@10,
db rank-1 for A1, A2 (and A3) next to A0.

**Primary outcome.** A2-vote vs A0-vote (143/304), McNemar exact test on execution accuracy.

- **Decision rule: A2 is called an improvement only if BOTH hold:** a net gain of **≥ 9 cases**
  (A2 ≥ 152/304, about +3 percentage points) **and** McNemar exact **p < 0.05**.
  A net gain of ≥ 9 with p ≥ 0.05 is reported as *not significant*, with its discordant counts.
  So is anything below 9 cases, whatever its p-value.
- The A2-rank1 vs A0-rank1 (149/304) comparison is secondary and uses the same rule.

## 5. Predictions — recorded 2026-09-24, before any Azure retriever was evaluated

| Item | Prediction |
|---|---|
| **A2 execution accuracy (vote) vs A0 (143/304)** | **≥ 155/304** (≥ +12 cases): above the bar in §4 |
| Gate 1 | not predicted |
| A2 ranking metrics (recall@5, nDCG@10) | not predicted |
| A2 database accuracy (vote@10, rank-1) | not predicted |
| Mechanism | not given |
| A3 | not predicted (may not be run) |

*History, kept deliberately:* an earlier draft of this section said "same, same, slightly up,
about 145, below the bar". That was copied from an example in the planning conversation, not
the author's own view. It was replaced by the prediction above on the same day, **before any
Azure retriever was evaluated**. Only the table above counts.

## 6. What each outcome would mean — agreed now

| Outcome | Interpretation |
|---|---|
| Gate 1 fails | A harness bug. No A2/A3 number is reported until it's found. |
| A2 ≈ A0 on execution | BM25 adds nothing on top of MiniLM for this data. That's a finding about the corpus. |
| A2 improves db accuracy, not execution | The reranking pattern again; analyse with `eval_retrieval`'s tools. |
| A2 improves execution | Inspect the gained **and** lost cases before believing it: are the gains on rare-term questions? |
| A2 worse | Lexical overlap pulls in wrong-database exemplars; check sibling-database cases. |

## 7. Procedure (commands)

```bash
python rag/eval_retriever.py --retriever faiss            # positive control, must pass
python rag/eval_retriever.py --retriever azure-vector     # Gate 1 + Gate 2 (A1)
python rag/eval_retriever.py --retriever azure-hybrid     # Gate 2 (A2) -- only if Gate 1 passed
python rag/build_prompts.py --retriever azure-hybrid 10
python rag/build_prompts.py --retriever azure-hybrid --db-policy rank1 10
python rag/generate_rag_mlx.py --prompts-path rag/data/rag_prompts_azhyb_k10.json \
    --output rag/data/qwen_rag_mlx_azhyb_k10_results.json          # overnight, fresh --output
# add "azhyb_k10" (and "rank1_azhyb_k10") to ARMS in rag/score_rerank_arms.py, then:
python rag/score_rerank_arms.py azhyb_k10
```
McNemar vs A0: the snippet in `docs/AZURE-PLAN.md` Phase 3 step 5 (A0 = `qwen_rag_fk_k10_execution_results.json`).

## 8. Caveats to carry into the finding

- One run per arm with greedy decoding. Differences of 1–2 cases are within oracle noise.
- Azure's hybrid depends on its RRF constant and the vector leg's k (50 here, `HYBRID_VECTOR_K`).
  A different k is a different arm, not a tuning knob to adjust after seeing results.
- `retrieved_scores` in Azure prompt files are Azure scores, not cosine: vector = 1/(2 − cos), hybrid = RRF.
- A3 needs the Basic tier (billed hourly). If it isn't run, say so. Don't imply it.
