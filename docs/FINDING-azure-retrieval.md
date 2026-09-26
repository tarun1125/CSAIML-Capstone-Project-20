# Finding — hybrid (BM25 + vector) retrieval on Azure AI Search does not move execution accuracy

**Status:** final, 2026-09-26. Interpretation (§2, §7) signed off by Tarun on 2026-09-26.
**Pre-registration:** [`docs/EXPERIMENT-azure-retrieval.md`](EXPERIMENT-azure-retrieval.md),
signed 2026-09-24, commit `f075044`. Every claim below is checked against it.
**Follows:** [`FINDING-reranking.md`](FINDING-reranking.md) (reordering exemplars: no effect) and
[`FINDING-rank1-vote.md`](FINDING-rank1-vote.md) (changing the database decision: +6, 0 lost).
**Artifacts:** `results/retrieval_eval_{faiss,azure-vector,azure-hybrid}.json`,
`rag/data/rag_prompts_{,rank1_}azhyb_k10.json`, `rag/data/qwen_rag_{,rank1_}azhyb_k10_execution_results.json`

---

## 0. The result

| Comparison | A0 (FAISS) | A2 (Azure hybrid) | net | gained / lost | McNemar exact p | pre-registered bar (≥ +9 **and** p < 0.05) |
|---|---|---|---|---|---|---|
| **Primary: vote policy** | 143/304 | **148/304** | **+5** | 19 / 14 | **0.4869** | **not met** |
| Secondary: rank-1 policy | 149/304 | 150/304 | +1 | 15 / 14 | 1.0000 | not met |

**Hybrid retrieval is better retrieval** (every Gate 2 metric improved; see §1) **and not better
task accuracy.** The primary comparison is a +5 net inside 33 discordant cases, indistinguishable
from churn. At least one of the 19 gains is an oracle false positive (§2.3), so the true net is
at most +4.

**The prediction was wrong.** Pre-registered: A2 ≥ 155/304. Observed: 148. The outcome matches
the pre-registered row *"A2 improves db accuracy, not execution: the reranking pattern again"*.

### Next to the two earlier retrieval experiments

Same model, decoding, schema cards, prompt rules and scorer. Only the retrieval step differs.

| Intervention | prompts changed | database decision changed | discordant | gained | lost | p |
|---|---|---|---|---|---|---|
| Cross-encoder reranking, K=10 | 300/304 | — | 42 | 20 | 22 | 0.8776 |
| Rank-1 database policy | 27/304 | 27 | 6 | 6 | 0 | 0.0313 |
| **Azure hybrid (A2), vote** | **304/304** | **19** | **33** | **19** | **14** | **0.4869** |

*The reranking row is against the FK-on A0 (143), as in `FINDING-reranking.md` and
`FINDING-rank1-vote.md`. The status line of `EXPERIMENT-reranking.md` used to say 48 discordant,
24/24, p = 1.0, computed against the no-FK file. It was corrected 2026-09-26.*

The pattern holds a third time. Interventions that rewrite the exemplar list in almost every
prompt produce churn. The one intervention that changed only the database decision produced a
clean gain.

---

## 1. Retrieval did improve (Gate 2)

| Metric | A0 FAISS | A1 Azure vector | A2 Azure hybrid |
|---|---|---|---|
| recall@5 | 0.9638 | 0.9638 | **0.9737** |
| MRR@10 | 0.9288 | 0.9288 | **0.9394** |
| nDCG@10 | 0.8928 | 0.8929 | **0.8975** |
| db vote@10 | 258/304 | 258/304 | **263/304** (10 gained / 5 lost, p = 0.30) |
| db rank-1 | 275/304 | 275/304 | **279/304** (7 / 3, p = 0.34) |

A2 shares a mean of **7.3 of 10** exemplars with FAISS, and changes the rank-1 exemplar in 78 of
304 cases. It is a real intervention. The database gains are mostly sibling confusions that BM25
resolved: `college_1/2` ×4, `csu_1`, `store_1` vs `hr_1`, `sakila_1` vs `store_1`, `flight_4` vs
`flight_2`. The losses are keyword overlaps that pulled in the wrong database (`sakila_1` → `world_1`,
`hr_1` → `world_1`, `baseball_1` → `hr_1`). None of these retrieval deltas is significant on its own.

---

## 2. Why better retrieval did not become better answers

### 2.1 Most flips have nothing to do with the database

Of the 33 discordant cases, only **7** involve a change in the predicted database (5 of the
19 gains, 2 of the 14 losses). The other **26** flipped with the **same database and the same
schema block**; the only difference was about 3 swapped examples. A greedy 1.5B model is sensitive
enough to its examples that changing them moves individual answers in both directions at similar
rates. That is the reranking result again, measured on a different intervention.

### 2.2 The database channel worked, but it is small

Where the database *did* change, it went the right way. The 5 gains include 4 cases where A2
fixed a wrong database, and the 2 losses include only 1 where A2 broke a right one. That is the
rank-1 mechanism, and it works here too. But hybrid changed the database decision in only 19
cases (rank-1 changed it in 27), so this channel carries a net of about +3. The 26-case churn
above drowns it.

Execution accuracy conditioned on getting the database right barely moved: **136/258 (52.7%) → 144/263 (54.8%)**.

### 2.3 One gain is an oracle false positive

`spider-store_1-25`, *"Who is the youngest employee…?"* Gold sorts by `birth_date` descending.
A2 sorted by **`hire_date` ascending**, the wrong field and the wrong question, and returned the same
single row (Jane Peacock), so the oracle scored it correct. A2 also predicted the wrong
database (`hr_1`) for this case. The row-equality oracle cannot see this. **The other 32 discordant
cases were not audited.** The net should be read as +5 before audit, at most +4 after.

### 2.4 Exploratory, not pre-registered

The gains skew **easy** (10 easy, 4 medium, 1 hard, 4 unlabelled) and the losses skew **medium**
(2 easy, 9 medium, 3 unlabelled). This is post hoc, on 33 cases. It's a hypothesis for a
future experiment, not a result.

---

## 3. Gate 1 and the deviation

Gate 1 (A1 parity) **missed its pre-registered count**: 298/304 identical top-10 lists against a
threshold of ≥ 300. It met the tie criterion: 0 different sets, 6/6 mismatches on an exact cosine
tie. All six mismatches are **one exemplar pair with identical question text**, *"How many students
are in each department?"*, as row 760 (`college_1`) and row 1123 (`college_2`). FAISS and Azure
order the exact tie differently. The same 10 exemplars and identical database decisions for all 304
cases rule out a harness bug. Proceeding was recorded as a dated **[DEVIATION]** before any A2 number
existed (commit `af3d073`), with the threshold unchanged. **The gate is reported as missed.**

Side observation: the pool contains an identical question labelled with two sibling databases.
No retriever can separate those two exemplars.

---

## 4. What was held fixed

The 304 held-out cases, K = 10, FK annotations on, `schema_cards`, `build_system_prompt`
(`PROMPT_HEADER` / `PROMPT_RULES`), Qwen2.5-Coder-1.5B-Instruct bf16 on MLX with greedy decoding
and max 300 tokens, `clean()`, `normalize.py`, and the execution oracle against Atlas. The Azure
index holds **FAISS's own float32 vectors** (`reconstruct_n`, not re-embedded) with exhaustive-KNN
cosine, and BM25 sees only the `question` field, as FAISS embeds only the question. Hybrid fuses
BM25 with a 50-deep vector leg by Azure's RRF and cuts to 10.

A0 is `qwen_rag_fk_k10_execution_results.json` (143/304, FK on), **not**
`qwen_rag_execution_results_mlx.json`, which is the no-FK run (141). The plan originally pointed
at the wrong file, and this was caught and fixed before the pre-registration was signed.

Measured along the way: Azure's exhaustive-KNN cosine score is **`1 / (2 − cosine)`** (matches
FAISS to 6 d.p.), and Azure hybrid is **deterministic**: a repeat run returned identical per-case lists.

---

## 5. Caveats

- One greedy run per arm. The oracle's noise floor is about 1 case in 277.
- Only one hybrid configuration was tested: Azure's default RRF with a 50-deep vector leg. A different
  depth, BM25 weighting or field set is a different arm, not a tuning knob.
- A3 (the semantic ranker) was **not run**. It needs the Basic tier, which is billed hourly.
- The oracle scores rows, not reasoning (§2.3). The discordant set was not fully audited.
- "Prompts changed 304/304" counts the full prompt. Only the exemplar block and, in 19 cases, the
  schema block differ; header and rules are identical.

---

## 6. Cost and effort

| Step | Where | Time | Cost |
|---|---|---|---|
| Index build (1,213 docs) | Azure AI Search Free | seconds | $0 |
| Gate 1 / Gate 2 (304 queries each) | Azure Free | about 20–30 s each | $0 |
| Prompt build, both policies | local + Azure Free | about 30 s each | $0 |
| Generation, both arms | local MLX | 6.6 + 6.4 min | $0 |
| Execution scoring | Atlas | minutes | existing cluster |

The whole experiment cost **$0 of Azure credit**.

---

## 7. What to do with this

- **Don't switch the benchmark pipeline to hybrid for accuracy.** It isn't distinguishable from
  FAISS on execution, and under the better database policy (rank-1) it adds +1.
- **Rank-1 remains the only retrieval change with a measured task gain.** The next retrieval lever,
  if any, should target the **database decision alone** (sibling disambiguation) without rewriting
  the exemplar list, because a rewritten list is where the churn comes from.
- **For the deployed service (Phase 5), Azure Search is a free choice on accuracy.** Vector mode
  reproduces FAISS (Gate 1), and hybrid is equivalent within noise, so pick on operational grounds
  (managed service, no index file in the image). Say this plainly: it's an infrastructure choice,
  not an accuracy improvement.

---

## 8. Reproducing

```bash
python rag/azure_index.py                                   # index (Free tier, azure.env)
python rag/eval_retriever.py --retriever faiss              # positive control
python rag/eval_retriever.py --retriever azure-vector       # Gate 1
python rag/eval_retriever.py --retriever azure-hybrid       # Gate 2
python rag/build_prompts.py --retriever azure-hybrid 10
python rag/build_prompts.py --retriever azure-hybrid --db-policy rank1 10
python rag/generate_rag_mlx.py --prompts-path rag/data/rag_prompts_azhyb_k10.json \
    --output rag/data/qwen_rag_mlx_azhyb_k10_results.json
python rag/generate_rag_mlx.py --prompts-path rag/data/rag_prompts_rank1_azhyb_k10.json \
    --output rag/data/qwen_rag_mlx_rank1_azhyb_k10_results.json
python rag/score_rerank_arms.py azhyb_k10 rank1_azhyb_k10
```
McNemar: `docs/AZURE-PLAN.md`, Phase 3 step 5.
