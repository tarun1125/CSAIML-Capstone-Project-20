# Finding — a hosted gpt-4o on the same prompts: +55 cases over Qwen 1.5B

**Status:** complete, 2026-09-26, including the follow-up arm **G1 + rank-1 (§3.1): 212/304**. Interpretation in §3 and §6 is marked *(draft — Tarun to confirm)*.
**Arm:** G1 of `docs/AZURE-PLAN.md`, Phase 4. Prediction recorded before the run (commit `88217db`).
**Follows:** [`FINDING-azure-retrieval.md`](FINDING-azure-retrieval.md) changed retrieval with the
generator fixed. This changes the generator with retrieval fixed. They are two questions and two numbers.
**Artifacts:** `rag/data/aoai_gpt4o_k10_results.json` (+ manifest),
`rag/data/qwen_rag_aoai_gpt4o_k10_execution_results.json`, `results/aoai_determinism.json`

---

## 0. The result

| | A0: Qwen2.5-Coder-1.5B, MLX, greedy | **G1: gpt-4o 2024-11-20, Azure OpenAI, T = 0** |
|---|---|---|
| Prompts | `rag/data/rag_prompts.json` | **byte-identical file** |
| Execution accuracy | 143/304 (47.0%) | **198/304 (65.1%)** |
| Paired (McNemar exact) | — | **69 gained / 14 lost**, 83 discordant, **p = 6.8 × 10⁻¹⁰** |

**Prediction (recorded before the run): 147–150. Observed: 198.** The prediction was far too
low. The model-size effect is about ten times anything the retrieval experiments moved (rank-1 +6,
Azure hybrid +5 n.s., reranking −2 n.s.).

### Next to every other change measured on this pipeline

| Change (all vs A0 = 143, same 304 cases) | Result | gained / lost | p |
|---|---|---|---|
| Cross-encoder reranking | 141 | 20 / 22 | 0.88 |
| Azure hybrid retrieval (A2) | 148 | 19 / 14 | 0.49 |
| Rank-1 database policy | 149 | 6 / 0 | 0.031 |
| **Generator → gpt-4o (G1)** | **198** | **69 / 14** | **6.8 × 10⁻¹⁰** |
| **gpt-4o + rank-1 policy (§3.1)** | **212** | **80 / 11** | **4.4 × 10⁻¹⁴** |

---

## 1. Held fixed

The prompts file itself: same exemplars (FAISS, K = 10), same vote database decision, FK
annotations, schema cards, header and rules, and the same system/user message split. Also
`clean()`, `normalize.py`, the AST guard and the execution oracle. **Changed:** the model, and the
decoding stack that comes with it: Azure OpenAI v1 endpoint, temperature 0, seed 42, max 300
tokens (`rag/generate_rag_aoai.py`). The deployment is pinned: Standard, East US, **NoAutoUpgrade**.

---

## 2. Determinism: T = 0 is not deterministic, but the score is stable

The first 50 cases were generated three times: two separate `--limit 50` runs, plus the full run's
first 50 (`results/aoai_determinism.json`).

| | A vs B | A vs full | B vs full |
|---|---|---|---|
| normalized query text differs | 3 | 6 | 6 |
| correctness flips | **0** | 2 | 2 |
| execution correct | 36 / 36 / 36 |

Across the three runs, 7 of 50 cases produced more than one distinct query. Two backend
`system_fingerprint`s were served within a single run. Correctness flipped on 0–2 cases per pair,
and the score was identical every time. Scaled to 304 cases, the noise is on the order of a
handful of cases, an order of magnitude below the +55. **Report G1 as ~198 ± a few, not as an exact
number.** A repeat run will not reproduce it byte for byte; Qwen on MLX with greedy decoding does.

---

## 3. Where the gain is, and what now limits it *(draft — Tarun to confirm)*

| | database right (258 cases) | database wrong (46 cases) |
|---|---|---|
| A0 Qwen | 136 (52.7%) | 7 |
| **G1 gpt-4o** | **192 (74.4%)** | **6** |

**The gain is entirely inside cases where retrieval picked the right database.** With the wrong
schema in the prompt, gpt-4o does no better than Qwen (6 vs 7): a stronger model cannot write a
correct query against a schema it was never shown. So the **database decision is now the binding
constraint**. It accounts for **40 of G1's 106 failures (38%)**, up from 39 of Qwen's 161 (24%).

That changes the value of the retrieval work. Rank-1 fixed a net 17 database decisions and was worth +6 with
Qwen, which converts a right schema into a right answer only 53% of the time. With a generator that
converts at 74%, the same database fixes should be worth more. That was a hypothesis when written; §3.1
tests it.

Failure types, A0 → G1: wrong rows 85 → 56, empty result 62 → 46, runtime error 8 → 3, rejected by
the guard 6 → 1. The 14 losses are all "ran, wrong/empty rows", and 10 of them had the right database.

### 3.1 Tested: gpt-4o + rank-1 = 212/304

Pre-registered before the run (commit `cef5846`): gpt-4o on `rag/data/rag_prompts_rank1_k10.json`
(the same FAISS exemplars; the database comes from the nearest exemplar), **primary comparison against
G1-vote (198)**, **prediction 200+**.

| Comparison | from → to | net | gained / lost | McNemar exact p |
|---|---|---|---|---|
| **G1-rank1 vs G1-vote (primary)** | 198 → **212** | **+14** | **17 / 3** | **0.0026** |
| G1-rank1 vs Qwen-rank1 | 149 → 212 | +63 | 76 / 13 | 5.4 × 10⁻¹² |
| G1-rank1 vs A0 (Qwen, vote) | 143 → 212 | +69 | 80 / 11 | 4.4 × 10⁻¹⁴ |

**Prediction correct (200+).** The mechanism is visible directly. Rank-1 changes the database decision in
27 cases: it **fixes 20**, breaks 3, and changes 4 from one wrong database to another. On the 20 fixed cases:

| | vote → rank-1 on the 20 fixed cases |
|---|---|
| Qwen 1.5B | 6 → 11 (**+5**) |
| **gpt-4o** | **5 → 17 (+12)** |

**Separating effect from noise:** the other **277 prompts are byte-identical** between the two runs,
because the database decision didn't change. On those, gpt-4o flipped 8 cases, net +2, which is pure
run-to-run noise (§2). The +14 is therefore about **+12 from the database fix and about +2 from noise**.
The same retrieval change was worth +6 with Qwen and about +12 with gpt-4o: **a policy's value depends on
the generator behind it.**

Cost $1.56 (525,698 in / 24,803 out tokens), p50 1.9 s, 0 errors.

---

## 4. Cost and latency

| Run | input / output tokens | est. cost |
|---|---|---|
| G1, 304 cases | 520,090 / 24,511 | **$1.55** |
| determinism, 2 × 50 cases | 168,166 / 8,015 | $0.50 |
| smoke tests (5 calls) | 9,421 / 517 | $0.03 |
| G1 + rank-1, 304 cases | 525,698 / 24,803 | $1.56 |
| **Phase 4 total** | **1,223,375 / 57,846** | **≈ $3.64** |

About **$0.005 per query**, or **$0.008 per correct answer**. Estimates use $2.50 / $10.00 per
1M input / output tokens. Check them against the invoice in Cost Management; token counts in the
manifests are exact. Latency from India to East US: **p50 2.0 s, p95 3.2 s** per call (A0 on a local
Mac: about 1.4 s per case).

---

## 5. Caveats

- One full run per arm. Run-to-run noise was measured on 50 cases only (§2).
- Not audited for oracle false positives (the row-equality oracle has produced them before:
  `FINDING-azure-retrieval.md` §2.3, `FINDING-rank1-vote.md`). The 69 gains are unaudited.
- `spider-flight_4-40` failed on an Atlas read timeout (socket timeout, 60 s), not on the query's
  logic. It counts as a failure, as in every arm.
- Two G1 failures are the JS habit `.length` on a Python list (`spider-pets_1-31`, `spider-bike_1-81`).
  `normalize.py` doesn't rewrite it. Adding that would be a pipeline change for every arm, not a G1 fix.
- gpt-4o's training data may include Spider, which is public. This is a comparison of two systems on
  this benchmark, not evidence about unseen schemas.
- gpt-4o-mini, the plan's first choice, could not be deployed: new deployments were blocked from
  2026-03-31. gpt-4o is a larger and costlier model than the plan assumed.

---

## 6. What to do with this *(draft — Tarun to confirm)*

- **Generator size is the largest lever measured in this project**, larger than every retrieval
  change combined. For the deployed service, the case for a hosted model is about +18 points at
  about half a cent per query.
- **Best configuration measured: gpt-4o + FAISS + rank-1 database policy, 212/304 (69.7%).** This is
  the configuration the deployed service (Phase 5) should serve, with the rank-1 policy, not the vote.
- **Retrieval work should target the database decision.** With a strong generator, each database fix
  converts to a correct answer at about 85% (17 of 20), so what's left of the 40 wrong-database failures
  is the next lever. Exemplar reordering (reranking, hybrid) isn't.
- **Report hosted-model numbers with their noise.** Unlike the MLX arms, G1 is not bit-reproducible.

---

## 7. Reproducing

```bash
python rag/generate_rag_aoai.py --limit 50 --output rag/data/aoai_gpt4o_k10_det_a.json
python rag/generate_rag_aoai.py --limit 50 --output rag/data/aoai_gpt4o_k10_det_b.json
python rag/generate_rag_aoai.py --output rag/data/aoai_gpt4o_k10_results.json
python rag/score_rerank_arms.py aoai_gpt4o_k10
python rag/generate_rag_aoai.py --prompts-path rag/data/rag_prompts_rank1_k10.json \
    --output rag/data/aoai_gpt4o_rank1_k10_results.json          # G1 + rank-1
python rag/score_rerank_arms.py aoai_gpt4o_rank1_k10
```
Needs `AZURE_OPENAI_*` in `azure.env` and the deployment above. McNemar: `docs/AZURE-PLAN.md`
Phase 3 step 5, with `qwen_rag_aoai_gpt4o_k10_execution_results.json` as the arm.
