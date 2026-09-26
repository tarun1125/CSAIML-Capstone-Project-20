# Plan — moving the RAG arm to Azure

**Status:** Phases 0–2 done (2026-09-24). On Azure: `rg-capstone-rag` + Free search service `srch-capstone-22b895` ($0) · **Written:** 2026-09-22 · **Revised:** 2026-09-23
after a code review against this plan · **Budget:** the $200 free-account credit, which expires
30 days after sign-up
**Branch to work on:** `azure-deploy` (exists; the code prerequisites below are in `ef8f032`)

> **Day 0 = 2026-09-24** (free trial, subscription `Azure subscription 1`, offer `FreeTrial_2014-09-01`,
> spending limit on) · **Teardown by Day 25 = 2026-10-19** · **Credit expires Day 30 = 2026-10-24**
> · Budget alerts: $25, $50 (add $100 before any Pay-As-You-Go upgrade)

### Already done in the code (2026-09-23, `ef8f032`)

These were blockers for Phases 4–5. They are fixed, so don't redo them:

- `rag/build_prompts.py` can be imported by a server: argv parsing lives in `main()`, faiss is
  imported lazily, and `build_system_prompt(neighbors, db, cards_by_db)` is the one function
  that assembles a prompt. **The service calls that, not the individual `render_*` pieces.**
- `fine_tuning/generation_utils.clean()` imports without MLX installed.
- `rag/schema_cards.build_cards()` returns all 23 databases on a clean clone (the 17 without local
  dumps come from `rag/schema_cards.json`), so the Docker image doesn't need `database/`.
- `atlas_env.connect()` reads `MONGODB_URI` from the environment first (the Key Vault reference),
  then falls back to `atlas-credentials.env`.
- The AST guard rejects `.client` / `.database` / `.collection` (these read other databases) and
  every operator except `+ - /` (`9**9**9` hangs the worker). Tests in `tests/test_query_safety.py`.
- `azure.env` is gitignored; `.dockerignore` keeps both `.env` files out of the build context.

---

## What this is, and what it is not

Two goals, kept apart on purpose.

1. **An experiment you can defend at D3.** Swap FAISS for Azure AI Search, keep *everything
   else* fixed, re-run the 304 held-out cases, and compare with McNemar's test. This
   produces a finding in the same shape as `FINDING-reranking.md` and `FINDING-rank1-vote.md`.
2. **A deployed service.** FastAPI on Azure Container Apps: question in → retrieved exemplars
   → generated PyMongo query → rows from Atlas, with per-stage latency in Application
   Insights. This closes the "cloud deployment" gap in the F2/F3 job postings.

**It is not a re-run of the benchmark on a different model.** The lesson of the Colab→MLX
confound applies directly: change one thing at a time. Retrieval changes in Phase 3 with
the generator held fixed (Qwen on MLX, locally). The generator changes in Phase 4 with
retrieval held fixed. They are two questions and never go into one number.

---

## Four facts about the free account that shape this plan

| Fact | Consequence |
|---|---|
| **The $200 credit expires 30 days after sign-up.** Unused credit is lost. | The plan fits in ~16 working days plus teardown by day 25. Don't create the account until you can start the next day. |
| **Free-trial subscriptions get 0 quota for Azure OpenAI models.** Upgrading to Pay-As-You-Go unlocks quota *and keeps the remaining credit.* ([Microsoft Q&A, Jun 2026](https://learn.microsoft.com/en-gb/answers/questions/5907451/free-trials-get-200-but-have-a-0-quota-limit-on-ll)) | Phases 1–3 need no LLM on Azure and run on the free trial. Upgrade only at Phase 4. |
| **The free trial has a spending limit; Pay-As-You-Go does not.** After upgrading, budgets *alert* but do not stop spending. | Budget alerts on day 0. After the upgrade, deleting the resource group is the only real off switch. |
| **AI Search Free tier: 50 MB, 3 indexes, no semantic ranker.** Free services can be deleted after long inactivity. | The index is ~2 MB (1,213 × 384 floats), so it fits easily. Semantic ranker needs Basic, which is billed hourly, so it's optional (arm A3). |

> ✅ **Resolved 2026-09-24:** the Free tier accepted the 384-dim exhaustive-KNN field; no Basic tier needed for A1/A2.
>
> ⚠️ **Check one thing on day 1:** that a vector field can be created on your Free search
> service. The limits documentation lists vector quota only for Basic and above. Arm A1
> deliberately uses **exhaustive KNN**, which Microsoft documents as *not* consuming vector
> quota. If index creation still fails on Free, switch to Basic for the experiment week and
> delete it afterwards.

---

## Budget

| Item | When | Est. cost | Note |
|---|---|---|---|
| AI Search — Free | Phases 2–3 | $0 | |
| AI Search — Basic *(only if Free rejects vectors, or for arm A3)* | ≤ 7 days | roughly $2–3/day | Billed hourly. Check the pricing calculator on the day. Delete as soon as the arm is scored. |
| Azure OpenAI — a small chat model | Phase 4, 2 runs × 304 cases | < $2 | Prompts are ~5.6k characters median (~1.6k tokens), so a run is ~0.5M input tokens |
| Azure OpenAI — service smoke tests | Phase 5–6 | < $1 | |
| Container Apps (consumption, scale to zero) | Phase 5–7 | ~$0 | The monthly free grant covers 180k vCPU-s, 360k GiB-s and 2M requests |
| Container Registry — Basic | Phase 5–7 | ~$3–5 | Or push to GitHub Container Registry for $0 |
| Log Analytics / App Insights | Phase 5–7 | < $2 | At demo volume |
| Key Vault | Phase 5–7 | cents | |
| **Total** | | **well under $50** | The remaining ~$150 is headroom, not a target. Don't spend it on GPU VMs. |

---

## Timeline — Day 0 is the day you create the account

| Days | Phase | Needs Azure? | Effort |
|---|---|---|---|
| 0 | **0 · Account & guardrails** | yes | 1 hour |
| 0–1 | **1 · Retriever seam, locally** | no | 1 evening |
| 1–2 | **2 · Build the AI Search index** | free trial | 1 evening |
| 2–7 | **3 · The retrieval experiment** ← the part you own | free trial | 2–3 evenings, generation runs unattended |
| 7 | **Decision:** upgrade to Pay-As-You-Go? | — | 10 minutes |
| 8–10 | **4 · Generator arm on Azure OpenAI** | PAYG | 1–2 evenings |
| 10–14 | **5 · Deploy the service** | PAYG | a weekend |
| 14–16 | **6 · Observe, measure, write up** | PAYG | 2 evenings |
| ≤ 25 | **7 · Teardown** | — | 30 minutes |

In the job-hunt plan this is weeks 2–3. Start Phase 0 on a day when Phase 1 can follow the
next evening.

---

## Phase 0 — Account & guardrails (Day 0) — ✅ done 2026-09-24

- [x] Create the Azure free account. Note the **expiry date** (Day 0 + 30) at the top of this file.
- [x] Install the CLI and sign in: `brew install azure-cli` → `az login` → `az account show`.
- [x] Register the providers you will use:
  ```bash
  for ns in Microsoft.Search Microsoft.App Microsoft.OperationalInsights \
            Microsoft.ContainerRegistry Microsoft.KeyVault Microsoft.CognitiveServices; do
    az provider register --namespace $ns
  done
  ```
- [x] One resource group for everything, so teardown is one command:
  ```bash
  az group create -n rg-capstone-rag -l centralindia --tags project=capstone-rag owner=tarun
  ```
  **Why Central India:** South India is one of the regions without the higher AI Search limits.
- [x] **Budget alerts:** Portal → Cost Management → Budgets, on the subscription, at **$25 / $50 /
  $100**, emailing you. *(Set: $25 and $50. Add $100 before any upgrade.)* Do this now. After a Pay-As-You-Go upgrade these alerts are the only warning you get.
- [ ] Put `AZURE_*` settings in a local `azure.env`. It is already in `.gitignore`; confirm with
  `git check-ignore azure.env` before the first secret goes in. This repo is public.
- [ ] Read **Cost control** (below Phase 7) now, and put the expiry date and the end-of-session
  checklist somewhere you'll see them.

---

## Phase 1 — A retriever seam, locally (Day 0–1, no Azure) — ✅ done 2026-09-24

**Status:** built and verified. Five prompt files rebuild byte-identically: `rag_prompts.json`,
`_fk`, `_rank1_k10`, `_k5` and `_rerankQ_k10`. `rag/eval_retriever.py --retriever faiss` reproduces all 8 published
bi-encoder metrics in `results/retrieval_eval.json` exactly (positive control), and
`tests/test_retrievers.py` pins every existing filename. `AzureSearchRetriever` raises
`NotImplementedError` until Phase 2 fills it in.

FAISS is called directly in **two** places, and the seam has to cover both:
`rag/build_prompts.py` (prompts → generation) and `rag/rerank.build_candidates()`, which
`rag/eval_retrieval.py` uses for the ranking metrics in Gate 2. If only `build_prompts.py` is
swapped, Gate 2 silently measures FAISS for every arm. (`demo_ui/live_inference.run_rag` is a
third, but only the demo uses it.)

- [x] `rag/retrievers.py` defines a `Retriever` protocol:
  `search(question: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]`, which returns
  **row indices** into `fewshot_metadata.json` with their scores. Rows, not exemplar ids: the ids
  are a mix of ints and strings (e.g. `12` vs `"spider-car_1-44"`), and an Azure key is always a
  string, so an id round-trip turns `12` into `"12"` and the `row_of_id` lookup misses.
  - `FaissRetriever` holds exactly the current `index.search(qvec, k)` logic.
  - `AzureSearchRetriever(mode="vector" | "hybrid" | "semantic")` is added in Phase 2.
  - **Keep the load-bearing import order:** `embed_utils` (torch) before `faiss`.
- [x] `build_prompts.py --retriever faiss|azure-vector|azure-hybrid|azure-semantic`, defaulting
  to `faiss`, parsed in `parse_args()` next to `--rerank`. The output filename gets the retriever
  label (`rag_prompts_azvec_k10.json`), and `write_manifest` records `retriever=`.
- [x] `rerank.build_candidates(top_n, retriever=...)` takes the same retriever. **Changed from
  the original plan:** `eval_retrieval.py` did *not* get a `--retriever` flag. It *is* the
  reranking experiment: its cross-encoder cache is keyed on FAISS candidates, and its sanity check
  is pinned to FAISS's recorded vote accuracy, so an Azure candidate list would fail both. Instead,
  **`rag/eval_retriever.py`** reuses its grading and `metrics_retrieval.summarize`, and does
  Gate 1 and Gate 2 for one retriever at a time (see Phase 3).
- [x] **Parity gate.** After the refactor, each of these must rebuild **byte-identically**
  (hash before, rebuild, hash after, then `git checkout -- rag/data/` to drop the rewritten manifests):
  ```bash
  shasum rag/data/rag_prompts.json rag/data/rag_prompts_rank1_k10.json rag/data/rag_prompts_k5.json
  python rag/build_prompts.py 10
  python rag/build_prompts.py --db-policy rank1 10
  python rag/build_prompts.py 5
  python rag/build_prompts.py --rerank rag/data/reranked_neighbors_rerank_question_only.json rerankQ 10
  git status --short rag/data | grep -v manifest     # must print nothing
  git checkout -- rag/data/                          # drop the rewritten manifests
  ```
  (The committed files are the "before". Run each command as written: zsh doesn't split a
  quoted `$args` variable into words.)
  Don't use `rag_prompts_nofk.json` as a parity target: the committed copy was built before
  `retrieved_scores` existed, so it differs from any current build.

*Division of labour:* the refactor is plumbing, so delegate it. Review the diff yourself,
and confirm the parity check yourself.

---

## Phase 2 — The AI Search index (Day 1–2) — ✅ done 2026-09-24

**Status:** service `srch-capstone-22b895` (Free, Central India), index `fewshot-exemplars`, 1,213/1,213
documents. `rag/azure_index.py` builds it; `AzureSearchRetriever` in `rag/retrievers.py` reads it
with the query key. Smoke test: Azure vector search returned FAISS's exact top-10 rows in order
on two sample questions. **This is not Gate 1.** Gate 1 is all 304 cases against the threshold
you pre-register. The index definition is saved at `results/azure_index_definition.json`.

- [x] Create the service: `az search service create -n srch-capstone-<suffix> -g rg-capstone-rag --sku free -l centralindia`
- [x] `rag/azure_index.py` creates the index and uploads the 1,213 pool documents:

  | Field | Type | Attributes | Why |
  |---|---|---|---|
  | `id` | `Edm.String` | key | **The row index as a string** (`"0"`…`"1212"`), so a hit maps straight back to its row (see Phase 1 for why not the exemplar id) |
  | `exemplar_id` | `Edm.String` | retrievable | The original id, for logs and debugging |
  | `question` | `Edm.String` | searchable, `en.microsoft` analyzer | The BM25 side of hybrid |
  | `database` | `Edm.String` | filterable, retrievable | Database vote / rank-1 |
  | `complexity` | `Edm.String` | filterable | Per-bucket breakdowns |
  | `normalized_query` | `Edm.String` | retrievable, **not** searchable | The exemplar text. Kept out of BM25, matching the FAISS design (only the question is embedded). |
  | `question_vector` | `Collection(Edm.Single)`, 384 dims | profile `exact` → **exhaustive KNN, cosine** | Exact search, like `IndexFlatIP` |

- [x] **Upload the vectors already in the FAISS index**, not freshly computed ones:
  `faiss.read_index("rag/data/fewshot.index").reconstruct_n(0, ntotal)` returns the exact float32
  rows FAISS searches. Re-embedding would usually match, but "usually" is a second variable in A1.
  Query vectors still come from `embed_utils.embed()`, just as in FAISS. Do not use Azure's integrated
  vectorizer or an Azure OpenAI embedding model for this experiment. That would change the
  embedding model and the search engine at the same time.
- [x] **Scores are not raw cosine, measured.** For the exhaustive-KNN cosine field,
  `@search.score = 1 / (2 − cosine)` (matches FAISS to 6 d.p.), so `cosine = 2 − 1/score`. It's monotone,
  so the order is cosine order. Hybrid scores are RRF (top hit ≈ 1/61 + 1/61 = 0.0333). The manifest
  records `score_kind`. Gate 1 compares **row lists**, and its tie check uses true cosines from the vectors.
- [x] Verify: the document count is 1,213; storage and vector usage are on the service's
  Overview → Usage tab.
- [x] Credentials: the admin key goes into `azure.env` for indexing. Querying uses the
  **query key**, which is read-only. The service in Phase 5 moves to managed identity.

**Design choices to be able to defend (these are yours):**

- **Exhaustive KNN, not HNSW.** FAISS here is exact (`IndexFlatIP`). HNSW is approximate, so
  using it would add ANN recall error as a second variable. At 1,213 documents, exact search
  costs nothing. The same argument appears in `build_retrieval_index.py`'s comments.
- **BM25 only on the question text**, for the same reason FAISS embeds only the question:
  the job is "find past questions phrased like this one".

---

## Phase 3 — The retrieval experiment (Day 2–7) — ✅ run 2026-09-26 (Day 2), finding in draft

**Status:** pre-registered (`f075044`), Gate 1 **missed** (298/304; all 6 mismatches are one
duplicate-question tie pair; proceeded under a dated deviation, `af3d073`), Gate 2 run, A2
generated and scored under both policies. **Result: A2-vote 148/304 vs A0 143, 19 gained / 14 lost,
McNemar p = 0.4869. The pre-registered bar (≥ +9 and p < 0.05) was not met, and the prediction
(≥ 155) was wrong.** A2-rank1 150 vs 149. A3 (semantic, Basic tier) was not run. Azure cost: $0.
Write-up: `docs/FINDING-azure-retrieval.md`. §2 and §7 are awaiting Tarun's sign-off, which is the
plan's own condition for the Day-7 upgrade.

**Pre-register it first.** Write `docs/EXPERIMENT-azure-retrieval.md` before running
anything, in the shape of `EXPERIMENT-reranking.md`: hypothesis, arms, metrics, what result
would mean what, and your prediction. **Write the prediction yourself.** The plan only lists
the arms.

### Arms

| Arm | Retrieval | Generator | Tier | Role |
|---|---|---|---|---|
| **A0** | FAISS, as today | Qwen 1.5B, MLX, greedy | — | Reference. **Already scored**; no re-run needed. |
| **A1** | Azure, vector-only, exhaustive KNN, same MiniLM vectors | same | Free | **Parity control.** It should reproduce A0's neighbour lists. |
| **A2** | Azure hybrid: BM25 on `question` + the same vector leg, fused by RRF | same | Free | **The actual question** |
| A3 *(optional)* | A2 + semantic ranker | same | Basic | A managed cross-encoder reranker, i.e. the reranking experiment again with Azure's model |

**Held fixed in every arm:** the 304 held-out cases, K = 10, FK annotations on, the database
policy (run `vote`, the canonical default, and `rank1` as a secondary), `schema_cards`,
`PROMPT_HEADER` / `PROMPT_RULES`, `generate_rag_mlx.py` at greedy with max 300 tokens,
`normalize.py`, and the execution oracle.

### Gates, in order

1. **A1 parity.** Exact cosine over the same normalised vectors should return the same
   top-10 in the same order. Expect identical lists on almost every case; any differences
   should trace to exact score ties. **Set your threshold before running** (for example
   ≥ 300/304 identical). If A1 misses it, there is a bug (vector upload, field mapping, or
   metric), and **no hybrid result means anything until it's found.** This is the positive
   control, the same job the rank-1 vote did for the reranking harness.
   **Built:** `python rag/eval_retriever.py --retriever azure-vector` compares every case's
   top-10 rows against FAISS. It writes `parity_vs_faiss` into `results/retrieval_eval_azure-vector.json`
   with `n_identical`, `n_same_set_diff_order`, `n_diff_set` and `n_mismatch_explained_by_tie`,
   plus, for each mismatch, both row lists with the true cosine of every row. Equal cosines at the
   divergence point are a tie; a different set is a bug. **Run `--retriever faiss` first**:
   it must pass its positive control, and it refuses to write output otherwise.
2. **Retrieval metrics** for A1/A2/A3: the same `eval_retriever.py` run writes recall@{1,3,5,10,20},
   MRR@10, nDCG@10 and rank-of-first-relevant (all at depth 50, like the published run), plus
   database accuracy under both policies (`db_vote_accuracy` @5/@10, `db_rank1_accuracy`). No generation,
   so it's cheap. **Basic-tier search only needs to exist while this script and `build_prompts.py`
   run for an arm.** Everything after is local.
   FAISS reference (A0): recall@5 0.9638 · MRR@10 0.9288 · nDCG@10 0.8928 · vote@10 0.8487 · rank-1 0.9046.
3. **Generation, locally.** One arm per night, with a **fresh `--output` every time**. The script
   resumes from any existing output file without checking which prompts file produced it, so
   reusing a path would mix two arms in one file.
   ```bash
   python rag/generate_rag_mlx.py --prompts-path rag/data/rag_prompts_azvec_k10.json \
       --output rag/data/qwen_rag_mlx_azvec_k10_results.json
   ```
   Retrieval already happened in step 1, so **generation needs no Azure resource running**. Pause
   Basic-tier search (see Cost control) before you start an overnight run.
4. **Score with `rag/score_rerank_arms.py`, not `score_rag.py`.** Add each arm to its `ARMS`
   dict (`"azvec_k10": "qwen_rag_mlx_azvec_k10_results.json"`), then
   `python rag/score_rerank_arms.py azvec_k10`. It normalizes, refuses partial runs (≠ 304), and
   writes `qwen_rag_azvec_k10_execution_results.json`. **Don't use `score_rag.py` with file
   arguments:** it overwrites `qwen_rag_execution_results_mlx.json` (a committed artifact, the
   no-FK run's case file), and reads its database diagnostic from the default `rag_prompts.json` whatever arm
   you pass. Read `execution_accuracy`, **not** `status`.
5. **Paired comparison** of each arm against A0 with McNemar's exact test. Report discordant
   counts, not just p. Remember the ≈0.4% oracle noise floor (1 in 277 cases flipped on a tie).
   **A0's file is `qwen_rag_fk_k10_execution_results.json` (143/304), not
   `qwen_rag_execution_results_mlx.json`.** The latter is the *no-FK* run (141/304): the FK
   run's case file was overwritten by the FK A/B test (see `CANONICAL_ARTIFACTS.md`), and the
   `_fk_k10` file is its re-scoring. Every Azure arm has FK on, so the no-FK file would vary FK
   *and* retrieval at once. That's the same trap `eval_retrieval.py` documents for the reranking
   experiment. The 143 vs the headline 142 is the oracle's known noise.
   This snippet was checked against the existing rank-1 arm (143 vs 149):
   ```bash
   python - <<'EOF'
   import json, sys; sys.path.insert(0, "rag")
   from eval_retrieval import mcnemar_exact
   load = lambda f: {str(r["id"]): r.get("execution_accuracy") is True for r in json.load(open(f))}
   a0 = load("rag/data/qwen_rag_fk_k10_execution_results.json")       # A0: 143/304, FK on
   for arm in ["azvec_k10", "azhyb_k10"]:
       print(arm, mcnemar_exact(a0, load(f"rag/data/qwen_rag_{arm}_execution_results.json")))
   EOF
   ```
6. **Capture.** Commit, per arm: the prompts file and its manifest, the raw results and their manifest,
   and the execution results. Also commit `results/azure_retrieval_eval.json` (Gate 1 + 2 numbers)
   and the McNemar output pasted into the finding. The manifests are what make the arm
   reproducible after the Azure account is gone.

### What each outcome would mean — decide before you look

- **A2 ≈ A0.** Lexical matching adds nothing on top of MiniLM for this data. That's plausible:
  the known failure is confusion between sibling databases (`college_1/2/3`, `chinook_1` /
  `store_1`), and those share vocabulary, so BM25 can't separate them.
- **A2 improves database prediction but not execution.** That's the reranking pattern again,
  and you already have the analysis tools for it.
- **A2 improves execution.** Look at *which* cases gained before you believe it. Rare entity
  names and column names are where BM25 should help.
- **A3.** Your own reranking finding predicts that ranking metrics move and task accuracy
  doesn't. Running it tests whether that result holds with a much larger, managed reranker.

Write `docs/FINDING-azure-retrieval.md` in the house style whatever the outcome. A null
result, measured properly, is a finding.

---

## Decision at Day 7 — upgrade to Pay-As-You-Go?

The upgrade is what lets you use an LLM on Azure (Phases 4–6). It keeps the remaining credit
but removes the spending limit.

- **Upgrade** if Phase 3 is scored and written up. Check the budget alerts are firing to your email first.
  *(2026-09-26: scored; finding drafted. Budgets confirmed at billing-account scope, $25 and $50,
  email on. Add $100 before upgrading. After the upgrade, the only Azure OpenAI option visible
  before it, **Provisioned**, must not be chosen: pick Standard or Global Standard.)*
- **Don't upgrade yet** if Phase 3 isn't finished. The deployed service can wait; the finding is the part that matters.
- **If you decide never to upgrade:** Phase 5 can still ship, with Qwen 1.5B running on CPU
  inside the container (a quantised GGUF via `llama.cpp`). Expect tens of seconds per query at
  this prompt length, and note that a quantised model is a different generator from the bf16
  benchmark model. That's acceptable for a demo; it doesn't count as a result.

---

## Phase 4 — Generator arm on Azure OpenAI (Day 8–10) — started 2026-09-26 (Day 2)

**Setup (done):** Azure OpenAI resource `rg-capstone-rag` (East US, S0); deployment `gpt-4o`,
model **gpt-4o 2024-11-20**, **Standard** (pay per token), 50K TPM, **NoAutoUpgrade** (pinned).
gpt-4o-mini was refused ("deprecated since 2026-03-31" for new deployments), and gpt-5-family models
are reasoning models without temperature 0. Central India offers no pay-per-token gpt-4o/-mini, hence
East US. Generator: `rag/generate_rag_aoai.py` (T = 0, seed 42, max 300 tokens, `clean()`, v1 endpoint).

**G1 pre-registration (recorded 2026-09-26, before any G1 result):**
- Comparison: G1 (gpt-4o on A0's exact `rag_prompts.json`) vs A0 (Qwen 1.5B MLX, **143/304**),
  McNemar exact, discordant counts reported.
- **Prediction (Tarun): G1 = 147–150 / 304.**
- No significance bar set. Report net, discordant counts and p as they fall.
- Determinism: first 50 cases run twice before the full run; report the flip rate on the
  normalized query. Three smoke-test calls already showed two different `system_fingerprint`s.

**G1 result (2026-09-26): 198/304 vs A0 143, 69 gained / 14 lost, p = 6.8 × 10⁻¹⁰. The prediction
(147–150) was far too low.** Determinism: query text differs on 3–6 of 50 cases per repeat, the score is
identical (36/50 ×3). About $2.05 in total, $0.005 per query, p50 2.0 s. Write-up:
`docs/FINDING-azure-generator.md`. Optional follow-up: G1 + rank-1 (about $1.55), with a prediction first.

**G1 + rank-1 pre-registration (recorded 2026-09-26, before the run):** gpt-4o on
`rag/data/rag_prompts_rank1_k10.json` (the same FAISS exemplars; the database comes from the rank-1
exemplar, 275/304 right vs the vote's 258). **Primary comparison: vs G1-vote (198/304)**, McNemar
exact, which isolates the database policy with the generator fixed. Secondary: vs Qwen rank-1 (149).
**Prediction (Tarun): 200+ / 304.** No significance bar set. Noise reference: about ±2 per 50 cases (§2 of
the finding).

- [ ] Create an Azure OpenAI (Foundry) resource and deploy **one small, cheap chat model**. Model
  availability varies by region, so check the Foundry portal and use any region that has it.
  The latency to India doesn't matter for a batch run.
- [ ] `rag/generate_rag_aoai.py` mirrors `generate_rag_mlx.py`: the same `system_prompt` / user
  split, temperature 0, the same `clean()` post-processing (import it from
  `fine_tuning/generation_utils`; since `ef8f032` that doesn't need MLX), resumable checkpoints, and a
  manifest recording the model name, version, deployment and API version.
- [ ] **Arm G1:** A0's exact prompts (`rag_prompts.json`) → Azure OpenAI → the same scoring path.
  Only the generator changes.
- [ ] **Determinism check:** run G1 twice, or run 50 cases twice. Hosted models at temperature
  0 aren't guaranteed to give identical output. Report the flip rate, the same way the README
  reports the oracle's.
- [ ] Record the **tokens and cost per query** from the API's usage field. That number is what
  feeds a self-host vs hosted argument later.

G1 against A0 answers a different question from Phase 3: *what does a larger hosted model buy
on this task, per rupee?* Keep it in its own section of the write-up.

---

## Phase 5 — Deploy the service (Day 10–14)

### Shape

```
client ──HTTPS──▶ Container Apps: FastAPI (scale 0→1, 1 vCPU / 2 GiB)
                   │ 1. embed question (MiniLM, CPU, weights baked into the image)
                   │ 2. retrieve top-10   ──▶ Azure AI Search   (managed identity or query key)
                   │ 3. build prompt      (build_prompts.build_system_prompt, imported)
                   │ 4. generate          ──▶ Azure OpenAI      (managed identity)
                   │ 5. AST guard          (evaluation/execute_queries.check_query_is_safe)
                   │ 6. execute, read-only ──▶ MongoDB Atlas    (URI from Key Vault)
                   └─ traces ──▶ Application Insights (per-stage latency)
```

### Files (delegate these, then review them)

```
service/app.py            POST /query {question} → {database, retrieved_ids, query, rows?, latency_ms{...}}
                          GET  /healthz
service/Dockerfile        python:3.12-slim, CPU-only torch, MiniLM weights downloaded at BUILD time
service/requirements.txt  fastapi uvicorn pymongo azure-search-documents openai azure-identity
                          azure-monitor-opentelemetry numpy torch(cpu) transformers
infra/deploy.sh           the az commands below, in order, idempotent
```

**Import, don't copy:** the service calls `build_system_prompt`, `majority_vote_database`,
`safe_eval_query` / `materialize_result` / `to_json_safe`, `normalize` and `clean` from the repo,
so the deployed path is the same code as the benchmark path. All of these import cleanly inside a
uvicorn process with no MLX and no faiss (verified in `ef8f032`).

**Don't reuse `demo_ui/live_inference.py` as-is.** It is written for a single demo user and
would misbehave as a service:
- Each call re-reads the index and metadata, rebuilds all schema cards, and opens a **new
  `MongoClient` that is never closed**. In the service, build the metadata, `cards_by_db`, the
  embedder, the Search client and **one** `MongoClient` once, in FastAPI's lifespan hook.
- `known_databases()` **fails open**: an empty allowlist means "allow everything". In the
  service, an empty allowlist is a startup error, and the database used is always the
  *predicted* one, checked against the 23 names in `rag/schema_cards.json`.

**Two more rules for the image:** `.dockerignore` already keeps `database/`, `data/`, the
adapters and both `.env` files out of the build context. `rag/schema_cards.json` supplies all 23
schemas, so nothing needs `database/`. Bake the MiniLM weights in at build time
(`python -c "from embed_utils import get_embedder; get_embedder()"` in a `RUN` step with
`HF_HOME` set inside the image), or every cold start downloads about 90 MB.

### Resources

- [ ] Container Registry (Basic) and a cloud build: `az acr create … --sku Basic`, then `az acr build -r <acr> -t capstone-rag:v1 -f service/Dockerfile .`
- [ ] A Container Apps environment backed by Log Analytics, then the app with a
  **system-assigned identity**, `--min-replicas 0 --max-replicas 1`, ingress on port 8000.
- [ ] **Role assignments for the app's identity**, no keys in environment variables:
  `AcrPull` on the registry, `Search Index Data Reader` on the search service,
  `Cognitive Services OpenAI User` on the OpenAI resource, `Key Vault Secrets User` on the vault.
  If the Free search tier won't do RBAC data access, store the **query key** (read-only) in Key Vault instead.
- [ ] Key Vault with RBAC, holding one secret: the Atlas URI, referenced from the app as
  `keyvaultref:<secret-uri>,identityref:system`. Pass the value via `--file`, not on the command line.
- [ ] Application Insights through `azure-monitor-opentelemetry`, with one span per pipeline stage.

Check each flag with `az <command> --help` as you go. The CLI changes more often than this plan will.

### Deploy, in order (this becomes `infra/deploy.sh`)

`azure.env` holds names only, no secrets: `RG=rg-capstone-rag LOC=centralindia ACR=… ENV=cae-capstone
APP=ca-capstone-rag KV=… SRCH=… AOAI=… LAW=law-capstone MYIP=<your public IP>`.

```bash
set -a; source azure.env; set +a

# 0. Before anything can reach Atlas: create the READ-ONLY Atlas user (read on the 23 dbs
#    only), and put its URI in a local file you'll delete in step 3.

# 1. Logs first, with a daily ingestion cap so a log storm can't run up a bill.
az monitor log-analytics workspace create -g $RG -n $LAW -l $LOC
az monitor log-analytics workspace update -g $RG -n $LAW --quota 0.1        # GB/day

# 2. Registry + image, built in the cloud (no local Docker needed).
az acr create -g $RG -n $ACR --sku Basic -l $LOC
az acr build -r $ACR -t capstone-rag:v1 -f service/Dockerfile .

# 3. Key Vault (RBAC). Give yourself permission to write secrets, then store the read-only URI.
az keyvault create -g $RG -n $KV -l $LOC --enable-rbac-authorization true
KV_ID=$(az keyvault show -n $KV --query id -o tsv)
az role assignment create --assignee "$(az ad signed-in-user show --query id -o tsv)" \
   --role "Key Vault Secrets Officer" --scope $KV_ID
az keyvault secret set --vault-name $KV -n atlas-uri-ro --file atlas-ro-uri.txt && rm atlas-ro-uri.txt

# 4. Environment (consumption only: no charge while idle) and the app, scale 0..1.
LAW_ID=$(az monitor log-analytics workspace show -g $RG -n $LAW --query customerId -o tsv)
LAW_KEY=$(az monitor log-analytics workspace get-shared-keys -g $RG -n $LAW --query primarySharedKey -o tsv)
az containerapp env create -g $RG -n $ENV -l $LOC --logs-workspace-id $LAW_ID --logs-workspace-key $LAW_KEY
az containerapp create -g $RG -n $APP --environment $ENV \
   --image $ACR.azurecr.io/capstone-rag:v1 --registry-server $ACR.azurecr.io --registry-identity system \
   --system-assigned --min-replicas 0 --max-replicas 1 --cpu 1 --memory 2Gi \
   --ingress external --target-port 8000

# 5. Grant the app's identity what it needs. No keys in env vars.
PID=$(az containerapp show -g $RG -n $APP --query identity.principalId -o tsv)
az role assignment create --assignee $PID --role "Key Vault Secrets User" --scope $KV_ID
az role assignment create --assignee $PID --role "Search Index Data Reader" \
   --scope $(az search service show -g $RG -n $SRCH --query id -o tsv)
az role assignment create --assignee $PID --role "Cognitive Services OpenAI User" \
   --scope $(az cognitiveservices account show -g $RG -n $AOAI --query id -o tsv)
#    (If the image pull failed in step 4, grant AcrPull on the registry the same way and restart.)

# 6. Wire the secret and settings in. This creates a new revision.
SECRET_URI=$(az keyvault secret show --vault-name $KV -n atlas-uri-ro --query id -o tsv)
az containerapp secret set -g $RG -n $APP --secrets "atlas-uri=keyvaultref:$SECRET_URI,identityref:system"
az containerapp update -g $RG -n $APP --set-env-vars MONGODB_URI=secretref:atlas-uri \
   SEARCH_ENDPOINT=https://$SRCH.search.windows.net AOAI_ENDPOINT=… AOAI_DEPLOYMENT=… \
   APPLICATIONINSIGHTS_CONNECTION_STRING=…

# 7. Lock it down before the first real request.
az containerapp ingress access-restriction set -g $RG -n $APP --rule-name me \
   --ip-address $MYIP/32 --action Allow
az containerapp show -g $RG -n $APP --query properties.outboundIpAddresses -o tsv
#    ^ add these (and only these) to the Atlas IP access list.
```

Role assignments can take a few minutes to take effect. If the first request gets a 403 from
Search or OpenAI, wait, then restart the revision before you debug anything else.

### Security — this endpoint `eval()`s model output, so all of these are required

- [ ] **Atlas read-only user**, scoped to the 23 databases. Never the admin URI from `atlas-credentials.env`.
- [ ] **Atlas IP access list:** add the app's outbound IPs
  (`az containerapp show … --query properties.outboundIpAddresses`). Never `0.0.0.0/0`.
- [ ] **Ingress restricted to your IP** (`az containerapp ingress access-restriction set … --action Allow`),
  plus an API-key header check in the app. Widen it only for a live interview demo, then narrow it again.
- [ ] **The AST allowlist runs before every `eval()`.** The Atlas socket timeout already caps
  runaway queries (`atlas_env.py`). Reject questions over 500 characters.
- [ ] **Rate limit** of 10 requests a minute: the token bucket from `dsa-design-drill` exercise 05, written by you.
- [ ] Logs record a **hash** of the question, never the raw prompt. That's the house convention.

---

## Phase 6 — Observe, measure, write up (Day 14–16)

Do the measurements in **one sitting**: resume at the start (see Cost control) and pause at the end.

- [ ] **Liveness:** `curl -s https://<fqdn>/healthz`, then one real query:
  `curl -s -H "x-api-key: $KEY" -H 'content-type: application/json' -d '{"question":"How many singers are there?"}' https://<fqdn>/query`.
  Check that `database`, `query` and `rows` look right before measuring anything.
- [ ] **Smoke benchmark:** `service/smoke.py` replays **the first 50 cases of `rag/data/rag_test.json`**
  (a fixed, pre-declared slice, not ones you picked). For each case it records client-side
  wall time, the service's `latency_ms` per stage (embed / retrieve / generate / guard /
  execute), the returned query, and **`results_match(rows, gold)`** against
  `data/gold_results.json`. That last field makes the smoke run a small, served
  accuracy check too. Write `results/azure_smoke.json` plus a manifest (`run_manifest.write_manifest`)
  that records the image tag, revision name, model deployment and date. Report p50/p95 per stage and end to end.
  Space requests ≥ 6 s apart to stay under the 10/min rate limit, otherwise you measure the limiter.
- [ ] **Cold start:** wait until the app has scaled to zero, which takes several minutes
  (`az containerapp replica list -g $RG -n $APP` returns nothing). Then time the first request.
  Do this 3 times and append the timings to `results/azure_smoke.json` under `cold_start_s`. Write down the
  trade-off: `min-replicas 1` removes cold starts but runs outside the free grant.
- [ ] **Server-side cross-check:** in App Insights → Logs, get per-stage percentiles from the spans
  (for example `dependencies | where timestamp > ago(1d) | summarize percentiles(duration, 50, 95), count() by name`,
  adjusting the table and name to where your spans land). Export it to `results/azure_appinsights_latency.csv`.
  If client and server numbers disagree by more than network time, find out why before writing either down.
- [ ] **Cost per query** = Azure OpenAI tokens × price, plus the (≈0) Container Apps share.
- [ ] README: a new section, *Deployment on Azure*, with the architecture sketch, the numbers,
  and a link to the finding.
- [ ] Record a 60-second screen capture of a working query, so there's something to show after teardown.

---

## Phase 7 — Teardown (by Day 25)

- [ ] Export whatever you want to keep: App Insights query results, the index definition JSON, screenshots.
- [ ] `az group delete -n rg-capstone-rag --yes`. This single command is why everything went into one resource group.
- [ ] Delete the Atlas read-only user and remove the IP entries.
- [ ] Check Cost Management the next day and confirm spend has stopped.
- [ ] Keep `infra/deploy.sh`. It's the proof that you can redeploy in about 20 minutes.

---

## Cost control — what bills while you're not using it

The free trial has a spending limit, so a mistake there only stops services. **After the
Pay-As-You-Go upgrade nothing stops spending automatically**; budget alerts only send email. So
treat every session as *resume → work → pause*, and check the portal's Cost analysis
(filtered to `rg-capstone-rag`) daily.

| Resource | Bills while idle? | "Stop" | "Start" |
|---|---|---|---|
| AI Search **Free** | No | nothing to do | — |
| AI Search **Basic** | **Yes, every hour, and it can't be paused** | **Delete the service** (`az search service delete -g $RG -n $SRCH --yes`) | Recreate it, then `python rag/azure_index.py` (1,213 docs, seconds) |
| Container Apps (consumption, min 0) | No, once scaled to zero. **But any request wakes it**, and public URLs get scanned by bots | `az containerapp ingress disable -g $RG -n $APP` | `az containerapp ingress enable -g $RG -n $APP --type external --target-port 8000`, then re-apply the IP restriction |
| Container Apps environment (consumption only) | No | — | — |
| Container Registry Basic | **Yes, a small flat daily fee**; it can't be paused | only at teardown | — |
| Azure OpenAI, **Standard** deployment | No, pay per token | delete the deployment once Phase 6 is done | — |
| Azure OpenAI, *Provisioned* deployment | **Yes, hourly.** Never choose this type | — | — |
| Log Analytics / App Insights | Per GB ingested | the 0.1 GB/day cap from step 1 | — |
| Key Vault | Per operation (cents) | — | — |

Check current prices in the pricing calculator on the day. The point of the table is the **Yes** rows.

**End of every session (about 2 minutes):**
```bash
set -a; source azure.env; set +a
az containerapp ingress disable -g $RG -n $APP              # nothing can wake it now
az containerapp update -g $RG -n $APP --min-replicas 0      # in case you raised it for a test
az search service show -g $RG -n $SRCH --query sku.name -o tsv   # "basic"? then delete it unless you need it tomorrow
az resource list -o table                                   # whole subscription: anything you don't recognise?
```
Anything in that last list outside `rg-capstone-rag` (a VM, a disk, a public IP, a second
OpenAI resource) is a mistake. Delete it.

**Start of a session:** re-enable ingress and re-apply the IP restriction (your home IP may have
changed), then run `/healthz`. If you're on Basic search and deleted it, recreate the service and the index.

**Kill switch**, if an alert fires and you don't know why: `az group delete -n rg-capstone-rag --yes --no-wait`.
Everything is rebuildable from `infra/deploy.sh` and `rag/azure_index.py`. The results you care
about are already committed (Phase 3 step 6, Phase 6).

**Calendar:** Day 25 teardown, Day 30 credit expiry. Put both in your calendar on Day 0.

---

## What goes on the resume — only after the numbers exist

A template. Fill in the blanks from the finding, not from this plan:

> Deployed the RAG pipeline on Azure (Container Apps, AI Search, Azure OpenAI, Key Vault,
> managed identity) and A/B-tested hybrid BM25 + vector retrieval against FAISS on the same
> 304 held-out cases: ___ vs ___ execution accuracy, McNemar exact p = ___.

**Defensibility:** Phase 3 and its conclusions are yours and should be `D3`. For the
deployment you need `D2`: be able to draw the diagram above from memory, explain the managed
identity flow, and say why each security control exists.

## Interview questions this creates — have an answer

- *Why exhaustive KNN rather than HNSW?* Exactness keeps the comparison clean, and scale makes it free.
- *Why was A1 necessary?* A harness that has never reproduced a known result can't be trusted to measure a new one.
- *Why hold the generator fixed?* Tell the Colab→MLX confound story. You have lived this one.
- *How does RRF work, and what does it ignore?* It fuses ranks, not scores, so it's insensitive to how the two retrievers scale their scores.
- *You `eval()` model output behind a public URL?* Walk through the AST allowlist, read-only
  user, IP allowlist, timeouts and rate limit. Say which one you'd trust least.
- *Scale-to-zero?* Cold-start latency against cost, with your measured numbers.
- *Hosted vs local generator?* G1's accuracy delta and cost per query.

---

## Sources

- [Free trial: 0 LLM quota until the Pay-As-You-Go upgrade (Microsoft Q&A)](https://learn.microsoft.com/en-gb/answers/questions/5907451/free-trials-get-200-but-have-a-0-quota-limit-on-ll)
- [Azure AI Search service limits](https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity) · [vector index size, and why exhaustive KNN doesn't use vector quota](https://learn.microsoft.com/en-us/azure/search/vector-search-index-size)
- [Container Apps billing and the monthly free grant](https://learn.microsoft.com/en-us/azure/container-apps/billing)
- [Azure free account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account)
