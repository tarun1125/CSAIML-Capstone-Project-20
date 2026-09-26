# Capstone (NL-to-MongoDB) — interview study guide

**Current tag: `D3` — your own work, the most defensible thing in the portfolio.**
It leads the resume and every F3 (agentic/RAG) application. Study here is
consolidation, not rescue: you know this material, but the numbers moved and the headline
conclusion reversed, so the *story* needs relearning even where the work doesn't.

---

## 0. The floor

> **What it is.** A benchmark for natural-language → MongoDB query generation. 1,517 cases
> across 23 databases — 1,213 as a few-shot retrieval pool, 304 held out — with every
> generated query on the held-out set asserted by **executing it against a live Atlas cluster
> and comparing returned rows**, not by string-matching a reference.

> **Why that matters.** String comparison is wrong in both directions: it marks a correct
> query wrong for using `find()` where the reference used `aggregate()`, and it can mark a
> subtly wrong query right for looking similar. **Finding a cheap oracle is most of the work
> in testing a system like this.**

🔴 **Never say "1,517-case benchmark asserted by an execution oracle."** 1,517 is the dataset;
the oracle asserts **304**. It's the first thing an interviewer unpicks.

---

## 1. The current numbers — relearn these, they changed

**Results on the full 304-case test set, unified MLX-LM stack:**

| Arm | Execution accuracy |
|---|---|
| Zero-shot baseline | **4.9%** (15/304) |
| **RAG, K=10, FK-included** | **46.7%** (142/304) ← the leader |
| LoRA fine-tuned (23-db, 1000-iter epoch-parity retrain) | **33.9%** (103/304) |

🔴 **Retired: "LoRA outperformed every RAG configuration."** RAG leads by ~13 points. The old
24.6% / 32.8% pair came from a different, confounded setup — see §2.

**By complexity (RAG):** easy 60.0% · medium 40.5% · hard 28.2%. And **79 of 304 cases have no
`complexity` value** — a genuine data-quality gap from the newest expansion round, surfaced as
its own bucket rather than dropped or force-fit. Say that if the breakdown comes up; how you
handled a missing label is more interesting than the label.

**The arms fail on different cases, not just at different rates:** both correct 64, RAG-only
77, fine-tuned-only 39, neither 124 — **union 180/304 (59.2%)**, well above either alone.
That's an argument for ensembling, and it's a better answer than "RAG won."

---

## 2. The confound ⭐ lead with this

**This is the strongest story in the whole portfolio. Have it word-perfect.**

Baseline and RAG were generated on Colab with HF `transformers` in **fp16 on a shared GPU**;
fine-tuning ran locally on **MLX-LM in bf16**. Three arms, two serving stacks — so every
cross-arm comparison varied the model *and* the stack.

Re-measured on one stack, **RAG went 24.6% → 50.8% on the identical, unchanged 61-case matched
slice** — same questions, same gold, same scoring code, only the serving stack changed.
**That reversed the project's previously reported headline conclusion, and the reversal was
published.**

**And it was diagnosed, not assumed.** fp16 GPU inference without deterministic-algorithm
flags is not bit-identical across separate Colab sessions, even under greedy decoding —
verified by making a prompt-content change and its **full revert back-to-back in one session**
and getting byte-for-byte identical output. Identical output within a session but different
output across sessions means **a session boundary, not the code, drove the earlier swing.**

### Why this is worth more than the original claim
- It's a **negative result about your own work**, found by you and published.
- The diagnosis is a real experiment, not a hypothesis.
- It generalises: *"if two arms of a comparison run on different stacks, you are not measuring
  what you think you're measuring."*

**Rehearse the 45-second version.** It answers "tell me about a time you were wrong",
"what's the hardest bug you've found", and "how do you know your benchmark is sound" — three
questions with one story.

---

## 3. Retrieval, and the reranking null ⭐ the resume lead

### 3.1 The pipeline
Embed the NL question with all-MiniLM-L6-v2 → FAISS `IndexFlatIP` over L2-normalised vectors
(exact cosine, brute force — correct at 1,213 items; IVF/HNSW would be over-engineering two
orders of magnitude early) → top-K exemplars → majority vote over their databases picks the
schema card → prompt → generate.

**Only the question text is embedded**, not the query or schema — a deliberate choice: the
retrieval job is "find past questions phrased like this one", and mixing in query syntax
blurs that signal. **§3.3 partially overturns this.**

### 3.2 The reranking result

**Every ranking metric improved. Execution accuracy did not.**

| Arm | recall@5 | MRR@10 | nDCG@10 | execution accuracy |
|---|---|---|---|---|
| Bi-encoder K=10 | 0.9638 | 0.9288 | 0.8928 | **143/304 = 47.0%** |
| + rerank, K=10 | 0.9836 | 0.9493 | 0.8987 | 141/304 = 46.4% |
| + rerank on question+query, K=5 | 0.9803 | 0.9510 | **0.9088** | 131/304 = 43.1% |

- **McNemar exact p = 0.8776** at K=10 — 42 of 304 flipped, 22 lost / 20 gained.
- **Not a weak intervention:** ~40% of exemplars in every prompt changed; identical ordering
  on 0/304 cases. It was pure churn with respect to correctness.
- **nDCG vs correctness: r = +0.307, p = 4.7e-08.** Real but weak — ranking explains
  **9.4% of the variance**. The other 91% is the generator.
- **Goodhart, observed:** the arm with the *best* nDCG (0.9088) has the *weakest* correlation
  to correctness (r = 0.200, down from 0.307). **Pushing the ranking metric up made it a worse
  predictor of the thing anyone cares about.**
- **A positive control that fires:** rank-1-vote vs bi-encoder, 6 discordant, 0 lost / 6
  gained, **p = 0.0313.** The harness detects real effects — the null isn't a measurement
  failure.
- **Why there was no room:** bi-encoder recall@10 was already 0.987 and MRR@10 0.929 *before*
  reranking. A 1,213-exemplar pool over 23 databases, queried in-distribution, is an easy
  retrieval problem. **If retrieval is at 0.99 recall, no reranker rescues 46% task accuracy —
  the failures are downstream.**

### 3.3 The second arm corrected the index's own reasoning
For a **bi-encoder** compressing both signals into one 384-dim vector, "query syntax blurs the
semantic signal" holds. For a **cross-encoder** attending across both texts jointly it does
not: reranking on `question + normalized_query` beat question-alone on nDCG@10 (0.909 vs
0.899), recall@1 and db-vote@10. *The original reasoning is sound for the architecture it was
written about and doesn't generalise to a joint encoder.*

### 3.4 Know the labels
- **Binary relevance** (recall, MRR): exemplar and case share a database. Same notion
  `db_match_count` already used, generalised from "appeared in top-K" to "where in top-K".
- **Graded relevance** (nDCG): 0 different db / 1 same db no collection overlap / 2 partial
  overlap with `gold_collections` / 3 exact collection-set match.
- **Same-database is a proxy for relevance, not relevance.** A cross-database exemplar showing
  the right aggregation shape may help more than a same-database one that doesn't. Say so.
- **recall@10 ≠ db_match.** 0.987 vs 0.849 — a majority vote can lose 4–3–3 with the correct
  database at rank 1. Both are correct; they measure different things.

---

## 4. Other findings worth having ready

- **FK schema annotations: A/B tested, null result.** 142/304 (46.7%) with vs 141/304 (46.4%)
  without. A 1-case, 0.3-point difference. *FK annotations don't measurably help or hurt on
  top of what schema + few-shot examples already provide.*
- **A data-integrity catch:** the K=10 baseline file was the **no-FK** arm (documented in
  `CANONICAL_ARTIFACTS.md` — the FK variant's case-level JSON was overwritten by the A/B run),
  so the original reranking comparison varied FK *and* the exemplar list. Re-scored to
  143/304. Finding that before publishing is the point.
- **LoRA:** rank 16, 1000 iterations, epoch-parity fix. Freeze base weights, learn a low-rank
  update on the attention projections; cheap because you train r×d instead of d×d.

---

## 5. Break it on purpose

| Break | Expect | Teaches |
|---|---|---|
| Score with string comparison instead of execution | Correct queries marked wrong | Why the oracle exists |
| Set K=1 | Accuracy drops; db-vote becomes rank-1 | What the vote is doing |
| Retrieve from a different database's pool only | Large drop | How much of RAG's gain is schema selection vs exemplar style |
| Run RAG and fine-tuned on *different* stacks again | The old conclusion reappears | The confound, felt |
| Remove pre-whitening equivalent / compare trending series | — | (see ARBITER's guide) |

The third one is the most interesting and you haven't run it: **how much of RAG's 46.7% is
picking the right schema card, versus the exemplars themselves?** That's a one-afternoon
ablation and it would sharpen every RAG answer you give.

---

## 6. Questions you will get

1. **"Walk me through the RAG system end to end."** §3.1, landing on the oracle.
2. **"How do you know retrieval is any good?"** §3.2 — and the honest answer that it was
   already saturated.
3. **"What metrics do you track for retrieval?"** recall@k, MRR@10, nDCG@10 with the graded
   label — and why you *also* went to the execution oracle, because ranking explained only
   9.4% of the variance.
4. **"Tell me about a time you were wrong."** §2.
5. **"Why didn't fine-tuning win?"** §1 + §2 — and that the arms fail on different cases.
6. **"What would you do next?"** The schema-vs-exemplar ablation (§5), and an ensemble given
   the 180/304 union.

---

## 7. Do not say

- ❌ "1,517-case benchmark asserted by an execution oracle" — 1,517 is the dataset, 304 is the
  oracle's caseload.
- ❌ "LoRA outperformed every RAG configuration" — reversed.
- ❌ "4.9% → 24.6%" or "32.8%" — stale, from the confounded setup.
- ❌ Claiming recall@k/MRR were always measured — they were added 2026-09-22, and the honest
  version ("I added ranking metrics and they didn't predict the task") is the better story.

---

## 8. The Azure arms (added 2026-09-26) — what holds the tag

**Tags, confirmed by Tarun 2026-09-26:** the two experiments `D3`, the deployment `D2`. They're now on
the resume. Holding them means being able to do all of this **without notes**. Re-check before
every technical round:

**Retrieval (`FINDING-azure-retrieval.md`)**
- [ ] Why Gate 1 existed, why exhaustive KNN rather than HNSW, and why 298/304 was reported as
  **missed** even though every mismatch is one exact tie.
- [ ] How RRF fuses BM25 and vector ranks, and why it doesn't care about score scales.
- [ ] Why better recall/MRR/nDCG didn't move execution: 26 of 33 flips had the same database, so the
  change was exemplar churn. This is the third time you've seen that pattern.
- [ ] The false positive in the gains (`store_1-25`: right row, wrong field). What the oracle can't see.

**Generator (`FINDING-azure-generator.md`)**
- [ ] Why the prompts file was held byte-identical, and the Colab confound as the reason it mattered.
- [ ] 136 → 192 of 258 with the right database; 7 vs 6 with the wrong one. Explain why this makes the
  database decision the binding constraint.
- [ ] How the +14 for rank-1 splits into about +12 from the database fix and about +2 noise (the 277
  byte-identical prompts).
- [ ] Why T = 0 isn't deterministic on a hosted model, and how you reported it (~198 ± a few).
- [ ] The Spider-contamination caveat, volunteered before you're asked.

**Deployment (README *Deployment on Azure*)**
- [ ] Draw the diagram from memory: every hop, and the identity it runs as.
- [ ] Why a NAT Gateway (a static egress IP for the Atlas access list), and what it costs while idle.
- [ ] Every security control on the `eval()` path, and which one you'd trust least.
- [ ] Cold start (34.5 s) against `min-replicas 1`: the cost trade-off, with numbers.

**Do not say**
- ❌ "Azure AI Search improved accuracy." It didn't: 148 vs 143, p = 0.49. It's an infrastructure choice.
- ❌ "gpt-4o scored 198" as an exact number. Say ~198 ± a few; it isn't bit-reproducible.
- ❌ "Production." It's a deployed demo service behind an IP allowlist, with no real users.
