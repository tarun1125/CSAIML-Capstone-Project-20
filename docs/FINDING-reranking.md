# Finding — cross-encoder reranking, and whether ranking metrics predict task accuracy

**Spec:** [`docs/EXPERIMENT-reranking.md`](EXPERIMENT-reranking.md)
**Status:** complete, 2026-09-22. All five arms run end to end on the same 304 held-out cases.
**Artifacts:** `results/retrieval_eval.json` (+ `.manifest.json`)

> **On the K=10 baseline, and a correction.** An earlier version of this write-up used
> `rag/data/qwen_rag_execution_results_mlx.json` (141/304) as the K=10 baseline. **That was
> wrong**: as [`CANONICAL_ARTIFACTS.md`](../CANONICAL_ARTIFACTS.md) documents, the FK
> variant's case-level JSON was overwritten by the FK-vs-no-FK A/B run, so the surviving
> file is the **no-FK** arm. Every arm in this experiment is built with FK annotations
> **on**, so that file varied FK *as well as* the exemplar list and no difference could be
> attributed to either. The K=10 baseline is now re-scored from
> `rag/data/qwen_rag_mlx_fk_normalized.json` into
> `rag/data/qwen_rag_fk_k10_execution_results.json` — **143/304 (47.04%)**, which also
> closes the 1-case gap that file's absence had left open. The K=5 rows were never
> affected; that baseline was generated here with FK on.
>
> The correction does not change this experiment's conclusion — the K=10 reranking null
> went from 24/24 (p=1.0000) to 22/20 (p=0.8776) — but every number below is the corrected
> one.

## 0. The headline

**Every ranking metric improved. Execution accuracy did not.**

The spec anticipated two outcomes and asked that this one be reported loudly if it
happened. It happened, and it is cleaner than expected:

| Arm | recall@5 | MRR@10 | nDCG@10 | **execution accuracy** |
|---|---|---|---|---|
| Bi-encoder, K=5 | 0.9638 | 0.9288 | 0.8928 | **141/304 = 46.38%** |
| + rerank from top-50, K=5 | 0.9836 | 0.9493 | 0.8987 | **129/304 = 42.43%** |
| + rerank on question+query, K=5 | 0.9803 | 0.9510 | 0.9088 | **131/304 = 43.09%** |
| Bi-encoder, K=10 *(current default)* | 0.9638 | 0.9288 | 0.8928 | **143/304 = 47.04%** |
| + rerank from top-50, K=10 | 0.9836 | 0.9493 | 0.8987 | **141/304 = 46.38%** |

Reranking moved recall@5 up 2 points, MRR@10 up 2 points, nDCG@10 up 0.6–1.6 points,
and recall@10 to a perfect 1.000. Execution accuracy went **nowhere at K=10 and
down ~4 points at K=5**. Not one arm beat the bi-encoder baseline.

### The cleanest number in the experiment

At K=10, reranking flipped the outcome on **42 of 304 cases** — 22 lost, 20 gained.
McNemar's exact p = **0.8776**.

That is not "the intervention was too small to measure". Reranking changed ~40% of
the exemplars in every prompt and changed the final answer on one case in six. It
is that the change was **pure churn with respect to correctness** — a coin flip, to
three significant figures.

---

## 1. What this means

**Ranking quality is not the binding constraint on this pipeline.** Optimising the
metric the retrieval field defaults to would have been wasted effort here, and the
experiment cost about 40 minutes of compute to establish that.

The per-case correlation says the same thing quantitatively. On the baseline arm,
per-case nDCG@10 against per-case execution correctness gives:

| | r | p | mean nDCG when correct | mean nDCG when wrong |
|---|---|---|---|---|
| nDCG@10 vs correctness | **+0.307** | 4.74e-08 | 0.950 | 0.842 |
| MRR@10 vs correctness | +0.274 | 1.17e-06 | — | — |

The relationship is **real but weak**: p ≈ 1e-07 is not noise at n=304, and
r = 0.307 means ranking quality explains **9.4% of the variance** in whether the
generated query returns the right rows. The other 91% lives in the generator, not
the retriever.

### The detail that sharpens it

Correlation with correctness **weakens as nDCG improves**:

| Arm | nDCG@10 | r (nDCG vs correctness) |
|---|---|---|
| Bi-encoder, K=10 | 0.8928 | **+0.307** |
| + rerank (question), K=10 | 0.8987 | +0.259 |
| + rerank (question), K=5 | 0.8987 | +0.254 |
| + rerank (question+query), K=5 | **0.9088** | **+0.200** |

The arm with the **best** nDCG has the **weakest** link to task success. Pushing the
ranking metric up made it a *worse* predictor of the thing anyone actually cares
about. That is Goodhart's law showing up inside a single afternoon's experiment,
and it is the strongest argument in this write-up against treating nDCG as a cheap
proxy for execution accuracy on this task.

**This is the same shape as this repo's calibration finding**: the measurement was
fine, the thing being measured was not the thing that mattered.

---

## 2. Ranking metrics in full

All three arms rerank the **same** FAISS top-50 candidate set for the same 304
cases, so the only variable is the ordering.

| Arm | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | nDCG@10 | db-vote@5 | db-vote@10 |
|---|---|---|---|---|---|---|---|---|
| Bi-encoder *(baseline)* | 0.9046 | 0.9375 | 0.9638 | 0.9868 | 0.9288 | 0.8928 | **0.8849** | 0.8487 |
| + rerank on question | 0.9243 | **0.9770** | **0.9836** | **1.0000** | 0.9493 | 0.8987 | 0.8816 | 0.8618 |
| + rerank on question + query | **0.9309** | 0.9671 | 0.9803 | 0.9934 | **0.9510** | **0.9088** | 0.8816 | **0.8684** |

- **Binary label** (recall, MRR): exemplar and case share a database.
- **Graded label** (nDCG): 0 different db / 1 same db no collection overlap /
  2 partial overlap with `gold_collections` / 3 exact collection-set match.
- **db-vote** is not a ranking metric — it is `build_prompts.py`'s majority vote over
  the top-K neighbours' databases, which selects *which database's schema gets pasted
  into the prompt*. It is the mechanism by which a reranking changes the prompt rather
  than just the ranking.

**The reranker was doing its job.** The cross-encoder is not broken and it is not a
no-op; it genuinely produces better rankings under every metric asked of it.

### Why reranking hurt at K=5 but not K=10

Look at the db-vote column: reranking helps the vote at K=10 (0.849 → 0.862 / 0.868)
and **hurts it at K=5** (0.885 → 0.882). Better ranking concentrates good exemplars at
the top, which is what recall@1 and MRR reward — but a majority vote over 5 items does
not want concentration, it wants a representative sample. Sharpening the top of the
list makes a short vote *more* exposed to the reranker being wrong about rank 1.

The per-case breakdown shows churn, not improvement:

| Arm | predicted database changed | fixed | broken |
|---|---|---|---|
| rerank-Q, K=5 | 21 | +8 | −10 |
| rerank-Q+query, K=5 | 28 | +11 | −13 |
| rerank-Q, K=10 | 23 | +12 | −8 |

### The second arm corrects the index's documented reasoning

`build_retrieval_index.py` argues that mixing query syntax into the embedded text
"blurs the semantic signal". For a **bi-encoder**, forced to compress both signals
into one 384-dim vector, that may well hold. For a **cross-encoder** it does not:
reranking on `question + normalized_query` beats question-alone on nDCG@10 (0.909 vs
0.899), recall@1 (0.931 vs 0.924) and db-vote@10 (0.868 vs 0.862).

The original reasoning is sound *for the architecture it was written about*, and does
not generalise to a joint encoder. Worth recording — though note that this arm also
has the weakest correlation to execution correctness, so "better" here buys nothing
downstream.

---

## 3. The paired comparison

At n=304 the 95% interval on a single proportion is about ±5.5 points, which would
swallow every effect in the table above. But the arms answer the **same** 304
questions with the same model, decoding, schema cards and scorer — differing only in
the exemplar list. So the test is McNemar's exact over the discordant cases.

| Comparison | both right | both wrong | only baseline | only arm | discordant | exact p |
|---|---|---|---|---|---|---|
| rerank-Q K=5 vs bi-encoder K=5 | 112 | 146 | **29** | 17 | 46 | 0.1038 |
| rerank-Q+query K=5 vs bi-encoder K=5 | 112 | 144 | **29** | 19 | 48 | 0.1934 |
| rerank-Q K=10 vs bi-encoder K=10 | 121 | 141 | **22** | 20 | 42 | **0.8776** |
| *(for contrast)* rank-1 vote K=10 vs bi-encoder K=10 | 143 | 155 | **0** | **6** | 6 | **0.0313** |

No arm is significantly worse. **No arm is even numerically better.** At K=5 the
direction is consistently negative (29 lost vs 17–19 gained); at K=10 it is a dead
heat. The honest summary is: reranking bought nothing, and may have cost a little at
small K.

**Discordance is high — 42–48 cases, one in seven.** The prompts really did change:

| Comparison | identical order | identical set | mean top-K overlap |
|---|---|---|---|
| K=5 bi-encoder vs rerank-Q | 2/304 | 23/304 | 0.608 |
| K=5 bi-encoder vs rerank-Q+query | 6/304 | 25/304 | 0.619 |
| K=10 bi-encoder vs rerank-Q | 0/304 | 4/304 | 0.657 |

About 40% of the exemplars in each prompt changed. The null result is **not** an
artifact of a weak intervention.

### Why there was so little room

The bi-encoder already scored recall@10 = 0.987 and MRR@10 = 0.929 before anything
was reranked. **The retrieval stage was close to saturated when the experiment
started.** Across all 304 cases, a same-database exemplar appeared within the top 50
every single time, and the median rank of the first relevant exemplar was 1 for every
arm including the baseline.

That is a finding about the corpus, not about the reranker: a 1,213-exemplar pool
over 23 databases, queried by questions drawn from the same distribution, is an easy
retrieval problem. **If retrieval is already at 0.99 recall, no reranker can rescue a
46% task accuracy** — the failures are downstream.

---

## 4. Cost

| Stage | Cost |
|---|---|
| Cross-encoder scoring, 304 × 50 = 15,200 pairs | **17 s** per arm, CPU |
| Ranking metrics, all three arms | ~1.5 s |
| Generation, per execution arm (304 cases, greedy, 300 tok) | ~6.5 min |
| Execution against Atlas, per arm | ~3 min |
| **Whole experiment, end to end** | **~45 min** |

The spec budgeted "minutes on Apple Silicon" for the reranker; 17 seconds is an order
of magnitude cheaper than that. Scores are cached to `rag/data/rerank_scores.json`,
keyed by variant and `(case_id, exemplar_id)`, so re-running the metrics never
re-runs the model.

**The asymmetry is what made the proxy question worth asking**: ranking evaluation is
~17 seconds, a full execution arm is ~10 minutes — a 35× difference. The answer, for
this pipeline, is that the cheap metric does not substitute for the expensive one.

---

## 5. Two errors in the spec

**§3's sanity check was wrong as written.** It asked that recall@10 under the binary
label equal `build_prompts.py`'s `db_match_count / 304` at TOP_K=10. Those are
different quantities and do not agree:

- recall@10 = **0.9868** — "at least one same-database exemplar landed in the top 10"
- db_match = **0.8487** — "the majority vote over those 10 databases picked the right one"

A vote can lose 4–3–3 with the correct database sitting at rank 1. Both numbers are
right; the check compared a hit-rate against a vote. Running it as specified fails and
sends you hunting a bug that does not exist.

The check that tests what §3 *intended* — "am I reading the same retrieval the
pipeline read?" — reconstructs the majority vote from the ranked list and compares it
to the recorded 0.8487. It is implemented, runs before anything else, **raises and
halts** on mismatch, and passes to 1e-9.

**§2 omitted that `gold_collections` is inconsistently qualified.** 331 of the 450
entries across the 304 test cases are written `car_1.cars_data`; the other 119 are
bare (`Department`). `collections_in` returns bare names, so without normalisation
two-thirds of cases never match and every graded label silently collapses to 1 —
producing a plausible, wrong nDCG with no error anywhere. Handled in
`relevance.gold_collections_of`, which strips the prefix only when it matches the
case's database, splitting on the **first** dot: `car_1.model_list.json` must become
`model_list.json`, and an rsplit yields `car_1.model_list`, which is not a collection.

**And one dependency correction.** The spec specified `sentence-transformers`, which
is not in this venv; installing it would add a fourth copy of `libomp.dylib` alongside
torch's, faiss's and sklearn's — the exact hazard the load-bearing import-order
comments exist because of. `rag/rerank.py` uses raw `transformers`
(`AutoModelForSequenceClassification`, num_labels=1), which is what
sentence-transformers' `CrossEncoder` wraps, so the scores are comparable to anything
published with that class. Same call `embed_utils.py` already made for the bi-encoder.

### A latent quirk in existing code

`build_prompts.py:majority_vote_database` does not do what its own docstring says. On
a tie it returns `neighbor_dbs[0]` without checking that this database is among the
tied ones: for neighbours `[A, B, B, C, C]` the tied set is `{B, C}` but it returns
`A`, a database that lost outright.

It fires on exactly **3 of 304** cases, and on all three the rank-1 database happened
to be correct — the entire difference between the recorded 258/304 (0.8487) and the
255/304 (0.8388) a tie-break restricted to the tied set gives. It is **replicated, not
corrected**, in `eval_retrieval.py`: this experiment measures the pipeline that
exists, and changing the vote while also changing the ranking would confound the two.
Fixing it deserves its own before/after.

---

## 6. Caveats

1. **This is exemplar retrieval, not document retrieval.** Retrieving few-shot examples
   to condition a generator is not the same task as retrieving passages containing an
   answer. The machinery and metrics transfer; the interpretation does not, entirely.
   An exemplar's job is partly to demonstrate *form*, and none of these labels can see
   form — which is one plausible reason better same-database ranking bought nothing.
2. **Same-database is a proxy label.** A cross-database exemplar showing the right
   aggregation shape may condition the model better than a same-database one that does
   not. The graded label narrows this; it does not close it. The execution oracle is
   what settles it, which is exactly what §5 did.
3. **n = 304.** Hence the paired analysis rather than two accuracy numbers.
4. **The reranker is off-the-shelf**, not fine-tuned on this domain. A fine-tuned one
   would rank better — but given that better ranking produced *no* execution gain and a
   *weaker* correlation, there is little reason to expect a fine-tuned one to change the
   conclusion. It would have to beat the bi-encoder on something other than ranking.
5. **nDCG is normalised against each arm's own retrieved candidates**, not the best
   possible ordering of the 1,213-exemplar pool. nDCG here measures *ordering* quality
   and recall@k measures *retrieval* quality; conflating them would let a reranker take
   credit for a candidate set it did not choose.
6. **The null is specific to this pipeline at this saturation level.** On a corpus where
   the bi-encoder recalled 0.6 rather than 0.99, reranking would plausibly matter a
   great deal. The claim is "ranking was not the binding constraint *here*", not
   "reranking never helps".
7. **`rag/data/rag_prompts_k5.json` was regenerated** from a stale 61-case artifact
   (pre-dating the 304-case expansion) to the current 304-case split. The old file is in
   git history.

---

## 7. What to do with this

- **Do not spend further effort on retrieval ranking for this pipeline.** It is
  saturated, and improving it demonstrably does not move the task metric.
- **Do not adopt nDCG/MRR as a cheap proxy for execution accuracy here.** r = 0.307
  at baseline, falling to 0.200 on the best-ranked arm.
- **The retrieval headroom is in database prediction, not exemplar ranking.** Replacing
  the majority vote with pure rank-1 gained 6 cases and lost none (p = 0.031) — see
  [`docs/FINDING-rank1-vote.md`](FINDING-rank1-vote.md).
- **The ~47% ceiling is mostly a generation problem.** 141 of 304 cases were wrong
  under *both* K=10 reranking arms regardless of which exemplars were supplied.
  That set is where the next experiment belongs.
- **Worth reporting as a negative result.** Most published reranking work stops at
  nDCG because it has no task-level oracle to check against. This repo has one, used
  it, and the metric did not survive contact with it.

---

## 8. Reproducing

```bash
python rag/rerank.py                        # arm 1 — question only (~17 s)
python rag/rerank.py --include-query         # arm 2 — question + query (~17 s)
python rag/eval_retrieval.py --emit-neighbors
python -m pytest tests/test_metrics_retrieval.py -q

# §5 execution arms
python rag/build_prompts.py 5
python rag/build_prompts.py 5  --rerank rag/data/reranked_neighbors_rerank_question_only.json rerankQ
python rag/build_prompts.py 5  --rerank rag/data/reranked_neighbors_rerank_question_plus_query.json rerankQQ
python rag/build_prompts.py 10 --rerank rag/data/reranked_neighbors_rerank_question_only.json rerankQ

for arm in k5 rerankQ_k5 rerankQQ_k5 rerankQ_k10; do
  python rag/generate_rag_mlx.py \
    --prompts-path rag/data/rag_prompts_${arm/k5/k5}.json \
    --output rag/data/qwen_rag_mlx_${arm}_results.json
done

python rag/score_rerank_arms.py              # needs Atlas
python rag/eval_retrieval.py --with-execution
```

Everything but the exemplar list is held fixed across arms: same model
(`mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16`), greedy decoding, max_tokens 300,
same schema cards, FK annotations on, same gold, same scorer
(`evaluation/execute_queries.run_model`, imported not reimplemented).
