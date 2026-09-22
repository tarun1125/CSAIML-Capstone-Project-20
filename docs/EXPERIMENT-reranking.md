# Experiment — cross-encoder reranking, and whether ranking metrics predict task accuracy

**Status:** specified 2026-09-22, not yet run.
**Why it exists:** the retrieval stage is currently evaluated by one number —
`build_prompts.py`'s `database retrieval accuracy` — and by downstream execution accuracy.
Neither tells you *how well ranked* the retrieved exemplars are, and nothing here has ever
tried reranking them.

The interesting part is not the reranker. It is that **this repo can do something most
reranking work cannot: check whether the ranking metrics actually predict the task.**
Almost every reranking result in the wild reports nDCG or MRR and stops, because there is no
task-level ground truth to check against. Here there is one — the execution oracle.

---

## 1. What already exists

| Piece | Where | Note |
|---|---|---|
| Exemplar pool, 1,213 items | `rag/data/fewshot_metadata.json` | `{id, question, database, complexity, normalized_query}` |
| FAISS index | `rag/data/fewshot.index` | `IndexFlatIP` over L2-normalised vectors = exact cosine. Brute force, correct at this scale. |
| Embedding fn | `rag/embed_utils.embed` | all-MiniLM-L6-v2. **Embeds the `question` text only** — a deliberate choice documented in `build_retrieval_index.py`. |
| Held-out test set, 304 cases | `rag/data/rag_test.json` | `{id, question, database, complexity, gold_collections, normalized_query, source}` |
| Retrieval call | `rag/build_prompts.py:263` | `scores, idxs = index.search(qvec, TOP_K)`, `TOP_K` overridable from argv |
| Existing retrieval metric | `rag/build_prompts.py:342` | `db_match_count / len(test_cases)` — logged, not persisted |
| Execution oracle | `evaluation/execute_queries.py`, `execute_gold.py` | Runs generated queries against the live cluster, compares returned rows |

**Two labels are already in the data. Neither needs new annotation.**

- **Binary relevance** — an exemplar is relevant if `exemplar["database"] == case["database"]`.
  This is not invented for the experiment; it is the same notion `db_match_count` already
  uses, generalised from "did it appear at all in top-K" to "where did it appear".
- **Graded relevance** — `rag_test.json` already carries **`gold_collections`**. Grade an
  exemplar by how much its `normalized_query`'s collection set overlaps the case's
  `gold_collections`. This is the label that makes nDCG meaningful.

---

## 2. Relevance labels — define once, in one module

New file: `rag/relevance.py`. Nothing else may define these.

```python
def collections_in(normalized_query: str) -> set[str]:
    """Collections a normalized query touches: the db.<coll> root plus every
    $lookup `from`. Parse it, don't regex the whole string -- a $lookup inside
    a $facet is still a read of that collection and a shallow regex will miss
    the nesting."""

def binary_relevance(exemplar: dict, case: dict) -> int:
    """1 if same database, else 0. The existing db_match notion."""
    return int(exemplar["database"] == case["database"])

def graded_relevance(exemplar: dict, case: dict) -> int:
    """0-3, for nDCG.
      0  different database
      1  same database, no collection overlap
      2  same database, partial overlap with case["gold_collections"]
      3  same database, exemplar's collection set == gold_collections
    """
```

> **Write the tie-break down and keep it.** Two exemplars at grade 3 are ordered by index
> position, not by score, so the metric is reproducible across runs.

> **State the limitation in the write-up, don't bury it.** Same-database is a *proxy* for
> relevance, not relevance itself. A cross-database exemplar demonstrating the right
> aggregation shape may help the model more than a same-database one that doesn't. The
> graded label narrows this but does not close it, and the execution oracle in §5 is what
> actually settles it.

---

## 3. Ranking metrics

New file: `rag/metrics_retrieval.py`. Pure functions over a ranked list of relevance grades —
no FAISS, no I/O, so they are unit-testable against hand-worked examples.

| Metric | Report at | Definition to implement |
|---|---|---|
| `recall@k` | k = 1, 3, 5, 10, 20 | fraction of cases with ≥1 relevant exemplar in the top k |
| `MRR@10` | — | mean of 1/rank of the **first** relevant exemplar; 0 if none in top 10 |
| `nDCG@10` | — | graded, standard log₂ discount, normalised by the ideal ordering |
| `mean_rank_first_relevant` | — | diagnostic; report the median too, the distribution is skewed |

**Unit tests first.** Hand-compute MRR and nDCG for three ranked lists and assert against
them. A silently wrong nDCG is the easiest way to produce a confident wrong conclusion here,
and it will not look wrong.

**Sanity check that must pass before anything else runs:** `recall@10` under the binary label
must equal the existing `db_match_count / 304` from `build_prompts.py` at `TOP_K=10`. If it
does not, one of the two is wrong — find out which before continuing.

---

## 4. The reranker

New file: `rag/rerank.py`.

```
retrieve top-N with FAISS (N = 50)   ->   cross-encoder scores (question, exemplar_question)
                                     ->   re-sort   ->   take top-K
```

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`. Standard
  baseline, small, CPU-fine.
- Cost: 304 × 50 ≈ **15,200 pairs**. Minutes on Apple Silicon. Cache scores to
  `rag/data/rerank_scores.json` keyed by `(case_id, exemplar_id)` so re-running the metrics
  never re-runs the model.
- **Import order:** `embed_utils` (torch) before `faiss`, same as
  `build_retrieval_index.py`. Three copies of `libomp.dylib` in this venv; wrong order
  segfaults with no traceback. Do not let isort merge the blocks.

**The reason a cross-encoder can beat the bi-encoder here is specific, and it is worth
writing down**: the index embeds the `question` text *only* — a documented, deliberate choice.
A cross-encoder sees both texts jointly and can attend across them, so it can use signal the
bi-encoder's single-vector bottleneck discards. If reranking does **not** help, that is
evidence the bi-encoder was already saturating this pool, which is a finding about the corpus
and should be reported as one.

### Second arm, cheap and worth having

Rerank on `question + normalized_query` rather than question alone. This directly tests the
choice `build_retrieval_index.py` documents — that adding query syntax "blurs the semantic
signal". A cross-encoder may not suffer that blurring the way a bi-encoder does. One extra
run, and it either confirms the original reasoning or corrects it.

---

## 5. The part that matters — does ranking predict the task?

Ranking metrics are the setup. **This is the experiment.**

Take the reranked top-K, feed it through the existing pipeline unchanged —
`build_prompts.py` → generation → `evaluation/execute_queries.py` → execution accuracy on the
same 304 held-out cases — and put it beside the baseline.

### The table to fill

| Arm | recall@5 | MRR@10 | nDCG@10 | **execution accuracy** |
|---|---|---|---|---|
| Bi-encoder, K=5 *(baseline)* | | | | |
| + rerank from top-50, K=5 | | | | |
| + rerank on question+query, K=5 | | | | |
| Bi-encoder, K=10 *(current default)* | | | | |
| + rerank from top-50, K=10 | | | | |

**Hold everything else fixed.** Same model, same decoding params, same seed, same schema
cards, same FK setting. The only variable is the exemplar list. Record the manifest the way
the other runs here do, so the comparison is reproducible.

### Both outcomes are results

- **Ranking metrics improve and execution accuracy follows** — clean win, and you have
  validated MRR/nDCG as a cheap proxy for this task. That is worth stating: it means future
  retrieval work can iterate on ranking metrics without paying for a full execution run.
- **Ranking metrics improve and execution accuracy does not** — the more interesting result.
  It says ranking quality is not the binding constraint, and that optimising the metric the
  field defaults to would have wasted effort here. **Report this one loudly if it happens.**
  It is the same shape as the finding that a strong logprob signal was badly calibrated: the
  measurement was fine, the thing being measured was the wrong thing.

Either way, quantify the relationship — correlate per-case `nDCG@10` against per-case
execution correctness across all 304. A near-zero correlation is the headline.

---

## 6. Deliverables

```
rag/relevance.py            labels, one definition each
rag/metrics_retrieval.py    recall@k, MRR, nDCG -- pure, unit-tested
rag/rerank.py               cross-encoder rerank + score cache
rag/eval_retrieval.py       driver: writes results/retrieval_eval.json
tests/test_metrics_retrieval.py   hand-worked assertions
docs/FINDING-reranking.md   the filled table and what it means
```

`results/retrieval_eval.json` carries every arm's metrics plus the config that produced it
(N, K, model name, whether query text was included) — so a number can always be traced to
the run that made it.

---

## 7. Honest caveats to carry into the write-up

1. **This is exemplar retrieval, not document retrieval.** Retrieving few-shot examples to
   condition a generator is a different task from retrieving passages that contain an answer.
   The machinery and the metrics transfer; the interpretation does not, entirely. Say so.
2. **Same-database is a proxy label.** See §2.
3. **n = 304.** Differences of a point or two in execution accuracy are inside the noise on
   this set — the 95% interval on a proportion at n=304 is roughly ±5.5 points. Use the
   paired structure: the arms answer the *same* 304 cases, so compare discordant cases
   directly rather than comparing two accuracy numbers.
4. **The reranker is off-the-shelf and not fine-tuned** on this domain. A fine-tuned reranker
   would likely do better; that is a follow-up, not a caveat that invalidates the result.
