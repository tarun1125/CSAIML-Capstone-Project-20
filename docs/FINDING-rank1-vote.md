# Finding — drop the majority vote, take the nearest neighbour

**Status:** complete, 2026-09-22.
**Follows:** [`docs/FINDING-reranking.md`](FINDING-reranking.md), which found that improving
exemplar *ranking* bought nothing. This tests the other retrieval lever.
**Artifacts:** `results/retrieval_eval.json`, `rag/data/qwen_rag_rank1_k10_execution_results.json`

---

## 0. The result

Replacing `build_prompts.py`'s majority vote over the top-10 neighbours' databases with
**the single nearest neighbour's database** — no vote at all — improves both retrieval and
the task:

| | database prediction | execution accuracy |
|---|---|---|
| majority vote, K=10 *(current)* | 258/304 = 84.87% | 143/304 = 47.04% |
| **pure rank-1, K=10** | **275/304 = 90.46%** | **149/304 = 49.01%** |

**Paired, over the same 304 cases:** 6 discordant, **6 gained, 0 lost**.
McNemar's exact **p = 0.03125**.

Rank-1 did not lose a single case. That is the whole result, and it is a one-line change
to a policy that has been silently costing accuracy since the vote was introduced.

### Why this matters next to the reranking null

The two experiments changed different things about the same retrieval step, with the same
model, decoding, schema cards and scorer:

| Intervention | cases whose prompt changed | discordant | gained | lost | p |
|---|---|---|---|---|---|
| Cross-encoder reranking (exemplar order) | 300/304 | 42 | 20 | 22 | 0.8776 |
| **Rank-1 vote (database choice)** | **27/304** | **6** | **6** | **0** | **0.0313** |

Reranking rewrote almost every prompt and produced churn. Rank-1 touched 27 prompts and
every discordant case went the right way. **Precision of intervention beat volume of
intervention** — and the metric the field would have told you to optimise (nDCG) pointed at
the intervention that did nothing.

---

## 1. Mechanism

`majority_vote_database()` takes an unweighted count over the top-K neighbours' databases.
An unweighted k-NN vote keeps **counts** and discards **similarity magnitude**. When two
databases are near-degenerate in embedding space, every pool example from either sibling
sits at roughly the same distance from the query, so the expected neighbour count for
sibling *c* is approximately `K · N_c / Σ N_sibling` — proportional to **pool frequency,
not relevance**. `argmax` over counts degenerates into `argmax` over pool frequency, and
the rarer sibling can never win however close it actually is. (Measured in detail in the
README's majority-vote section: `chinook_1`, 33 pool examples, loses **all 9** of its test
cases to `store_1`, 84.)

Rank-1 keeps magnitude by construction: it asks which single exemplar is nearest, which is
the question the index was built to answer.

### The gains are traceable, not luck

Of the 27 cases whose prompt changed:

| | count | execution gained | execution lost |
|---|---|---|---|
| database prediction **fixed** by rank-1 | 20 | **5** | 0 |
| database prediction **broken** by rank-1 | 3 | 0 | **0** |
| database prediction changed, still wrong | 4 | 0 | 0 |

Five of the six gains are cases where rank-1 put the **right schema** in the prompt and the
model then got the query right. The three cases rank-1 breaks cost nothing, because the
model was already wrong on all three.

**The other 20 db-fixes did not convert.** Fixing the schema block is necessary but not
sufficient — the model still has to write the query. That is consistent with the reranking
finding: retrieval is not the binding constraint, it is just no longer *a* constraint on
these 5 cases.

### The sixth gain is not real

The sixth discordant case, `spider-college_2-121`, has a **byte-identical prompt** in both
arms and a byte-identical generated query. It flipped because the query ends in

```
{'$group': ...}, {'$sort': {'count': -1}}, {'$limit': 1}
```

and two groups tie on `count`. MongoDB returns whichever it likes, so the same query
against the same data returned `{'Fall', 2002}` on one run and `{'Spring', 2008}` on the
next. One arm matched gold, the other did not.

**This is a property of the execution oracle, not of either arm.** Measured rate: **1 of
277** identical-prompt cases, ≈0.4%. Worth knowing when reading any paired result in this
repo — a discordance of 1–2 cases is within it. It does not threaten this result (5 of the
6 gains are mechanical, and there were 0 losses), but the honest statement of the effect is
**+5 attributable cases, +1 coin-flip**.

**Generation itself is fully deterministic**: 277/277 identical prompts produced
byte-identical generations. Greedy decoding on MLX reproduces exactly.

---

## 2. What was held fixed

Everything but the database-prediction policy. Same model
(`mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16`), greedy, max_tokens 300, FK annotations
on, same schema cards, same gold, same scorer
(`evaluation/execute_queries.run_model`, imported not reimplemented). **The same 10
exemplars, in the same order, appear in both arms' prompts** — rank-1 changes only which
database's schema block is pasted in. That is why the rank-1 row's ranking metrics are
identical to the baseline's by construction, and why any execution difference cannot be
attributed to ranking.

### A baseline correction this experiment forced

The obvious K=10 baseline, `rag/data/qwen_rag_execution_results_mlx.json`, is the **no-FK**
arm's per-case results — the FK variant's case-level JSON was overwritten by the
FK-vs-no-FK A/B run, as [`CANONICAL_ARTIFACTS.md`](../CANONICAL_ARTIFACTS.md) documents.
Every arm here has FK **on**, so using that file would have varied FK as well as the
policy. It was caught by an anomaly this experiment made visible: 9 cases flipped outcome
with an identical prompt, which is impossible for a deterministic generator.

The baseline is re-scored from `rag/data/qwen_rag_mlx_fk_normalized.json` into
`rag/data/qwen_rag_fk_k10_execution_results.json` — **143/304 (47.04%)**. This also
supplies the missing FK case-level file `CANONICAL_ARTIFACTS.md` says would close its
documented 1-case discrepancy.

---

## 3. Caveats

1. **p = 0.031 at 6 discordant pairs.** Significant, and thin. The exact test's floor at
   n=6 all-one-way is 0.031, so this is the *strongest* result obtainable at this
   discordance — but it rests on 6 cases, one of which (§1) is a coin flip. Treat it as
   "consistent, mechanically explained, and worth adopting", not "established to three
   decimal places."
2. **The database-prediction gain is much larger than the execution gain.** +17 cases of
   retrieval (258→275) yielded +6 of execution. Most fixed schemas did not convert, which
   is the reranking finding restated: the generator is the binding constraint.
3. **Adopting this invalidates downstream artifacts.** Switching the default changes the
   predicted database, and so the prompt, on 27 cases — every `qwen_rag_*.json`, the score
   CSVs and the cross-arm figures would need regenerating. Measured cost of a full
   regeneration with this harness: **~10 minutes** (6.5 min generation + 3 min Atlas
   scoring). That is a much weaker objection than it was before this harness existed.
4. **Not tested at other K.** Rank-1 is K-independent by definition, but the *baseline* it
   beats is not — a vote over 5 is a different (and, per the reranking experiment, more
   accurate at 88.5%) baseline than a vote over 10.
5. **This does not change `majority_vote_database` itself.** The policy is selected by a
   flag; the default is unchanged, and `rag_prompts.json` rebuilds byte-identically.

---

## 4. The weighted-vote alternative does not exist

The obvious middle path — keep the vote's noise-averaging but weight each neighbour by
similarity, so the pool-frequency degeneracy goes away without adopting rank-1 wholesale —
was tested and **there is nothing there**.

`rag/analyze_vote_weighting.py` replays every weighting on the recorded retrieval offline,
in about a second. The natural family is `score(db) = Σ exp(s_i / τ)` over that database's
neighbours, because it interpolates between the two policies already measured: as τ → ∞
every weight goes to 1 and it *is* the unweighted count vote; as τ → 0 the largest
similarity dominates every sum and it *is* pure rank-1.

| weighting | correct | accuracy | vs vote | vs rank-1 |
|---|---|---|---|---|
| unweighted vote *(current)* | 258/304 | 0.8487 | — | — |
| sum of similarity | 256/304 | 0.8421 | +1/−3, p=0.63 | +4/−23, p=0.0003 |
| softmax τ=1.0 | 256/304 | 0.8421 | +1/−3, p=0.63 | +4/−23, p=0.0003 |
| softmax τ=0.5 | 258/304 | 0.8487 | +2/−2, p=1.00 | +4/−21, p=0.0009 |
| softmax τ=0.2 | 266/304 | 0.8750 | +10/−2, p=0.039 | +4/−13, p=0.049 |
| sum of similarity⁴ | 265/304 | 0.8717 | +10/−3, p=0.092 | +3/−13, p=0.021 |
| sum of 1/rank | 271/304 | 0.8914 | +14/−1, p=0.0010 | +2/−6, p=0.29 |
| softmax τ=0.1 | 269/304 | 0.8849 | +12/−1, p=0.0034 | +3/−9, p=0.15 |
| softmax τ=0.05 | 275/304 | 0.9046 | +19/−2, p=0.0002 | +2/−2, p=1.00 |
| **softmax τ=0.02** | **276/304** | **0.9079** | +20/−2, p=0.0001 | +1/−0, p=1.00 |
| softmax τ=0.01 | 275/304 | 0.9046 | +20/−3, p=0.0005 | **+0/−0** |
| **pure rank-1** | 275/304 | 0.9046 | +20/−3, p=0.0005 | — |

**Accuracy is monotone in sharpness, and the optimum is the rank-1 endpoint.** Nothing in
the middle beats both ends. The best cell, τ=0.02 at 276/304, is **one case** above rank-1
(p = 1.0000) and differs from rank-1's predictions on **exactly 1 of 304 cases** — below
the ~1-in-277 oracle noise floor measured in §1. By τ=0.01 the two policies are
prediction-identical on all 304.

**No execution arm was run for this**, deliberately. The best weighted policy produces a
prompt different from the rank-1 arm's on a single case; generating and scoring it would
spend ~10 minutes to measure a one-case difference that the exact test already calls noise.
An experiment that cannot distinguish its arms is not worth running, and saying so is
cheaper than running it and reporting a number nobody should trust.

### What this rules out, and why it is a real finding

The hypothesis was that the vote contributes *something* — noise-averaging across several
neighbours — that rank-1 discards, and that a weighting could keep it. The sweep says it
does not. Once similarity magnitude is respected at all, the best thing to do with the
other nine neighbours is **ignore them**.

Two supporting details:

- **Plain similarity-sum is worse than counting** (256 vs 258). Predicted in advance and
  confirmed: cosine similarities here run 0.31–0.999 with mean 0.66, so summing them is
  barely distinguishable from counting, and 84 pool examples at ~0.60 still outweigh 33 at
  ~0.65. *Any* weighting that sums over members inherits the pool-frequency prior. Only a
  weighting sharp enough to be dominated by its maximum escapes it — and a sum dominated by
  its maximum is rank-1 with extra steps.
- **`sum of 1/rank` reaches 271/304**, beating the vote significantly (p = 0.0010) while
  still losing to rank-1 (p = 0.29). Rank-decay recovers most of the gap without using
  similarity at all, which is consistent: what the unweighted vote gets wrong is treating
  rank-10 as equal to rank-1.

So the choice really is binary — vote or rank-1 — and §0 already answered it.

---

## 5. Recommendation

**Switch the default to rank-1, and regenerate.** The evidence is a significant paired
improvement with zero regressions and a mechanism that explains it. The cost is ~10 minutes
of compute plus refreshing the downstream artifacts.

A weaker but cheaper alternative if the regeneration is unwelcome: keep the vote, but
**weight it by similarity** rather than counting. That preserves the vote's noise-averaging
where it helps while fixing the pool-frequency degeneracy — untested here, and it would
need its own arm.

---

## 6. Reproducing

```bash
python rag/analyze_vote_weighting.py                 # offline weighting sweep, ~1 s
python rag/build_prompts.py 10 --db-policy rank1     # -> rag_prompts_rank1_k10.json, 275/304
python rag/generate_rag_mlx.py \
  --prompts-path rag/data/rag_prompts_rank1_k10.json \
  --output rag/data/qwen_rag_mlx_rank1_k10_results.json
python rag/score_rerank_arms.py rank1_k10            # needs Atlas
python rag/eval_retrieval.py --with-execution
```

`--db-policy` is stripped from argv before `build_prompts.py`'s existing positional
parsing, so every current invocation stays byte-identical; verified by rebuilding
`rag_prompts.json` and comparing SHA-256.
