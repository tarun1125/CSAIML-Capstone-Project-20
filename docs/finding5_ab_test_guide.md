# Finding 5: FK-Annotated RAG -- A/B Test Guide (MLX-LM edition)

## What changed since the original version of this guide

The original protocol below ran both variants (with FK / without FK) inside
a **single Colab session**, because Colab's fp16 GPU inference had already
been shown (see the project's own "RAG accuracy has real session-to-session
variance" finding) to drift between separate sessions even with identical
code -- running both variants in one continuous session was the only way to
guarantee any accuracy difference was caused by the prompt change, not by
which session happened to run.

That whole problem doesn't apply here anymore. Every arm in this project now
runs on **MLX-LM, locally, on Apple Silicon** -- one deterministic process on
your own machine, not a shared cloud GPU across separate sessions. This is
actually a *better* setting for this A/B test than Colab ever was, not a
downgrade: same principle (only the prompt differs, nothing else), with the
extra confound (which session did you happen to get) removed entirely rather
than merely controlled for. So: no Colab, no "single session" requirement,
no notebook cell to hand-edit -- just two ordinary local commands.

## What the FK annotations are

- **`rag/schema_cards.py`** -- an `FK_EDGES` dictionary with 22 canonical
  join relationships across the original 6 databases, including 3
  type-mismatch warnings (`str->int, use $toInt`). Each schema card carries
  an `fk_edges` list.
- **`rag/build_prompts.py`** -- `render_schema_block()` renders these below
  each collection's fields, e.g.:
  ```
  - concert: { concert_ID (int), ... Stadium_ID (str) }
      FK: Stadium_ID -> stadium.Stadium_ID (str->int, use $toInt)
  ```
  It now also takes an `include_fk` toggle (2026-08-28) so the exact
  pre-Finding-5 prompt can be reproduced on demand for this A/B test,
  instead of the two variants also risking drifting apart in some unrelated
  way over time.

## Why this should help

The retrospective identified that 12 of 46 RAG misses (26%) involved
multi-collection joins. Without FK annotations, the model has to guess
which fields link collections and doesn't know about the 3 type-mismatch
pairs that need `$toInt` casts. The FK lines give the model explicit join
paths. **Scope note**: `FK_EDGES` only covers the original 6 databases --
extending it to the other 17 (round 2/3 additions) would need the same
evidence standard the original 6 got (real FK relationships confirmed
against the actual dumps, not guessed), which wasn't done this session. Run
this A/B test on the full 304-case set anyway -- databases without FK
entries just render identically in both variants (an implicit, free
no-op control on the other 17 databases), so their scores act as a
consistency check that nothing outside the 6 target databases moved.

## Step 1: Generate both prompt variants (Mac terminal)

```bash
cd ~/Documents/ai-projects/capstone-project/CSAIML-Capstone-Project-20

# With FK annotations (default -- also writes the stably-named rag_prompts_fk.json)
python rag/build_prompts.py 10

# Without FK annotations (reproduces the pre-Finding-5 prompt exactly)
python rag/build_prompts.py 10 nofk

# Verify: FK lines present in one, absent in the other
python -c "
import json
fk = json.loads(open('rag/data/rag_prompts_fk.json').read())
nofk = json.loads(open('rag/data/rag_prompts_nofk.json').read())
fk_count = sum(1 for p in fk if 'FK:' in p['system_prompt'])
nofk_count = sum(1 for p in nofk if 'FK:' in p['system_prompt'])
print(f'fk variant:   {fk_count}/{len(fk)} prompts contain FK annotations')
print(f'nofk variant: {nofk_count}/{len(nofk)} prompts contain FK annotations (should be 0)')
assert nofk_count == 0, 'nofk variant leaked an FK line -- do not proceed, investigate first'
"
```

## Step 2: Generate both variants via MLX-LM (Mac terminal, same machine, no Colab)

```bash
python rag/generate_rag_mlx.py \
    --prompts-path rag/data/rag_prompts_fk.json \
    --output rag/data/qwen_rag_mlx_fk_results.json

python rag/generate_rag_mlx.py \
    --prompts-path rag/data/rag_prompts_nofk.json \
    --output rag/data/qwen_rag_mlx_nofk_results.json
```

Both runs use the identical model load, identical machine, identical
decoding settings -- the only input that differs between them is the FK
lines in the prompt. Resumable like every other MLX-LM generation script in
this project (checkpointed after every case) if either run gets interrupted.

## Step 3: Normalize and score both (Mac terminal)

```bash
python normalize.py rag/data/qwen_rag_mlx_fk_results.json rag/data/qwen_rag_mlx_fk_normalized.json
python normalize.py rag/data/qwen_rag_mlx_nofk_results.json rag/data/qwen_rag_mlx_nofk_normalized.json

# Score the FK variant
python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_fk_normalized.json
cp rag/outputs/rag_vs_baseline_scores_mlx.csv rag/outputs/rag_vs_baseline_scores_mlx_fk.csv

# Score the no-FK variant
python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json rag/data/qwen_rag_mlx_nofk_normalized.json
cp rag/outputs/rag_vs_baseline_scores_mlx.csv rag/outputs/rag_vs_baseline_scores_mlx_nofk.csv

# Compare
echo "=== NO-FK ==="
cat rag/outputs/rag_vs_baseline_scores_mlx_nofk.csv
echo ""
echo "=== WITH FK ==="
cat rag/outputs/rag_vs_baseline_scores_mlx_fk.csv
```

(Double-check `rag/score_rag.py`'s exact positional-arg order against
`python rag/score_rag.py --help` before running -- this guide reflects the
signature as of this session; if it's since changed, trust the script.)

## What to expect

- **Baseline**: not run again here -- both variants share the same baseline
  file, since only the RAG prompt changed.
- **RAG no-FK** (304-case, unified MLX-LM stack): should land close to the
  already-measured 44.7% (136/304) full-set number -- this run uses the
  identical no-FK prompt content that number came from, just regenerated.
- **RAG with FK**: if FK annotations help, this should be higher than the
  no-FK variant on the 6 databases FK_EDGES actually covers. The 12
  join-dependent misses (originally identified on the smaller 61-case set)
  are the rough ceiling for improvement there.
- **Because both variants run on the same local MLX-LM process**, any
  difference is attributable to the prompt change, not session/stack
  variance -- the entire point of moving this test off Colab.

## Interpreting results

At n=304 (much larger than the original n=61 this test was designed
against), the worst-case standard error at p=0.5 is roughly 2.9 percentage
points (vs ~6.4pp at n=61) -- meaningfully more sensitive to a real, small
effect. So:
- A gain of 1-4 cases (~0.3-1.3pp) is directionally positive but likely
  still within noise.
- A gain of 9+ cases (~3pp+) is a meaningful signal at this sample size.
- Zero or negative means FK annotations didn't help (or confused the
  model) -- also worth knowing, and cheap to have checked for real rather
  than assumed.

Either way, the FK annotations are low-cost metadata that make the schema
more informative and improve prompt interpretability regardless of the
accuracy delta -- keeping them is reasonable even on a small or neutral
result.
