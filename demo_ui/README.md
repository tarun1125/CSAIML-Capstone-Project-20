# Live Demo UI

Streamlit app for the capstone defense: shows real Baseline vs RAG vs
Fine-Tuned (LoRA) NL-to-PyMongo results side by side, plus optional live
model inference for Q&A.

## Files

- `curated_examples.py` — 5 real, source-verified example cases (question,
  database, gold query/result, and each arm's actual generated query,
  execution status, and result). Every number in here was pulled directly
  from the repo's own execution-result JSON files, re-verified against the
  CURRENT epoch-parity-fixed (1000-iteration) fine-tuned adapter — see the
  module docstring for exact source file paths and why one originally
  shortlisted example (`spider-chinook_1-13`) was dropped after re-checking.
- `live_inference.py` — optional live model-calling wrapper. Reuses the
  repo's own prompt-building and generation code (rag/build_prompts.py,
  generate_baseline_mlx.py, fine_tuning/spot_check.py, evaluation/execute_queries.py)
  rather than reimplementing any of it, so a live answer is produced by the
  same code path that generated the reported benchmark numbers.
- `app.py` — the Streamlit app itself: Baseline / RAG / Fine-Tuned / Compare
  All 3 / Live Inference tabs.

## Running it

```bash
cd CSAIML-Capstone-Project-20
source .venv/bin/activate        # needed for Live Inference tab; Curated tabs work with plain streamlit too
streamlit run demo_ui/app.py
```

`streamlit` has been added to `requirements.txt`. If it's not yet installed
in your `.venv`:

```bash
source .venv/bin/activate
pip install streamlit
```

## Curated mode vs Live mode

**Curated (default, first 4 tabs)** replays the 5 pre-verified examples.
Zero model loading, zero network calls, cannot fail or hang — this is what
you should actually present live to the panel.

**Live Inference (5th tab, experimental)** actually loads Qwen2.5-Coder-1.5B
via MLX and generates a fresh query for whatever question you type in. It
needs `mlx_lm` importable (i.e. run from the project's `.venv`), and for the
RAG arm it additionally needs `faiss` + the sentence-transformers embedder
already used elsewhere in this repo. First generation per arm is slow
(cold model load, a few seconds to ~a minute depending on the machine) —
**rehearse this once before the defense**, don't try it live for the first
time in front of the panel. The 23-db fine-tuned adapter variants only
support the 23 databases they were actually trained on (the app tells you
which ones if you pick something else); if you want to demo the fine-tuned
arm on an out-of-scope database, use the 6-db adapter variant instead, or
stick to Curated mode.

An "execute against Atlas" checkbox is available in Live mode — off by
default. It needs `atlas-credentials.env` present and network access to
Atlas. It's optional because for these example questions the generated
query is usually judgeable by inspection; it's there in case a panel
question specifically asks "does it actually run."

## How curated_examples.py was built (for reproducibility)

The 5 examples were selected from the 304-case holdout to match five
specific baseline/RAG/fine-tuned pass-fail combinations (see each
example's `pattern` field), then every field was re-extracted directly
from:

- `rag/data/qwen_baseline_testslice_execution_results_mlx.json`
- `rag/data/qwen_rag_execution_results_mlx.json`
- `data/finetuned_full304_23db_1000iter_execution_results.json`
- `data/gold_results.json`

by joining on each case's `id` field and pulling `query` / `status` /
`result` / `execution_accuracy` verbatim — nothing was hand-typed. If the
fine-tuning adapter is retrained again, re-run this same join before
trusting the fine-tuned column of any existing curated example, since
(as already happened once here) a case's outcome can flip between
adapter versions.
