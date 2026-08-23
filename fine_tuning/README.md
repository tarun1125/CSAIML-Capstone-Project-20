# Fine-tuning stage — data prep

LoRA fine-tuning of `Qwen/Qwen2.5-Coder-1.5B-Instruct` via MLX-LM, run
locally on Tarun's M5 Pro MacBook (24GB unified memory). This folder holds
the data-prep step only — training itself is the next stage.

## Where the data came from

- **Source**: `rag/data/rag_fewshot_pool.json` (244 cases) — the same pool
  the RAG arm's FAISS index is built from, consumed here as plain
  supervised (question → gold PyMongo query) pairs instead of embeddings.
- **Never touched for training**: `rag/data/rag_test.json` (the 61-case
  held-out set baseline and RAG were already scored against). It's copied
  through unchanged to `data/holdout_eval_cases.json`, deliberately *not*
  named `test.jsonl` and *not* in chat format, so nothing about the file
  invites `mlx_lm.lora --test` to read it as training/eval data. The
  fine-tuned arm has to be scored on these exact 61 ids, through the same
  `evaluation/execute_queries.py` + `normalize.py` pipeline the other two
  arms used, for the 4-arm comparison (baseline 3/61, RAG 13–15/61,
  fine-tuned TBD) to mean anything.

## Training-target style: baseline-style, not RAG-style

Per the decision already recorded in the project status doc: the
`SYSTEM_PROMPT` used for every training example is the fixed, full
6-database/27-collection schema — copied **verbatim** from
`mongodb_nl_to_sql_1.ipynb` cell 5 (the exact prompt the baseline Qwen
generation run was conditioned on), not paraphrased and not the RAG arm's
retrieved-schema-plus-few-shot-examples version. This keeps a
fine-tuned-zero-shot score directly comparable to the existing baseline
number, and keeps this arm complementary to (not entangled with) RAG — a
future fine-tuned+RAG arm stays a clean addition later.

## What `prepare_data.py` does

1. Loads the 244-case pool and the 61-case held-out set.
2. **Two leakage checks**, both hard `assert`s, not just logged warnings:
   id-set overlap, and exact question-text overlap (a second, independent
   check — id overlap alone wouldn't catch the same question text
   somehow carrying two different ids). Both passed clean on this run: 0
   overlap either way. `rag/build_split.py` already asserted this at
   split-creation time; re-checking here is cheap insurance against the
   pool/test files having drifted apart since.
3. Splits the pool into train/valid — 90/10, stratified by `database`,
   `random_state=42` (same seeding convention as `rag/build_split.py`).
   90/10 rather than 80/20: 244 examples is already a small corpus for
   LoRA-tuning a 1.5B model, and the valid split here only needs to give a
   training-loss sanity signal, not serve as the project's real
   leaderboard metric — that's still the 61-case execution-accuracy eval
   against live Atlas. Spending more of the scarce 244 on `valid` than
   that would buy nothing.
4. Writes `data/train.jsonl` (219 examples) and `data/valid.jsonl` (25
   examples) in MLX-LM's chat-JSONL format:
   `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
   — the assistant turn is the case's `normalized_query` verbatim (the same
   raw PyMongo-expression string `evaluation/execute_queries.py` already
   evals and scores, so no new output parser is needed downstream).
5. Writes `data/split_manifest.json` — counts by database/complexity for
   train/valid/holdout, plus the two leakage-check results, so the split
   is auditable without re-running the script.

## Right-sizing check (under-engineered / over-engineered / fit)

- **The right fit**: a stratified train/valid split with an explicit,
  asserted leakage check, seeded for reproducibility — matches the
  rigor already established by `rag/build_split.py` for this same
  project, no more and no less.
- **Would be over-engineered right now**: k-fold cross-validation, a
  separate held-out-per-database calibration set, or synthetic data
  augmentation. At 244 real examples the variance those would control for
  is smaller than the variance already coming from fp16/session
  nondeterminism documented elsewhere in this project — not worth the
  complexity yet.
- **Worth watching, not yet acted on**: at ~1000 tokens/example average
  (940–1270 range) driven mostly by the ~3.7k-character fixed schema
  block repeated in every single example, this is a legitimate LoRA
  data-efficiency concern — most of every example's tokens are the same
  boilerplate schema, not signal. Not fixed here because doing so (e.g.
  truncating the prompt, or switching to a shorter/DB-specific schema)
  would break the "baseline-style, zero-shot-comparable" requirement above.
  Flagging it now so it isn't mistaken for an oversight if the LoRA run
  turns out sample-inefficient.

## Next stage (training — smoke-tested, full run in progress/pending)

Per the project's guide (already delivered): MLX-LM LoRA, rank=16 /
alpha=32 (→ `scale=2.0`, see below), num-layers=16, batch=4, ~200
iterations (≈4 epochs over the 219-example train set), against the
pre-converted `mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16`.

Hyperparameters live in `fine_tuning/lora_config.yaml`, not a CLI flag —
2026-08-21: the installed `mlx-lm` (0.31.3) has no `--lora-parameters` CLI
flag at all; LoRA hyperparameters only go in a YAML config's
`lora_parameters` block, and that block takes `scale` directly rather than
`alpha`. Converted via the standard convention `scale = alpha / rank =
32 / 16 = 2.0` (confirmed against ml-explore/mlx-lm's own docs/example
config, not guessed).

Real run:
```
mlx_lm.lora --config fine_tuning/lora_config.yaml
```

Smoke test (cheap sanity check before the full run — same config, CLI
flags override the matching YAML keys):
```
mlx_lm.lora --config fine_tuning/lora_config.yaml \
  --iters 20 --adapter-path fine_tuning/adapters_smoketest \
  --steps-per-report 5 --steps-per-eval 10
```

`max_seq_length: 2048` in the config is a safety margin over the observed
max example length (~1270 tokens) — not a tight fit, since MLX-LM
truncates silently past the configured length rather than erroring, and a
silently truncated training example is worse than a slightly wasteful
context budget.

After training: fuse the adapter, convert to GGUF via `llama.cpp`
(`mlx_lm.fuse --export-gguf` does not support Qwen — confirmed
Mistral/Mixtral/Llama only), import into Ollama via a Modelfile, then
evaluate the fine-tuned model on `data/holdout_eval_cases.json`'s 61 cases
through the existing execution-accuracy pipeline to complete the 4-arm
comparison.

## Files in this folder

```
fine_tuning/
  prepare_data.py            this script
  README.md                  this file
  data/
    train.jsonl               219 examples, MLX-LM chat format
    valid.jsonl               25 examples, MLX-LM chat format
    holdout_eval_cases.json   61 cases, verbatim copy of rag/data/rag_test.json
                              — reference only, NOT for training/MLX validation
    split_manifest.json       counts + leakage-check results, for audit
```

Re-running `python fine_tuning/prepare_data.py` from the repo root regenerates
all four `data/` files deterministically (fixed `random_state=42`).
