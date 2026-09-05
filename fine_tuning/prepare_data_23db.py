# Fine-tuning data prep for a SECOND LoRA adapter -- trained on all 23
# databases, not just the original 6. Deliberately a separate script from
# fine_tuning/prepare_data.py (untouched, still produces the original
# 6-database adapter at fine_tuning/adapters) -- this one writes to its own
# output directory (fine_tuning/data_23db/) and is meant to be paired with
# a new adapter path (fine_tuning/adapters_23db, see lora_config_23db.yaml)
# so the original adapter/data/results are never at risk of being
# overwritten.
#
# 2026-08-27 REVISION -- BATCHED SCHEMA, not one 23-db monolith per example:
# The first version of this script put the FULL schema for all 23 databases
# into every single training example (matching the original 6-db design
# literally). Measured against real data, that prompt is 22,955 characters
# (~5,700-6,500 tokens) -- ~6x the original, requiring max_seq_length=8192
# and a much heavier memory config (batch_size=1, num_layers=8) just to fit
# on the M5 Pro without truncating examples. Tarun asked whether there was
# a better option -- there is: partition the 23 databases into small,
# FIXED, deterministic batches (see BATCH_SIZE below), and give each
# training example only its OWN database's batch schema, not all 23. This
# still trains ONE adapter on real examples spanning all 23 databases --
# nothing is skipped -- it just never shows the model all 23 schemas
# stacked together in one context, which none of the actual questions in
# this dataset need anyway (every question is scoped to a single
# database). Verified by actually running this script's main() against real
# repo data (chunking the 23 sorted database names into groups of 4): 6
# batches, worst-case batch is 8,909 characters (~2,227-2,545 tokens) --
# back in the same size class as the ORIGINAL 6-db prompt (~3,700 chars),
# letting lora_config_23db.yaml stay close to the ORIGINAL, already-working
# memory config instead of the aggressive 8192/batch_size=1 compromise the
# monolith version needed.
#
# WHY BATCHING IS SAFE HERE (READ BEFORE CHANGING BATCH_SIZE): the whole
# point of showing a bounded schema instead of all 23 is that at BOTH
# training and generation time, this project always knows the ground-truth
# target database for every case (case["database"]) -- the baseline arm's
# own full-schema design already assumes the model can be given schema
# information without seeing the raw SQL/question first, and the RAG arm
# already narrows to a (retrieved, imperfect) subset for the same reason.
# Giving a small, bounded, always-correct batch is a strictly easier
# version of a problem this project's other two arms already solve less
# precisely. This still asks the model to identify which OF THE BATCH's
# few databases a question is about (same difficulty class as the original
# "which of 6" problem), it just never asks it to also ignore 17 more
# irrelevant schemas while doing so.
#
# MANIFEST-DRIVEN CONSISTENCY (do not weaken this): fine_tuning/
# generate_predictions_23db.py does NOT independently recompute the
# batching -- it reads the exact batch definitions AND the exact,
# already-assembled system-prompt text for each batch straight out of this
# script's own split_manifest.json output. Two independent scripts
# "computing the same deterministic batching" sounds safe until pool/test
# files or BATCH_SIZE drift between when one runs and when the other runs
# -- reading the literal recorded prompt text removes that risk entirely,
# not just makes it unlikely.
#
# Source of truth: rag/data/rag_fewshot_pool.json (1,213 cases, 23
# databases) -- the SAME pool the RAG arm's FAISS index and the original
# 6-db fine-tuning both draw from. rag/data/rag_test.json (304 cases) is
# NEVER read as training material -- same two leakage checks as the
# original prepare_data.py (id overlap, then question-text overlap).
#
# Output format: same MLX-LM chat JSONL convention as the original --
# {"messages": [{"role": "system", ...}, {"role": "user", ...},
#               {"role": "assistant", ...}]}
# read by `mlx_lm.lora --data fine_tuning/data_23db ...`.

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from generate_baseline_mlx import build_full_schema_system_prompt  # noqa: E402  reuse, don't reimplement/hand-transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.prepare_data_23db")

RANDOM_STATE = 42          # same seed convention as rag/build_split.py / prepare_data.py
VALID_FRACTION = 0.10      # same fraction as the original 6-db prep

# Databases per batch. Verified this session by actually running this
# script's main() against real repo data: batch_size=4 -> 6 batches, worst
# case 8,909 chars (~2,227-2,545 tokens), smallest 2,968 chars; this keeps
# every batch's prompt in roughly the same size class as the original 6-db
# prompt (~3,700 chars), so lora_config_23db.yaml can stay close to the
# original, already-proven-working memory config instead of needing the
# aggressive compromise a single 23-db monolith prompt would require.
# Larger batch sizes (6 -> 4 batches, 8 -> 3 batches) would push worst-case
# prompts larger still and were not re-verified against real data after
# this revision -- 4 was chosen as the more conservative option given this
# project's prior OOM history on the M5 Pro at even ~1,000-token examples.
BATCH_SIZE = 4

# Purely a sanity threshold for the log warning below -- NOT enforced.
RECOMMENDED_MIN_MAX_SEQ_LENGTH = 4096

# 2026-08-28 FIX -- collision-aware batching (was plain alphabetical chunking).
# Real bug found by scoring the first 23-db retrain against live data: naive
# sorted-then-chunked batching put chinook_1 + college_1 + college_2 +
# college_3 in the SAME batch (they're alphabetically adjacent), and those
# four databases share overlapping domain vocabulary under different naming
# conventions (college_1 uses UPPERCASE collections/fields, college_2 uses
# lowercase, college_3 uses TitleCase; chinook_1 and store_1 are confirmed
# byte-identical underlying data under two different labels -- see
# rag/data/rag_fewshot_pool.json's chinook_1/store_1 entries). Verified
# directly against the 304-case scored output: this one batch scored 8/79
# (10.1%) vs 55/225 (24.4%) for every other, non-colliding batch -- the
# model was borrowing a same-batch sibling's collection/field names, every
# single time, for every one of college_3's 7 test cases (0/7 correct,
# 0 exceptions). COLLISION_GROUPS below is the fix: a database is never
# batched with another member of a group it belongs to. store_1/sakila_1 is
# included on weaker evidence (one observed leak, `store_1` predicting
# sakila's `film_category`/`film` collections) -- cheap to keep separated
# regardless. Extend this list if a future retrain surfaces another pair.
COLLISION_GROUPS = [
    {"college_1", "college_2", "college_3"},   # same "college" domain, different naming conventions
    {"chinook_1", "store_1"},                   # confirmed byte-identical underlying data, different naming
    {"flight_2", "flight_4"},                   # same "flight" domain
    {"store_1", "sakila_1"},                    # weaker evidence (1 observed leak), cheap to separate
]


def _collision_group_ids(db: str) -> set:
    return {i for i, group in enumerate(COLLISION_GROUPS) if db in group}


def build_database_batches(databases: set, batch_size: int) -> list:
    """Partitions a set of databases into batches of up to batch_size each,
    deterministically (sorted, then greedily first-fit assigned) so
    re-running this against the same database set always produces the same
    batches -- AND so that no batch ever contains two databases from the
    same COLLISION_GROUPS entry (see the fix note above). The manifest this
    script writes is the actual consistency guarantee downstream scripts
    rely on (see module docstring) -- determinism here is a nice property,
    not the safety net.

    Greedy first-fit: walk databases in sorted order, place each one in the
    first existing batch that (a) has room and (b) doesn't already hold a
    same-collision-group member; open a new batch only if none qualifies.
    With 23 databases, batch_size=4, and 4 small collision groups (max size
    3), this always succeeds -- there's no scenario here where a database
    could get stranded with nowhere valid to go. If COLLISION_GROUPS grows
    enough that stops being true, this will silently just open extra
    batches (each database still lands somewhere) rather than crash --
    verify batch count/sizes in the log output after any such change."""
    sorted_dbs = sorted(databases)
    batches: list[list[str]] = []
    for db in sorted_dbs:
        my_groups = _collision_group_ids(db)
        placed = False
        for batch in batches:
            if len(batch) >= batch_size:
                continue
            if any(_collision_group_ids(existing) & my_groups for existing in batch):
                continue
            batch.append(db)
            placed = True
            break
        if not placed:
            batches.append([db])
    return batches


def to_example(case: dict, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case["question"]},
            {"role": "assistant", "content": case["normalized_query"]},
        ]
    }


def main():
    pool_path = REPO_ROOT / "rag" / "data" / "rag_fewshot_pool.json"
    test_path = REPO_ROOT / "rag" / "data" / "rag_test.json"
    rag_prompts_path = REPO_ROOT / "rag" / "data" / "rag_prompts.json"
    out_dir = ROOT / "data_23db"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    log.info("Loaded %d fewshot-pool cases from %s (training source)", len(pool), pool_path)
    log.info("Loaded %d held-out test cases from %s (reference only, NOT for training)", len(test), test_path)

    # --- Leakage check #1: id overlap between pool and held-out test ------
    pool_ids = {str(c["id"]) for c in pool}
    test_ids = {str(c["id"]) for c in test}
    id_overlap = pool_ids & test_ids
    assert not id_overlap, f"LEAK: ids appear in both pool and held-out test: {id_overlap}"
    assert len(pool_ids) == len(pool), "duplicate ids inside fewshot pool"
    assert len(test_ids) == len(test), "duplicate ids inside held-out test"
    log.info("Leakage check #1 (id overlap): PASS -- 0 ids shared between pool and held-out test")

    # --- Leakage check #2: exact question-text overlap ---------------------
    pool_questions = {c["question"].strip() for c in pool}
    test_questions = {c["question"].strip() for c in test}
    question_overlap = pool_questions & test_questions
    assert not question_overlap, f"LEAK: identical question text in both halves: {question_overlap}"
    log.info("Leakage check #2 (question-text overlap): PASS -- 0 questions shared between pool and held-out test")

    # --- Build batched schema prompts ---------------------------------------
    all_databases = {c["database"] for c in test} | {c["database"] for c in pool}
    batches = build_database_batches(all_databases, BATCH_SIZE)
    log.info("Partitioned %d databases into %d batches of up to %d each", len(all_databases), len(batches), BATCH_SIZE)

    batch_records = []
    db_to_batch_index = {}
    max_prompt_chars = 0
    for i, batch_dbs in enumerate(batches):
        label = f"batch {i + 1}/{len(batches)} of the fine-tuned arm's schema training prompt -- databases in this group: {', '.join(batch_dbs)}"
        prompt, coverage, n_colls = build_full_schema_system_prompt(rag_prompts_path, set(batch_dbs), label=label)
        if coverage["missing"]:
            log.warning("Batch %d: NO schema available for %s -- these databases' examples will use a "
                        "prompt that omits their own schema entirely.", i, coverage["missing"])
        approx_tokens_low = round(len(prompt) / 4)
        approx_tokens_high = round(len(prompt) / 3.5)
        log.info("Batch %d (%s): %d chars (~%d-%d tokens), %d collections",
                  i, batch_dbs, len(prompt), approx_tokens_low, approx_tokens_high, n_colls)
        max_prompt_chars = max(max_prompt_chars, len(prompt))
        batch_records.append({
            "index": i,
            "databases": batch_dbs,
            "system_prompt": prompt,
            "chars": len(prompt),
            "approx_tokens_range": [approx_tokens_low, approx_tokens_high],
            "collections": n_colls,
            "coverage": coverage,
        })
        for db in batch_dbs:
            db_to_batch_index[db] = i

    max_approx_tokens_high = round(max_prompt_chars / 3.5)
    log.info("Largest batch prompt: %d chars (~%d tokens worst case). If lora_config_23db.yaml's "
              "max_seq_length is below ~%d (this max + question/answer overhead), examples in that "
              "batch WILL be truncated -- verify before training, don't assume.",
              max_prompt_chars, max_approx_tokens_high, RECOMMENDED_MIN_MAX_SEQ_LENGTH)

    # --- Train / valid split of the pool only ------------------------------
    databases = [c["database"] for c in pool]
    train, valid = train_test_split(
        pool,
        test_size=VALID_FRACTION,
        stratify=databases,
        random_state=RANDOM_STATE,
    )
    log.info("Train: %d cases -- by database: %s", len(train), dict(Counter(c["database"] for c in train)))
    log.info("Valid: %d cases -- by database: %s", len(valid), dict(Counter(c["database"] for c in valid)))

    train_ids = {str(c["id"]) for c in train}
    valid_ids = {str(c["id"]) for c in valid}
    assert not (train_ids & valid_ids), "LEAK: overlap between train and valid splits"
    log.info("Train/valid split check: PASS -- 0 ids shared between train and valid")

    # --- Write MLX-LM chat-format JSONL, each example using ITS OWN batch's
    #     schema (looked up via db_to_batch_index), not the full 23-db one --
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    def prompt_for(case):
        return batch_records[db_to_batch_index[case["database"]]]["system_prompt"]

    with train_path.open("w", encoding="utf-8") as f:
        for case in train:
            f.write(json.dumps(to_example(case, prompt_for(case))) + "\n")
    log.info("Wrote %d training examples -> %s", len(train), train_path)

    with valid_path.open("w", encoding="utf-8") as f:
        for case in valid:
            f.write(json.dumps(to_example(case, prompt_for(case))) + "\n")
    log.info("Wrote %d validation examples -> %s", len(valid), valid_path)

    # --- Manifest -- generate_predictions_23db.py reads "batches" and
    #     "database_to_batch_index" straight from here, VERBATIM, rather
    #     than recomputing anything (see module docstring). ---------------
    manifest = {
        "source_pool": str(pool_path.relative_to(REPO_ROOT)),
        "source_holdout": str(test_path.relative_to(REPO_ROOT)),
        "random_state": RANDOM_STATE,
        "valid_fraction": VALID_FRACTION,
        "training_style": "baseline-style, BATCHED full schema (each example sees its own database's "
                           "batch of up to BATCH_SIZE databases, not all 23) -- see module docstring "
                           "for why this changed from the original all-23-at-once design",
        "batch_size": BATCH_SIZE,
        "n_batches": len(batches),
        "max_batch_prompt_chars": max_prompt_chars,
        "batches": batch_records,
        "database_to_batch_index": db_to_batch_index,
        "counts": {
            "pool_total": len(pool),
            "train": len(train),
            "valid": len(valid),
            "holdout_eval_reference_only": len(test),
        },
        "by_database": {
            "train": dict(Counter(c["database"] for c in train)),
            "valid": dict(Counter(c["database"] for c in valid)),
            "holdout_eval": dict(Counter(c["database"] for c in test)),
        },
        "leakage_checks": {
            "pool_vs_holdout_id_overlap": sorted(id_overlap),
            "pool_vs_holdout_question_overlap": sorted(question_overlap),
            "train_vs_valid_id_overlap": sorted(train_ids & valid_ids),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote manifest -> %s (includes full batch prompt text -- generation reads this, "
              "does not recompute)", manifest_path)

    log.info("DONE. train=%d valid=%d holdout(untouched, at %s)=%d, %d batches",
              len(train), len(valid), test_path.relative_to(REPO_ROOT), len(test), len(batches))
    log.info("Next: mlx_lm.lora --config fine_tuning/lora_config_23db.yaml")


if __name__ == "__main__":
    main()
