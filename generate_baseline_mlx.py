# Baseline generation for the 304-case test slice, via MLX-LM -- the
# apples-to-apples counterpart to fine_tuning/generate_predictions.py and
# rag/generate_rag_mlx.py.
#
# WHY THIS SCRIPT EXISTS: baseline and RAG were originally generated on
# Colab (HF transformers, fp16, GPU); fine-tuning runs on MLX-LM (Apple
# Metal) because that's the only place LoRA training could happen. That
# split confounds "did the technique help" with "did the serving stack
# change" -- Tarun's call (see project status doc, "Unifying Qwen arms onto
# MLX-LM for the 304-case comparison") was to stop mixing stacks and run
# ALL THREE arms through the same MLX-LM machinery. This script is the
# baseline half of that; rag/generate_rag_mlx.py is the RAG half. Neither
# touches the historical 305-case Stage-1 baseline number (Qwen 4.3%/305) --
# that stays as previously reported. This is specifically the "Baseline
# (test-slice)" number rag/score_rag.py compares RAG/fine-tuned against.
#
# WHY THE SCHEMA BLOCK IS BUILT THE WAY IT IS BELOW (read this before
# touching SYSTEM_PROMPT_HEADER/RULES or the schema-assembly code):
# The original 6-database baseline prompt (fine_tuning/prepare_data.py) is
# a hand-authored constant -- it was never generated from schema_cards.py.
# The 304-case test slice spans 23 databases. rag/schema_cards.py's
# COLLECTIONS used to be a hand-maintained dict that only ever listed the
# original 6 and drifted out of sync twice as the dataset expanded -- it's
# now auto-discovered directly from whatever's dumped locally under
# database/mongodb/ (see that file's own module comment), so this script's
# PREFERRED source is schema_cards.build_cards(): real field types, real FK
# annotations, and it covers exactly as many databases as have a local
# dump -- which grows automatically every time dump_atlas_to_local.py
# (repo root) is re-run, no code change needed here ever again.
#
# FALLBACK, for any database build_cards() doesn't cover yet (no local dump
# under database/mongodb/<db>/ -- e.g. before dump_atlas_to_local.py has
# been run for that database): pull real, already-verified schema straight
# out of rag/data/rag_prompts.json instead of leaving the gap silent. That
# file's RAG prompts happen to carry real schema for most databases
# already, harvested from whichever retrieval predicted them at least
# once. This project has been consistently disciplined about matching real
# data and never fabricating schema, so a database missing from BOTH
# sources is logged loudly (see `log_schema_coverage` below) rather than
# guessed at.
#
# THE KNOWN GAP, as of this script's last run: college_3 and chinook_1 are
# in neither source (0 occurrences of either as a `predicted_database`
# value anywhere in rag_prompts.json, and no local dump existed yet under
# database/mongodb/). Run `python dump_atlas_to_local.py --databases
# college_3 chinook_1` (needs a real Atlas connection) to close this for
# good -- once those two have local dumps, this script picks them up
# automatically via the build_cards() path, no further edit here required.
#
# Side finding surfaced while building this (recorded in the project status
# doc): chinook_1 was never predicted even once by retrieval, and its
# cases' actual nearest-neighbor votes land overwhelmingly on store_1
# instead. Round-2 notes call store_1 "confirmed identical underlying data
# to chinook_1", and round-3 then added chinook_1 back in as a "new"
# database -- these two are very likely still a near-duplicate pair, which
# would make retrieval confusion between them a structural certainty, not a
# retrieval-quality bug. Worth a closer look before round 4.
#
# Usage:
#   python generate_baseline_mlx.py
#   python generate_baseline_mlx.py --test-path rag/data/rag_test.json \
#       --output data/qwen_baseline_mlx_testslice_results.json
#
# Resumable: if --output already exists, ids already present are skipped
# and generation picks up where it left off. Results are written to disk
# after EVERY case (not just at the end) -- this is a long, local,
# interruptible run (304 cases, no Colab time limit but no crash-safety
# net either), so losing an hour of generation to one dropped connection
# would be a real, avoidable cost.
#
# Output feeds the same two steps every arm already uses:
#   python normalize.py data/qwen_baseline_mlx_testslice_results.json data/qwen_baseline_mlx_testslice_normalized.json
#   python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "fine_tuning"))
from spot_check import STOP_MARKERS, clean  # noqa: E402  (reuse, don't re-derive)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_baseline_mlx")

MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"  # same model, no adapter -- base-model arm

PROMPT_HEADER = (
    "You are a MongoDB query expert.\n"
    "When given a natural-language question and a database schema, you "
    "output ONLY the raw PyMongo query — no explanation, no markdown, no prose.\n\n"
)
PROMPT_RULES = (
    "\nRules:\n"
    "1. Output ONLY the PyMongo expression (e.g. list(db.singer.find({...})))\n"
    "2. Use db.<collection>.<method>() syntax -- use db['model_list.json'] "
    "for that one collection if it appears in the schema below\n"
    "3. Do NOT wrap in ```python or any markdown\n"
    "4. Do NOT add any explanation before or after\n"
    "5. ALWAYS exclude the MongoDB-assigned _id field from your output -- "
    "include \"_id\": 0 in the projection argument of every find() call and "
    "in every $project stage, unless the question explicitly asks for the "
    "_id value itself.\n"
    "6. $size inside a $match/find() FILTER only accepts a literal integer "
    "for an exact array-length match, e.g. {\"field\": {\"$size\": 3}}. It "
    "does NOT accept a comparison operator -- {\"field\": {\"$size\": "
    "{\"$gte\": 4}}} is INVALID and will error at query time. For "
    "'at least/more than/fewer than N items', either (a) use $expr with the "
    "aggregation $size operator: {\"$expr\": {\"$gte\": [{\"$size\": "
    "\"$field\"}, 4]}}, or (b) compute a count via $group/$addFields first, "
    "then $match on that count field."
)

# Verbatim from the original 6-database baseline prompt
# (fine_tuning/prepare_data.py). NOT extended to the 17 new databases --
# those string-typed-numeric-field mismatches were never independently
# confirmed the way these 6 were (via schema_cards.py's field-type
# inference against real dumps), so guessing at new ones for databases
# this script has no direct data access to would be exactly the kind of
# fabrication this project has consistently avoided. If new mismatches
# turn up in the round-3 databases' misses, add them here explicitly,
# with the same evidence standard the original 3 got.
KNOWN_STRING_TYPED_NUMERIC_NOTE = (
    "\nNote: several logically-numeric fields in the ORIGINAL 6 databases "
    "are stored as strings (concert.Stadium_ID, Dogs.age, Dogs.weight, "
    "cars_data.Horsepower, cars_data.MPG, singer_in_concert.Singer_ID) -- "
    "use $toInt / $toDouble when comparing or aggregating on these. This "
    "note has NOT been independently verified for the other databases "
    "below -- check field types in the schema itself if a comparison looks "
    "wrong.\n"
    "IMPORTANT: cars_data.Horsepower and cars_data.MPG also contain the "
    "LITERAL STRING \"null\" for some rows (not real nulls, not missing -- "
    "an actual \"null\" string value). Calling $toDouble/$toInt on that "
    "value throws a MongoDB server error and fails the whole query. ALWAYS "
    "exclude it first, e.g. add {\"MPG\": {\"$ne\": \"null\"}} to your "
    "$match stage BEFORE any $toDouble/$toInt/$avg/$max/$min on that field.\n"
)


def build_schema_block(rag_prompts_path: Path, test_databases: set) -> tuple[str, dict]:
    """Preferred source: schema_cards.build_cards() -- real field types, real
    FK annotations, covers every database with a local dump under
    database/mongodb/. Fallback, for any test-slice database build_cards()
    doesn't cover yet: extract real, already-verified schema straight out
    of rag/data/rag_prompts.json (see module docstring for the full story).
    Returns the assembled multi-database schema text plus a coverage
    report dict distinguishing which source covered which database."""
    sys.path.insert(0, str(REPO_ROOT / "rag"))
    from schema_cards import build_cards  # noqa: E402  local import: needs REPO_ROOT on sys.path first

    blocks_by_db: dict[str, str] = {}
    source_by_db: dict[str, str] = {}

    cards_by_db: dict[str, list] = {}
    for c in build_cards():
        cards_by_db.setdefault(c["database"], []).append(c)
    for db, db_cards in cards_by_db.items():
        lines = [f"{db}:"]
        for c in db_cards:
            fields = ", ".join(f"{k} ({v})" for k, v in c["fields"].items())
            lines.append(f"- {c['collection']}: {{ {fields} }}")
            for fk in c.get("fk_edges", []):
                lines.append(f"    FK: {fk}")
        blocks_by_db[db] = "\n".join(lines)
        source_by_db[db] = "build_cards"

    still_missing = test_databases - set(blocks_by_db)
    if still_missing:
        prompts = json.loads(rag_prompts_path.read_text(encoding="utf-8"))
        for p in prompts:
            db = p["predicted_database"]
            if db not in still_missing or db in blocks_by_db:
                continue
            sp = p["system_prompt"]
            start = sp.find("Schema (")
            end = sp.find("\n\nRules:")
            if start == -1:
                continue
            raw_block = sp[start:end if end != -1 else len(sp)]
            lines = raw_block.splitlines()
            coll_lines = [ln for ln in lines[1:] if ln.strip()]
            blocks_by_db[db] = f"{db}:\n" + "\n".join(coll_lines)
            source_by_db[db] = "rag_prompts_fallback"

    covered = sorted(set(blocks_by_db) & test_databases)
    missing = sorted(test_databases - set(blocks_by_db))
    schema_text = "\n\n".join(blocks_by_db[db] for db in sorted(blocks_by_db) if db in test_databases)

    return schema_text, {
        "covered": covered,
        "missing": missing,
        "from_build_cards": sorted(db for db in covered if source_by_db.get(db) == "build_cards"),
        "from_rag_prompts_fallback": sorted(db for db in covered if source_by_db.get(db) == "rag_prompts_fallback"),
    }


def build_full_schema_system_prompt(rag_prompts_path: Path, databases: set, label: str) -> tuple[str, dict, int]:
    """Assembles the complete fixed-schema SYSTEM_PROMPT for an arbitrary set
    of databases -- PROMPT_HEADER + a 'Schema (N databases, M collections --
    <label>):' header + build_schema_block()'s real schema text + the
    string-typed-numeric note + PROMPT_RULES. Pulled out of main() as a
    standalone, reusable function (2026-08-27, extending fine-tuning to
    train on the full 23-database set) so any other script needing a fixed
    full-schema prompt for a given database set -- e.g.
    fine_tuning/prepare_data_23db.py's 23-db training prompt and
    fine_tuning/generate_predictions_23db.py's matching generation prompt --
    calls this ONE function instead of hand-copying the assembly. Guarantees
    byte-identical prompts wherever the same (databases, label) pair is
    used -- same 'reuse, don't reimplement' discipline as
    build_generalization_prompt() in fine_tuning/generate_predictions_304.py.
    Returns (system_prompt, coverage, n_collections)."""
    schema_text, coverage = build_schema_block(rag_prompts_path, databases)
    n_colls = schema_text.count("\n- ")
    system_prompt = (
        PROMPT_HEADER
        + f"Schema ({len(coverage['covered'])} databases, {n_colls} collection(s) -- {label}):\n\n"
        + schema_text
        + "\n"
        + KNOWN_STRING_TYPED_NUMERIC_NOTE
        + PROMPT_RULES
    )
    return system_prompt, coverage, n_colls


def log_schema_coverage(coverage: dict, test_cases: list):
    missing = set(coverage["missing"])
    affected = [c for c in test_cases if c["database"] in missing]
    log.info(
        "Schema coverage: %d/%d test-slice databases have real schema "
        "(%d via schema_cards.build_cards() local dumps, %d via the "
        "rag_prompts.json fallback).",
        len(coverage["covered"]), len(coverage["covered"]) + len(missing),
        len(coverage.get("from_build_cards", [])), len(coverage.get("from_rag_prompts_fallback", [])),
    )
    if missing:
        log.warning(
            "NO real schema available for %s -- %d of %d test cases (%.1f%%) "
            "will run against a prompt that omits their own database's "
            "schema entirely. This is a known, counted gap (see this "
            "script's module docstring), not a silent one. Affected ids: %s",
            sorted(missing), len(affected), len(test_cases),
            len(affected) / len(test_cases) * 100,
            [c["id"] for c in affected],
        )


def load_checkpoint(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in existing}
    except (json.JSONDecodeError, KeyError):
        log.warning("Could not read existing %s as a valid checkpoint -- starting fresh.", output_path)
        return {}


def save_checkpoint(output_path: Path, results_by_id: dict, case_order: list):
    ordered = [results_by_id[c["id"]] for c in case_order if c["id"] in results_by_id]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", default=str(REPO_ROOT / "rag" / "data" / "rag_test.json"))
    parser.add_argument("--rag-prompts-path", default=str(REPO_ROOT / "rag" / "data" / "rag_prompts.json"),
                         help="Source of the real per-database schema blocks (see module docstring)")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "qwen_baseline_mlx_testslice_results.json"))
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    test_path = Path(args.test_path)
    output_path = Path(args.output)
    cases = json.loads(test_path.read_text(encoding="utf-8"))
    log.info("Loaded %d test-slice cases from %s", len(cases), test_path)

    test_databases = {c["database"] for c in cases}
    system_prompt, coverage, n_colls = build_full_schema_system_prompt(
        Path(args.rag_prompts_path), test_databases,
        label="the full-schema arm; compare later against the retrieved-schema RAG arm",
    )
    log_schema_coverage(coverage, cases)
    log.info("Assembled fixed baseline SYSTEM_PROMPT: %d characters, %d databases, %d collections",
              len(system_prompt), len(coverage["covered"]), n_colls)

    done = load_checkpoint(output_path)
    if done:
        log.info("Resuming: %d/%d cases already have a saved result in %s -- skipping those.",
                  len(done), len(cases), output_path)

    log.info("Stop markers in use for generation cleanup: %s (works around mlx-lm issue #973)", STOP_MARKERS)
    log.info("Loading %s (no adapter -- base-model baseline arm) ...", MODEL)
    model, tokenizer = load(MODEL)
    log.info("Model loaded. Beginning generation over %d cases (%d remaining).", len(cases), len(cases) - len(done))

    run_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        case_start = time.monotonic()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.monotonic() - case_start
        generated = clean(raw)

        log.info(
            "[%d/%d] id=%s db=%s (%.1fs) -> %s",
            i, len(cases), case["id"], case["database"], elapsed, generated[:80].replace("\n", " "),
        )

        done[case["id"]] = {
            "id": case["id"],
            "question": case["question"],
            "database": case["database"],
            "complexity": case.get("complexity"),
            "generated_query": generated,
        }
        save_checkpoint(output_path, done, cases)  # write progress after EVERY case

    total_elapsed = time.monotonic() - run_start
    n_generated_this_run = len(cases) - len(done) + len(done)  # kept for clarity; done now == len(cases)
    log.info("Generation complete: %d/%d cases in %s -> %.1fs total this run",
              len(done), len(cases), output_path, total_elapsed)
    log.info(
        "Next: python normalize.py %s data/qwen_baseline_mlx_testslice_normalized.json",
        output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
    )
    log.info("Then: python rag/score_rag.py 10 data/qwen_baseline_mlx_testslice_normalized.json")


if __name__ == "__main__":
    main()
