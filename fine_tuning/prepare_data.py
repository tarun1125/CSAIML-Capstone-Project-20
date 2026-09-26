# Fine-tuning data prep for the LoRA stage of the capstone (Mac / MLX-LM).
#
# Source of truth: rag/data/rag_fewshot_pool.json (244 cases) -- the SAME
# pool the RAG arm's FAISS index is built from, but here consumed as raw
# supervised (question -> gold PyMongo query) pairs instead of embeddings.
# rag/data/rag_test.json (61 cases) is NEVER read for training material --
# it is the held-out set the baseline (3/61) and RAG (13-15/61) arms were
# already scored against, and the fine-tuned arm has to be scored on the
# exact same 61 cases to produce a valid 4-arm comparison. This script only
# copies it through unchanged, as a labeled reference file, specifically so
# nothing downstream can accidentally fine-tune on it.
#
# Training-target style, confirmed with Tarun (see capstone-project-status.md,
# "MacBook migration + LoRA fine-tuning"): BASELINE-style, not RAG-style --
# fixed full-6-database-schema SYSTEM_PROMPT, no retrieved few-shot examples.
# This keeps a fine-tuned-zero-shot score directly comparable to the existing
# baseline 3/61 number, and keeps the fine-tuning arm complementary to (not
# entangled with) the RAG arm -- a future fine-tuned+RAG arm stays a clean
# addition later instead of this stage silently becoming a RAG variant.
#
# SYSTEM_PROMPT below is copied VERBATIM (not paraphrased) from
# mongodb_nl_to_sql_1.ipynb cell 5 -- the exact string the baseline Qwen
# generation run was conditioned on. Training on any reworded version of
# this prompt would reintroduce exactly the kind of prompt-template drift
# build_prompts.py's own comments warn about avoiding between arms.
#
# Output format: MLX-LM's chat JSONL convention -- each line
# {"messages": [{"role": "system", ...}, {"role": "user", ...},
#               {"role": "assistant", ...}]}
# read by `mlx_lm.lora --data fine_tuning/data ...` (train.jsonl + valid.jsonl
# auto-detected as chat format because every record has a "messages" key).
# The assistant turn is the case's normalized_query verbatim -- the same
# raw-PyMongo-expression-string target evaluation/execute_queries.py already
# knows how to eval() and score, so nothing downstream needs a new parser.

import json
import logging
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fine_tuning.prepare_data")

RANDOM_STATE = 42          # same seed convention as rag/build_split.py
VALID_FRACTION = 0.10      # 244 cases is small for LoRA -- keep as much as
                           # possible in train; 10% (~24 cases) is enough for
                           # a training-loss sanity signal, not itself a
                           # leaderboard metric (that's still the 61-case
                           # execution-accuracy eval against live Atlas).

SYSTEM_PROMPT = """You are a MongoDB query expert.
When given a natural-language question and a database schema, you output ONLY the raw PyMongo query — no explanation, no markdown, no prose.

Schema (6 databases, 27 collections -- the full-schema arm; compare later
against the retrieved-schema RAG arm):

concert_singer:
- stadium: { Stadium_ID (int), Location (str), Name (str), Capacity (int), Highest (int), Lowest (int), Average (int) }
- concert: { concert_ID (int), concert_Name (str), Theme (str), Stadium_ID (str), Year (str) }
- singer: { Singer_ID (int), Name (str), Country (str), Song_Name (str), Song_release_year (str), Age (int), Is_male (str) }
- singer_in_concert: { concert_ID (int), Singer_ID (str) }

pets_1:
- Student: { StuID (int), LName (str), Fname (str), Age (int), Sex (str), Major (int), Advisor (int), city_code (str) }
- Has_Pet: { StuID (int), PetID (int) }
- Pets: { PetID (int), PetType (str), pet_age (int), weight (float) }

network_1:
- Friend: { student_id (int), friend_id (int) }
- Highschooler: { ID (int), name (str), grade (int) }
- Likes: { student_id (int), liked_id (int) }

car_1:
- continents: { ContId (int), Continent (str) }
- car_makers: { Id (int), Maker (str), FullName (str), Country (str) }
- countries: { CountryId (int), CountryName (str), Continent (int) }
- model_list.json: { ModelId (int), Maker (int), Model (str) }    (collection name literally contains ".json" -- access as db['model_list.json'], never db.model_list.json)
- cars_data: { Id (int), MPG (str), Cylinders (int), Edispl (float), Horsepower (str), Weight (int), Accelerate (float), Year (int) }
- car_names: { MakeId (int), Model (str), Make (str) }

world_1:
- city: { ID (int), Name (str), CountryCode (str), District (str), Population (int) }
- country: { Code (str), Name (str), Continent (str), Region (str), SurfaceArea (float), IndepYear (int, nullable), Population (int), LifeExpectancy (float, nullable), GNP (float), GNPOld (float, nullable), LocalName (str), GovernmentForm (str), HeadOfState (str, nullable), Capital (int, nullable), Code2 (str) }
- countrylanguage: { CountryCode (str), Language (str), IsOfficial (str), Percentage (float) }

dog_kennels:
- Breeds: { breed_code (str), breed_name (str) }
- Charges: { charge_id (int), charge_type (str), charge_amount (int) }
- Sizes: { size_code (str), size_description (str) }
- Treatment_Types: { treatment_type_code (str), treatment_type_description (str) }
- Owners: { owner_id (int), first_name (str), last_name (str), street (str), city (str), state (str), zip_code (str), email_address (str), home_phone (str), cell_number (str) }
- Dogs: { dog_id (int), owner_id (int), abandoned_yn (str), breed_code (str), size_code (str), name (str), age (str), date_of_birth (str), gender (str), weight (str), date_arrived (str), date_adopted (str), date_departed (str) }
- Professionals: { professional_id (int), role_code (str), first_name (str), street (str), city (str), state (str), zip_code (str), last_name (str), email_address (str), home_phone (str), cell_number (str) }
- Treatments: { treatment_id (int), dog_id (int), professional_id (int), treatment_type_code (str), date_of_treatment (str), cost_of_treatment (int) }

Note: several logically-numeric fields are stored as strings (e.g.
concert.Stadium_ID, Dogs.age, Dogs.weight, cars_data.Horsepower, cars_data.MPG,
singer_in_concert.Singer_ID) -- use $toInt / $toDouble when comparing or
aggregating on these.

Rules:
1. Output ONLY the PyMongo expression (e.g. list(db.singer.find({...})))
2. Use db.<collection>.<method>() syntax -- use db['model_list.json'] for that one collection
3. Do NOT wrap in ```python or any markdown
4. Do NOT add any explanation before or after"""


def to_example(case: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]},
            {"role": "assistant", "content": case["normalized_query"]},
        ]
    }


def main():
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    pool_path = repo_root / "rag" / "data" / "rag_fewshot_pool.json"
    test_path = repo_root / "rag" / "data" / "rag_test.json"
    out_dir = root / "data"
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
    # A stronger check than id overlap alone -- catches the case where the
    # same question text somehow got assigned two different ids (would still
    # leak test-set signal into training even with disjoint id sets).
    pool_questions = {c["question"].strip() for c in pool}
    test_questions = {c["question"].strip() for c in test}
    question_overlap = pool_questions & test_questions
    assert not question_overlap, f"LEAK: identical question text in both halves: {question_overlap}"
    log.info("Leakage check #2 (question-text overlap): PASS -- 0 questions shared between pool and held-out test")

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

    # --- Write MLX-LM chat-format JSONL --------------------------------
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for case in train:
            f.write(json.dumps(to_example(case)) + "\n")
    log.info("Wrote %d training examples -> %s", len(train), train_path)

    with valid_path.open("w", encoding="utf-8") as f:
        for case in valid:
            f.write(json.dumps(to_example(case)) + "\n")
    log.info("Wrote %d validation examples -> %s", len(valid), valid_path)

    # --- Carry the held-out set through unchanged, clearly labeled ---------
    # Deliberately NOT reformatted into the chat/JSONL training shape, and
    # deliberately NOT named test.jsonl, so nothing about this file invites
    # `mlx_lm.lora --data fine_tuning/data --test` to touch it. Final
    # evaluation of the fine-tuned model must run through the SAME
    # evaluation/execute_queries.py + normalize.py pipeline the baseline and
    # RAG arms used, against these exact 61 ids, for the 4-arm comparison to
    # be valid.
    holdout_path = out_dir / "holdout_eval_cases.json"
    holdout_path.write_text(json.dumps(test, indent=2), encoding="utf-8")
    log.info("Copied %d held-out cases -> %s (reference only -- see file header note)", len(test), holdout_path)

    # --- Manifest -----------------------------------------------------------
    manifest = {
        "source_pool": str(pool_path.relative_to(repo_root)),
        "source_holdout": str(test_path.relative_to(repo_root)),
        "random_state": RANDOM_STATE,
        "valid_fraction": VALID_FRACTION,
        "training_style": "baseline (fixed full-6-database schema, no retrieved few-shot examples)",
        "counts": {
            "pool_total": len(pool),
            "train": len(train),
            "valid": len(valid),
            "holdout_eval_untouched": len(test),
        },
        "by_database": {
            "train": dict(Counter(c["database"] for c in train)),
            "valid": dict(Counter(c["database"] for c in valid)),
            "holdout_eval": dict(Counter(c["database"] for c in test)),
        },
        "by_complexity": {
            "train": dict(Counter(c.get("complexity") for c in train)),
            "valid": dict(Counter(c.get("complexity") for c in valid)),
        },
        "leakage_checks": {
            "pool_vs_holdout_id_overlap": sorted(id_overlap),
            "pool_vs_holdout_question_overlap": sorted(question_overlap),
            "train_vs_valid_id_overlap": sorted(train_ids & valid_ids),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote manifest -> %s", manifest_path)

    log.info("DONE. train=%d valid=%d holdout(untouched)=%d total_accounted=%d/%d",
              len(train), len(valid), len(test), len(train) + len(valid) + len(test), len(pool) + len(test))


if __name__ == "__main__":
    main()
