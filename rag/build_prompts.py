# The core retrieval step. For each of the 21 held-out test questions:
#   1. Embed the question and search the 100-example few-shot FAISS index
#      for the top-3 nearest neighbors (by cosine similarity).
#   2. Infer the TARGET DATABASE by majority vote across those 3
#      neighbors' `database` field (ties broken by the single nearest
#      neighbor, since FAISS already ranks by similarity -- rank-1 is the
#      most-trusted signal). This reuses the exact same embedding call
#      already being made for few-shot retrieval instead of standing up a
#      second, separate schema-retrieval index for a 6-way decision.
#   3. Pull the FULL schema for that one predicted database (every
#      collection in it, not a filtered subset) via schema_cards.py. Going
#      database-level rather than collection-level avoids the schema-linking
#      failure mode discussed up front: a fine-grained top-k *collection*
#      retrieval can leave out a collection a $lookup join needs, even when
#      the database prediction itself is correct.
#   4. Assemble a system prompt: the SAME header/rules wording as the
#      baseline arm (mongodb_nl_to_sql_1.ipynb cell 5), so the only real
#      difference between baseline and RAG prompts is retrieved content --
#      few-shot examples + one database's schema -- not prompt template
#      drift. Output format is unchanged too: a raw PyMongo expression
#      string, so the existing evaluation/execute_queries.py scoring path
#      (safe_eval_query / to_json_safe / results_match) works on RAG's
#      output with zero changes.
#
# Runs entirely locally (CPU, no Atlas connection, no GPU) -- only the
# actual Qwen generation call happens on Colab, reading this script's
# output (data/rag_prompts.json) directly.

import json
import logging
from collections import Counter
from pathlib import Path

import faiss

from embed_utils import embed
from schema_cards import build_cards

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_prompts")

TOP_K = 3

PROMPT_HEADER = (
    "You are a MongoDB query expert.\n"
    "When given a natural-language question, some example question/query "
    "pairs, and a database schema, you output ONLY the raw PyMongo query "
    "— no explanation, no markdown, no prose.\n\n"
)
PROMPT_RULES = (
    "\nRules:\n"
    "1. Output ONLY the PyMongo expression (e.g. list(db.singer.find({...})))\n"
    "2. Use db.<collection>.<method>() syntax -- use db['model_list.json'] "
    "for that one collection if it appears in the schema below\n"
    "3. Do NOT wrap in ```python or any markdown\n"
    "4. Do NOT add any explanation before or after\n"
    "5. Follow the style of the example query/answer pairs above -- same "
    "operators, same level of nesting -- when the current question is "
    "similar in shape to one of them"
)


def majority_vote_database(neighbor_dbs: list[str]) -> str:
    """neighbor_dbs is rank-ordered, nearest first (FAISS search order).
    Majority wins; a tie is broken by the nearest neighbor (index 0) since
    it's the single most-trusted signal FAISS gave us."""
    counts = Counter(neighbor_dbs)
    top_count = max(counts.values())
    tied = [db for db, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return tied[0]
    return neighbor_dbs[0]  # tie-break: rank-1 neighbor


def render_schema_block(db_name: str, cards_by_db: dict) -> str:
    cards = cards_by_db.get(db_name, [])
    lines = [f"Schema ({db_name}, {len(cards)} collection(s) -- retrieved database, full schema):\n"]
    for c in cards:
        fields = ", ".join(f"{k} ({v})" for k, v in c["fields"].items())
        lines.append(f"- {c['collection']}: {{ {fields} }}")
    return "\n".join(lines)


def render_examples_block(neighbors: list[dict]) -> str:
    lines = [f"Example question/query pairs (top-{len(neighbors)} most similar past questions):\n"]
    for i, n in enumerate(neighbors, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Question: {n['question']}")
        lines.append(f"Query: {n['normalized_query']}")
        lines.append("")
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    rag_dir = root / "rag"

    test_cases = json.loads((data_dir / "rag_test.json").read_text(encoding="utf-8"))
    log.info("Loaded %d held-out test cases", len(test_cases))

    index = faiss.read_index(str(rag_dir / "fewshot.index"))
    metadata = json.loads((rag_dir / "fewshot_metadata.json").read_text(encoding="utf-8"))
    log.info("Loaded FAISS index (ntotal=%d) and %d metadata rows", index.ntotal, len(metadata))

    cards = build_cards()
    cards_by_db: dict[str, list] = {}
    for c in cards:
        cards_by_db.setdefault(c["database"], []).append(c)
    log.info("Loaded schema cards for %d database(s)", len(cards_by_db))

    prompts = []
    db_match_count = 0

    for case in test_cases:
        case_id = case["id"]
        question = case["question"]
        gold_database = case["database"]

        qvec = embed([question])
        _scores, idxs = index.search(qvec, TOP_K)
        neighbors = [metadata[i] for i in idxs[0]]
        neighbor_dbs = [n["database"] for n in neighbors]

        predicted_database = majority_vote_database(neighbor_dbs)
        database_match = predicted_database == gold_database
        db_match_count += database_match

        system_prompt = (
            PROMPT_HEADER
            + render_examples_block(neighbors)
            + "\n"
            + render_schema_block(predicted_database, cards_by_db)
            + "\n"
            + PROMPT_RULES
        )

        prompts.append({
            "id": case_id,
            "question": question,
            "gold_database": gold_database,
            "predicted_database": predicted_database,
            "database_match": database_match,
            "retrieved_ids": [n["id"] for n in neighbors],
            "retrieved_neighbor_databases": neighbor_dbs,
            "complexity": case.get("complexity"),
            "system_prompt": system_prompt,
        })

        log.info("[%s] gold_db=%s predicted_db=%s match=%s neighbors=%s",
                  case_id, gold_database, predicted_database, database_match,
                  [n["id"] for n in neighbors])

    out_path = data_dir / "rag_prompts.json"
    out_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    accuracy = db_match_count / len(test_cases)
    log.info("SUMMARY  database retrieval accuracy: %d/%d = %.1f%%",
              db_match_count, len(test_cases), accuracy * 100)
    log.info("This is a diagnostic on retrieval quality, NOT the final metric -- "
              "it just tells you how often step 2 picked the right database "
              "before Qwen even runs. Saved -> %s", out_path)


if __name__ == "__main__":
    main()
