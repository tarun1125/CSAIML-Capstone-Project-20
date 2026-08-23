# The core retrieval step. For each held-out test question:
#   1. Embed the question and search the few-shot FAISS index for the
#      top-K nearest neighbors (by cosine similarity). K went 3 -> 5 after
#      the first RAG pass (the one retrieval miss, car-e5, was a 2-1 split
#      where 2 borderline neighbors outvoted 1 correct one -- a wider vote
#      is less exposed to a couple of noisy nearest hits) and 5 -> 10 once
#      the golden set was large enough to support it.
#   2. Infer the TARGET DATABASE by majority vote across those K
#      neighbors' `database` field (ties broken by the single nearest
#      neighbor, since FAISS already ranks by similarity -- rank-1 is the
#      most-trusted signal; an even K makes an exact tie more likely than
#      an odd one, which is exactly why that tie-break exists rather than
#      just calling max() and hoping). This reuses the exact same embedding
#      call already being made for few-shot retrieval instead of standing
#      up a second, separate schema-retrieval index for a 6-way decision.
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
# output (rag/data/rag_prompts.json) directly.
#
# TOP_K is overridable from the command line for a controlled K-sweep
# (K=3 vs K=5 vs K=10, same 61-case test split, same gold, same scoring
# code) -- this was previously a hardcoded constant that changed by hand
# across three separate dataset sizes (3 -> 5 -> 10), which confounded
# "more examples" with "bigger pool, bigger test set." Comparing K on a
# fixed split needed this to be a parameter, not a constant:
#   python rag/build_prompts.py         # TOP_K=10 (default, unchanged
#                                        # behavior -- still writes
#                                        # rag_prompts.json)
#   python rag/build_prompts.py 3       # writes rag_prompts_k3.json
#   python rag/build_prompts.py 5       # writes rag_prompts_k5.json
# K=10's output filename is deliberately left as the original
# rag_prompts.json (not rag_prompts_k10.json) so nothing already reading
# that default filename (score_rag.py's diagnostic, the RAG notebook)
# breaks without an explicit opt-in.

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import faiss

from embed_utils import embed
from schema_cards import build_cards

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_prompts")

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10

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
    "5. ALWAYS exclude the MongoDB-assigned _id field from your output -- "
    "include \"_id\": 0 in the projection argument of every find() call and "
    "in every $project stage, unless the question explicitly asks for the "
    "_id value itself. A result that includes an unwanted _id will not "
    "match the expected answer even if every other field is correct.\n"
    "6. Follow the style of the example query/answer pairs above -- same "
    "operators, same level of nesting -- when the current question is "
    "similar in shape to one of them"
)

# Restores a warning that existed in the baseline arm's system prompt
# (mongodb_nl_to_sql_1.ipynb cell 5) but was dropped when this RAG prompt
# was first built -- confirmed as the direct cause of 3 of the first RAG
# run's 16 misses (cs-h2, cs-c1, dk-e2): each needed a $toInt/$toDouble
# cast before comparing or joining on one of these fields, and without the
# warning the model just used the raw string value, which either matches
# nothing (a join) or compares wrong (a numeric filter). Scoped per
# database so a prompt only mentions fields that are actually in its own
# retrieved schema, not baseline's whole-repo list.
STRING_TYPED_NUMERIC_FIELDS = {
    "concert_singer": ["concert.Stadium_ID", "singer_in_concert.Singer_ID"],
    "dog_kennels": ["Dogs.age", "Dogs.weight"],
    "car_1": ["cars_data.Horsepower", "cars_data.MPG"],
}


def render_numeric_string_note(db_name: str) -> str:
    fields = STRING_TYPED_NUMERIC_FIELDS.get(db_name)
    if not fields:
        return ""
    return (
        "\nNote: some logically-numeric fields in this schema are stored as "
        f"strings ({', '.join(fields)}) -- use $toInt / $toDouble when "
        "comparing, filtering, or joining on these.\n"
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
        # Finding 5: append FK (join) annotations so the model knows which
        # fields link collections and whether a $toInt cast is needed.
        fk_edges = c.get("fk_edges", [])
        if fk_edges:
            for fk in fk_edges:
                lines.append(f"    FK: {fk}")
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
    rag_dir = root / "rag"
    rag_data_dir = rag_dir / "data"
    rag_data_dir.mkdir(parents=True, exist_ok=True)

    test_cases = json.loads((rag_data_dir / "rag_test.json").read_text(encoding="utf-8"))
    log.info("Loaded %d held-out test cases -- TOP_K=%d for this run", len(test_cases), TOP_K)

    index = faiss.read_index(str(rag_data_dir / "fewshot.index"))
    metadata = json.loads((rag_data_dir / "fewshot_metadata.json").read_text(encoding="utf-8"))
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
            + render_numeric_string_note(predicted_database)
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

    out_name = "rag_prompts.json" if TOP_K == 10 else f"rag_prompts_k{TOP_K}.json"
    out_path = rag_data_dir / out_name
    out_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    accuracy = db_match_count / len(test_cases)
    log.info("SUMMARY  TOP_K=%d  database retrieval accuracy: %d/%d = %.1f%%",
              TOP_K, db_match_count, len(test_cases), accuracy * 100)
    log.info("This is a diagnostic on retrieval quality, NOT the final metric -- "
              "it just tells you how often step 2 picked the right database "
              "before Qwen even runs. Saved -> %s", out_path)


if __name__ == "__main__":
    main()
