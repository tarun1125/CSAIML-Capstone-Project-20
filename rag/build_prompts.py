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

# IMPORT ORDER IS LOAD-BEARING -- embed_utils (torch) BEFORE faiss.
#
# This venv ships three separate copies of libomp.dylib: torch/lib/,
# faiss/.dylibs/ and sklearn/.dylibs/. Whichever one is loaded first wins the
# process, and if faiss's copy wins, the first torch forward pass segfaults
# the interpreter -- SIGSEGV, exit 139, no traceback, no Python-level error to
# catch. Reproduced minimally: `import faiss` then torch -> crash on the first
# model(**enc); `import torch` then faiss -> clean run, correct results.
#
# This script used to import faiss first (alphabetical), so `python
# rag/build_prompts.py 10` -- step 3 of the README's "Reproducing a full run"
# -- died silently right after "Embedder ready.", before the first prompt was
# built. It is the likeliest reason rag_prompts.json and the RAG results
# committed alongside it drifted out of sync (see this repo's run manifests).
#
# Ruff/isort will want to re-sort these back into one alphabetical block. Do
# not let it: the blank line and this comment are what keep them apart.
from embed_utils import MODEL_NAME as EMBED_MODEL_NAME
from embed_utils import embed

import faiss  # noqa: E402  MUST come after embed_utils -- see above

from schema_cards import build_cards

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_manifest import write_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_prompts")

# Reranking experiment (docs/EXPERIMENT-reranking.md §5): an optional
# `--rerank <file> <label>` pair, stripped out of argv BEFORE the positional
# parsing below so every existing invocation keeps working byte-identically.
# The file is rag/data/reranked_neighbors_<arm>.json, written by
# rag/eval_retrieval.py --emit-neighbors: {case_id: [exemplar_id, ...]} in
# reranked order. Everything downstream of neighbour selection -- majority
# vote, schema block, prompt header, rules -- is untouched, which is the whole
# point: the ONLY variable between the baseline and a rerank arm is the
# exemplar list.
# Database-prediction policy (docs/EXPERIMENT-rank1-vote.md). Same argv-stripping
# trick as --rerank below, for the same reason: every existing invocation must
# stay byte-identical.
#
#   vote    (default) majority vote over the top-K neighbours' databases --
#           unchanged, current behaviour
#   rank1   take the single nearest neighbour's database, no vote at all
#
# WHY THIS IS WORTH AN ARM: on the recorded retrieval in rag_prompts.json, pure
# rank-1 predicts the database correctly on 275/304 (90.5%) against the vote's
# 258/304 (84.9%) -- better on 20 cases, worse on 3, McNemar exact p = 0.0005.
# The mechanism is in the README: an unweighted k-NN vote keeps counts and
# discards similarity magnitude, so for near-degenerate sibling databases the
# expected neighbour count is proportional to POOL FREQUENCY rather than
# relevance, and the rarer sibling can never win however close it actually is.
#
# The open question this flag exists to answer is whether that +5.6-point
# retrieval gain reaches execution accuracy -- the same question the reranking
# experiment asked, on the lever that still has headroom.
DB_POLICY = "vote"
if "--db-policy" in sys.argv:
    _i = sys.argv.index("--db-policy")
    DB_POLICY = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
    if DB_POLICY not in ("vote", "rank1"):
        raise SystemExit(f"--db-policy must be 'vote' or 'rank1', got {DB_POLICY!r}")

RERANK_FILE = None
RERANK_LABEL = None
if "--rerank" in sys.argv:
    _i = sys.argv.index("--rerank")
    RERANK_FILE = sys.argv[_i + 1]
    RERANK_LABEL = sys.argv[_i + 2]
    del sys.argv[_i:_i + 3]

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
# 2026-08-28, Finding 5 A/B test (MLX-adapted -- see docs/finding5_ab_test_guide.md):
#   python rag/build_prompts.py 10        # FK annotations included (default, current behavior)
#   python rag/build_prompts.py 10 nofk   # FK annotations omitted -- reproduces the pre-Finding-5 prompt
INCLUDE_FK = not (len(sys.argv) > 2 and sys.argv[2].lower() == "nofk")

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
    "similar in shape to one of them\n"
    "7. $size inside a $match/find() FILTER only accepts a literal integer "
    "for an exact array-length match, e.g. {\"field\": {\"$size\": 3}}. It "
    "does NOT accept a comparison operator -- {\"field\": {\"$size\": "
    "{\"$gte\": 4}}} is INVALID and will error at query time. For "
    "'at least/more than/fewer than N items', either (a) use $expr with the "
    "aggregation $size operator: {\"$expr\": {\"$gte\": [{\"$size\": "
    "\"$field\"}, 4]}}, or (b) compute a count via $group/$addFields first, "
    "then $match on that count field."
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
    note = (
        "\nNote: some logically-numeric fields in this schema are stored as "
        f"strings ({', '.join(fields)}) -- use $toInt / $toDouble when "
        "comparing, filtering, or joining on these.\n"
    )
    if db_name == "car_1":
        # cars_data.Horsepower/MPG also hold the LITERAL STRING "null" for
        # some rows (not a real null, not missing -- an actual "null" text
        # value). $toDouble/$toInt on that value throws a MongoDB server
        # error and fails the whole query. Confirmed cause of two real
        # scored failures (spider-car_1-44, spider-car_1-81) -- both cast
        # Horsepower/MPG straight to $toDouble with no "null"-string guard.
        note += (
            "IMPORTANT: cars_data.Horsepower and cars_data.MPG contain the "
            "literal string \"null\" for some rows. ALWAYS exclude it first, "
            "e.g. add {\"MPG\": {\"$ne\": \"null\"}} to your $match stage "
            "BEFORE any $toDouble/$toInt/$avg/$max/$min on that field.\n"
        )
    return note


# 2026-08-28: chinook_1/store_1 retrieval-confusion finding, verified and
# CORRECTED this session. Round 2's dataset-expansion note claimed store_1
# is "confirmed identical underlying data" to chinook_1 -- true for 9 of 11
# collections (Album/Artist/Customer/Employee/Genre/InvoiceLine/MediaType/
# Playlist/Track match byte-for-byte on every field, verified directly
# against database/mongodb/{chinook_1,store_1}/*.json), but NOT exactly:
# Invoice.InvoiceDate is offset by a constant 2 years between the two
# exports (chinook_1's 2009-01-01 == store_1's 2007-01-01, same invoice,
# same amount, same customer, same address), and PlaylistTrack's actual
# playlist/track associations differ between the two dumps (not just
# reordered -- different membership). Because of this, chinook_1 and store_1
# were deliberately NOT merged into one database label this session, even
# though doing so would fix RAG's retrieval confusion (chinook_1's few-shot
# neighbors currently vote store_1 100% of the time, 0 exceptions -- they're
# similar enough in embedding space that content-based retrieval structurally
# cannot tell them apart). A full merge would need to rewrite each of
# chinook_1's ~42 gold queries into store_1's naming AND verify none of them
# touch InvoiceDate or PlaylistTrack (where the merge would silently produce
# a wrong-but-plausible answer) -- not done, flagged as a real follow-up if
# this confusion turns out to matter more than documenting it here. The
# college_1/college_2/college_3 batch-collision bug (see
# fine_tuning/prepare_data_23db.py's COLLISION_GROUPS) is a DIFFERENT root
# cause even though it looks similar on the surface -- that one is a
# fine-tuning batch-composition bug with a clean, low-risk fix (never put
# same-domain databases in one training batch); this one is a RAG
# retrieval-ambiguity problem inherent to the data itself, with no
# equivalently clean fix available.


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


def render_schema_block(db_name: str, cards_by_db: dict, include_fk: bool = True) -> str:
    cards = cards_by_db.get(db_name, [])
    lines = [f"Schema ({db_name}, {len(cards)} collection(s) -- retrieved database, full schema):\n"]
    for c in cards:
        fields = ", ".join(f"{k} ({v})" for k, v in c["fields"].items())
        lines.append(f"- {c['collection']}: {{ {fields} }}")
        # Finding 5: append FK (join) annotations so the model knows which
        # fields link collections and whether a $toInt cast is needed.
        # include_fk=False reproduces the pre-Finding-5 prompt exactly, for
        # the A/B test (see docs/finding5_ab_test_guide.md) -- run this
        # script twice, once per value, rather than diffing two prompt
        # styles that also happen to differ in ways unrelated to FK lines.
        fk_edges = c.get("fk_edges", []) if include_fk else []
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

    # Reranked order, if this is a rerank arm. Loaded once, keyed by case id.
    # row_of_id maps an exemplar id back to its FAISS row so the reranked ids
    # resolve to the same metadata rows a FAISS search would have produced --
    # these are re-orderings of the SAME candidate pool, not a different pool.
    reranked_by_case = None
    if RERANK_FILE:
        reranked_by_case = json.loads(Path(RERANK_FILE).read_text(encoding="utf-8"))
        log.info("Rerank arm '%s': loaded %d reranked orders from %s",
                 RERANK_LABEL, len(reranked_by_case), RERANK_FILE)
    row_of_id = {m["id"]: i for i, m in enumerate(metadata)}

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
        if reranked_by_case is None:
            scores, idxs = index.search(qvec, TOP_K)
            neighbor_rows = [int(i) for i in idxs[0]]
            neighbor_scores = [float(s) for s in scores[0]]
        else:
            # Search the FULL candidate depth the reranker saw, so the cosine
            # scores recorded below are the real ones for these exemplars
            # rather than a lookup against a shorter search that may not
            # contain them at all.
            # str(case_id): some ids in rag_test.json are integers, and JSON
            # object keys are always strings -- so a case id that went into
            # the neighbours file as 12 comes back out as "12".
            order = reranked_by_case[str(case_id)]
            depth = max(len(order), TOP_K)
            scores, idxs = index.search(qvec, depth)
            sim_of_row = {int(r): float(sc) for r, sc in zip(idxs[0], scores[0])}
            neighbor_rows = [row_of_id[e] for e in order[:TOP_K]]
            neighbor_scores = [sim_of_row.get(r, float("nan")) for r in neighbor_rows]
        neighbors = [metadata[i] for i in neighbor_rows]
        neighbor_dbs = [n["database"] for n in neighbors]

        # rank1 deliberately ignores neighbours 2..K for the DATABASE decision
        # only -- all K exemplars still go into the prompt. The two jobs the
        # retrieved list does (pick a schema, supply examples) are separable,
        # and this arm separates them.
        predicted_database = (
            neighbor_dbs[0] if DB_POLICY == "rank1"
            else majority_vote_database(neighbor_dbs)
        )
        database_match = predicted_database == gold_database
        db_match_count += database_match

        system_prompt = (
            PROMPT_HEADER
            + render_examples_block(neighbors)
            + "\n"
            + render_schema_block(predicted_database, cards_by_db, include_fk=INCLUDE_FK)
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
            # The FAISS similarity scores used to be computed and dropped on
            # the floor (`_scores, idxs = index.search(...)`). Rank alone is
            # enough for Recall@k / MRR / nDCG, but not enough to separate a
            # RETRIEVAL GAP (the right exemplar was never retrieved at all)
            # from RETRIEVAL NOISE (it was retrieved, but ranked below
            # distractors that crowded it out) -- with ranks only you can see
            # WHERE something placed but not how close the call was. The index
            # is IndexFlatIP over normalized embeddings, so these are cosine
            # similarities in [-1, 1], higher = more similar, in rank order.
            # Purely additive: nothing above this line reads them.
            # For a rerank arm these are still the FAISS cosine similarities
            # of the selected exemplars, but they are NO LONGER MONOTONE in
            # rank -- the cross-encoder chose the order. `rerank_arm` below is
            # what tells a reader which of the two situations they are in.
            "retrieved_scores": neighbor_scores,
            "rerank_arm": RERANK_LABEL,
            "complexity": case.get("complexity"),
            "system_prompt": system_prompt,
        })

        log.info("[%s] gold_db=%s predicted_db=%s match=%s neighbors=%s",
                  case_id, gold_database, predicted_database, database_match,
                  [n["id"] for n in neighbors])

    if DB_POLICY != "vote":
        suffix = f"_{RERANK_LABEL}" if RERANK_LABEL else ""
        out_name = f"rag_prompts_{DB_POLICY}{suffix}_k{TOP_K}.json"
    elif RERANK_LABEL:
        out_name = f"rag_prompts_{RERANK_LABEL}_k{TOP_K}.json"
    elif not INCLUDE_FK:
        out_name = "rag_prompts_nofk.json"
    elif TOP_K == 10:
        out_name = "rag_prompts.json"
    else:
        out_name = f"rag_prompts_k{TOP_K}.json"
    out_path = rag_data_dir / out_name
    out_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    write_manifest(
        out_path,
        script=__file__,
        stage="retrieval / prompt build",
        top_k=TOP_K,
        include_fk=INCLUDE_FK,
        embedding_model=EMBED_MODEL_NAME,
        faiss_index="rag/data/fewshot.index",
        faiss_index_ntotal=index.ntotal,
        faiss_index_type=type(index).__name__,
        n_cases=len(prompts),
        n_schema_databases=len(cards_by_db),
        n_schema_cards=len(cards),
        database_retrieval_accuracy=db_match_count / len(test_cases) if test_cases else None,
        records_retrieved_scores=True,
        db_policy=DB_POLICY,
        rerank_arm=RERANK_LABEL,
        rerank_neighbors_file=RERANK_FILE,
    )
    if INCLUDE_FK and TOP_K == 10 and not RERANK_LABEL and DB_POLICY == "vote":
        # Also keep a stably-named FK copy for the A/B test (see
        # docs/finding5_ab_test_guide.md) so rag_prompts.json can keep
        # changing with future K-sweeps/database expansions without the
        # A/B test's "fk" reference silently going stale.
        fk_copy_path = rag_data_dir / "rag_prompts_fk.json"
        fk_copy_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
        log.info("Also wrote FK copy -> %s", fk_copy_path)

    accuracy = db_match_count / len(test_cases)
    log.info("SUMMARY  TOP_K=%d  database retrieval accuracy: %d/%d = %.1f%%",
              TOP_K, db_match_count, len(test_cases), accuracy * 100)
    log.info("This is a diagnostic on retrieval quality, NOT the final metric -- "
              "it just tells you how often step 2 picked the right database "
              "before Qwen even runs. Saved -> %s", out_path)


if __name__ == "__main__":
    main()
