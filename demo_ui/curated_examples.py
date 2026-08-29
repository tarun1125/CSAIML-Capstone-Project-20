"""
Real, source-verified demo examples for the live capstone defense UI.

Every field here (question / database / generated query / execution
status / result) was pulled directly from this repo's own execution-result
JSON files -- NOT hand-typed or fabricated. See the extraction script used
to build this file: demo_ui/README.md -> "How curated_examples.py was built".

Source files (as of the 2026-08-28 epoch-parity-fixed fine-tuning run):
  baseline : rag/data/qwen_baseline_testslice_execution_results_mlx.json
  rag      : rag/data/qwen_rag_execution_results_mlx.json
  finetuned: data/finetuned_full304_23db_1000iter_execution_results.json
  gold     : data/gold_results.json

Each example carries a `pattern` label describing the specific
baseline/RAG/fine-tuned pass-fail combination it demonstrates, chosen to
match the comparison matrix Tarun asked for. IMPORTANT: these were
re-verified against the NEW 1000-iteration fine-tuned adapter (not the
original under-trained 200-iteration one) -- one originally-shortlisted
example (spider-chinook_1-13) was DROPPED here because the fine-tuned
arm's outcome flipped from pass to fail once the epoch-fix adapter was
substituted in, so it no longer demonstrates the pattern it was
originally picked for. Do not re-add it without re-checking all three
arms against the live JSON files again.
"""

import logging

logger = logging.getLogger("demo_ui.curated_examples")

CURATED_EXAMPLES = [
    {
        "id": "spider-network_1-44",
        "pattern": "Baseline hard-fails (wrong schema guess) -> RAG passes -> Fine-tuned passes",
        "question": "What is Kyle's id?",
        "database": "network_1",
        "gold": {
            "query": 'db.Highschooler.find({"name": "Kyle"}, {"_id": 0, "ID": 1})',
            "result": [{"ID": 1934}],
        },
        "baseline": {
            "query": "db.employees.find_one({'first_name': 'Kyle', 'last_name': 'Johnson'}, {'_id': 0})",
            "status": "FAIL",
            "result": None,
            "execution_accuracy": False,
            "note": "Baseline never saw this schema -- it guesses a generic 'employees' collection with "
                    "first_name/last_name fields that don't exist in network_1 (real collection is "
                    "Highschooler, field is 'name'). Classic zero-shot schema hallucination.",
        },
        "rag": {
            "query": "db.Highschooler.find({'name': 'Kyle'}, {'_id': 0, 'ID': 1})",
            "status": "PASS",
            "result": [{"ID": 1934}],
            "execution_accuracy": True,
            "note": "Retrieval surfaced the correct schema card for network_1, so RAG grounds the query "
                    "in the real collection/field names.",
        },
        "finetuned": {
            "query": "db.Highschooler.find({'name': 'Kyle'}, {'_id': 0, 'ID': 1})",
            "status": "PASS",
            "result": [{"ID": 1934}],
            "execution_accuracy": True,
            "note": "Fine-tuning learned the network_1 schema directly into the weights -- same correct "
                    "query as RAG, without needing a retrieval step at inference time.",
        },
    },
    {
        "id": "spider-world_1-2",
        "pattern": "Baseline runs but wrong logic (empty result) -> RAG passes -> Fine-tuned passes",
        "question": "Which continent has the most diverse languages?",
        "database": "world_1",
        "gold": {
            "query": 'db.countrylanguage.aggregate([{"$lookup": {"from": "country", "localField": "CountryCode", '
                     '"foreignField": "Code", "as": "c"}}, {"$unwind": "$c"}, {"$group": {"_id": "$c.Continent", '
                     '"count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 1}, '
                     '{"$project": {"_id": 0, "Continent": "$_id"}}])',
            "result": [{"Continent": "Africa"}],
        },
        "baseline": {
            "query": "db.countries.aggregate([{'$group': {'_id': '$continent', 'num_languages': {'$sum': 1}}}, "
                      "{'$sort': {'num_languages': -1}}, {'$limit': 1}])",
            "status": "PASS",
            "result": [],
            "execution_accuracy": False,
            "note": "Runs without error but is logically wrong on two counts: wrong collection name "
                    "('countries' doesn't exist -> empty result) and it never joins to the language table, "
                    "so even if the collection name were right it would just count country rows, not "
                    "distinct languages.",
        },
        "rag": {
            "query": "db.countrylanguage.aggregate([{'$lookup': {'from': 'country', 'localField': 'CountryCode', "
                      "'foreignField': 'Code', 'as': 'c'}}, {'$unwind': '$c'}, {'$group': {'_id': '$c.Continent', "
                      "'lang_count': {'$sum': 1}}}, {'$sort': {'lang_count': -1}}, {'$limit': 1}, "
                      "{'$project': {'_id': 0, 'Continent': '$_id'}}])",
            "status": "PASS",
            "result": [{"Continent": "Africa"}],
            "execution_accuracy": True,
            "note": "Correctly performs the $lookup join before grouping.",
        },
        "finetuned": {
            "query": "db.country.aggregate([{'$lookup': {'from': 'countrylanguage', 'localField': 'Code', "
                      "'foreignField': 'CountryCode', 'as': 'cl'}}, {'$unwind': '$cl'}, {'$group': {'_id': "
                      "'$Continent', 'langs': {'$addToSet': '$cl.Language'}}}, {'$sort': {'langs': -1}}, "
                      "{'$limit': 1}, {'$project': {'_id': 0, 'Continent': '$_id'}}])",
            "status": "PASS",
            "result": [{"Continent": "Africa"}],
            "execution_accuracy": True,
            "note": "Takes a DIFFERENT valid path than RAG -- joins from country outward and de-dupes "
                    "languages with $addToSet instead of counting rows -- but lands on the same correct "
                    "answer. Good talking point: there isn't one 'correct query', execution accuracy "
                    "checks the RESULT, not textual similarity to gold.",
        },
    },
    {
        "id": "spider-apartment_rentals-61",
        "pattern": "Baseline runs but wrong (empty) -> RAG hard-fails (malformed pipeline) -> Fine-tuned passes",
        "question": "Give me the minimum and maximum bathroom count among all the apartments.",
        "database": "apartment_rentals",
        "gold": {
            "query": 'db.Apartments.aggregate([{"$group": {"_id": None, "min_bathroom_count": {"$min": '
                     '"$bathroom_count"}, "max_bathroom_count": {"$max": "$bathroom_count"}}}, '
                     '{"$project": {"_id": 0, "min_bathroom_count": 1, "max_bathroom_count": 1}}])',
            "result": [{"min_bathroom_count": 1, "max_bathroom_count": 3}],
        },
        "baseline": {
            "query": "db.apartments.aggregate([{'$group': {'_id': None, 'min_bathroom_count': {'$min': "
                      "'$bathroom_count'}, 'max_bathroom_count': {'$max': '$bathroom_count'}}}])",
            "status": "PASS",
            "result": [],
            "execution_accuracy": False,
            "note": "Pipeline logic is actually correct -- it fails purely on a lowercase/uppercase "
                    "collection-name mismatch ('apartments' vs the real 'Apartments'), which returns an "
                    "empty result instead of an error.",
        },
        "rag": {
            "query": 'db.Apartments.aggregate([ {"$group": {"_id": None, "min_bathroom_count": {"$min": '
                     '"$bathroom_count"}}, {"$group": {"_id": None, "max_bathroom_count": {"$max": '
                     '"$bathroom_count"}}} ])',
            "status": "FAIL",
            "result": None,
            "execution_accuracy": False,
            "note": "Malformed pipeline -- two separate $group stages accidentally nested as sibling keys "
                    "inside ONE dict instead of two separate list elements. This is a genuine syntax bug, "
                    "not a logic error -- good example that RAG's retrieved few-shot examples don't "
                    "guarantee syntactic correctness.",
        },
        "finetuned": {
            "query": "db.Apartments.aggregate([{'$group': {'_id': None, 'min_bathroom_count': {'$min': "
                      "'$bathroom_count'}, 'max_bathroom_count': {'$max': '$bathroom_count'}}}, "
                      "{'$project': {'_id': 0, 'min_bathroom_count': 1, 'max_bathroom_count': 1}}])",
            "status": "PASS",
            "result": [{"min_bathroom_count": 1, "max_bathroom_count": 3}],
            "execution_accuracy": True,
            "note": "Gets both the correct casing AND the correct combined-$group structure. Stable "
                    "across both the 200-iter and 1000-iter fine-tuned adapters -- a robust case, not a "
                    "coin-flip.",
        },
    },
    {
        "id": "spider-bike_1-59",
        "pattern": "Baseline hard-fails (unsupported shell chaining) -> RAG passes -> Fine-tuned fails",
        "question": "What is the station station and end station for the trips with the three smallest ids?",
        "database": "bike_1",
        "gold": {
            "query": 'db.trip.find({}, {"_id": 0, "start_station_name": 1, "end_station_name": 1})'
                     '.sort({"id": 1}).limit(3)',
            "result": [
                {"start_station_name": "Howard at 2nd",
                 "end_station_name": "San Francisco Caltrain 2 (330 Townsend)"},
                {"start_station_name": "South Van Ness at Market",
                 "end_station_name": "San Francisco Caltrain 2 (330 Townsend)"},
                {"start_station_name": "Market at Sansome",
                 "end_station_name": "Harry Bridges Plaza (Ferry Building)"},
            ],
        },
        "baseline": {
            "query": "db.trip.find().sort({'id': 1}).limit(3).project({'station': 1, 'end_station': 1, "
                      "'_id': 0})",
            "status": "FAIL",
            "result": None,
            "execution_accuracy": False,
            "note": "Chains .project() onto a cursor after .limit() -- PyMongo cursors don't support that "
                    "chained method (that's Mongo-shell JS style); also invents field names "
                    "'station'/'end_station' that don't exist (real fields are start_station_name / "
                    "end_station_name).",
        },
        "rag": {
            "query": "db.trip.find({}, {'_id': 0, 'start_station_name': 1, 'end_station_name': 1})"
                      ".sort({'id': 1}).limit(3)",
            "status": "PASS",
            "result": [
                {"start_station_name": "Howard at 2nd",
                 "end_station_name": "San Francisco Caltrain 2 (330 Townsend)"},
                {"start_station_name": "South Van Ness at Market",
                 "end_station_name": "San Francisco Caltrain 2 (330 Townsend)"},
                {"start_station_name": "Market at Sansome",
                 "end_station_name": "Harry Bridges Plaza (Ferry Building)"},
            ],
            "execution_accuracy": True,
            "note": "Correct field names and correct PyMongo-valid method chaining (projection passed as "
                    "the find() argument, not chained afterward).",
        },
        "finetuned": {
            "query": "db.trip.find({'id': {'$in': [1, 2, 3]}}, {'_id': 0, 'start_station_name': 1, "
                      "'end_station_name': 1}).sort({'id': 1})",
            "status": "PASS",
            "result": [],
            "execution_accuracy": False,
            "note": "Runs cleanly (correct syntax, correct field names) but silently assumes trip ids "
                    "start at 1 -- {'$in': [1,2,3]} instead of 'smallest 3 ids' via sort+limit -- so it "
                    "returns zero rows on this dataset. HONEST INCLUSION: this is the one case in the demo "
                    "set where the fine-tuned model is worse than RAG -- useful to show a professor that "
                    "you're not cherry-picking only wins.",
        },
    },
    {
        "id": "spider-car_1-105",
        "pattern": "All three arms pass (bonus 'everyone gets it right' example)",
        "question": "What is the maximum accelerate for all the different cylinders?",
        "database": "car_1",
        "gold": {
            "query": 'db.cars_data.aggregate([{"$group": {"_id": "$Cylinders", "max_accelerate": {"$max": '
                     '"$Accelerate"}}}, {"$project": {"_id": 0, "Cylinders": "$_id", "max_accelerate": 1}}, '
                     '{"$sort": {"Cylinders": 1}}])',
            "result": [
                {"max_accelerate": 13.5, "Cylinders": 3},
                {"max_accelerate": 24.8, "Cylinders": 4},
                {"max_accelerate": 20.1, "Cylinders": 5},
                {"max_accelerate": 21, "Cylinders": 6},
                {"max_accelerate": 22.2, "Cylinders": 8},
            ],
        },
        "baseline": {
            "query": "db.cars_data.aggregate([{'$group': {'_id': '$Cylinders', 'max_accelerate': {'$max': "
                      "'$Accelerate'}}}])",
            "status": "PASS",
            "result": [
                {"_id": 6, "max_accelerate": 21}, {"_id": 5, "max_accelerate": 20.1},
                {"_id": 3, "max_accelerate": 13.5}, {"_id": 8, "max_accelerate": 22.2},
                {"_id": 4, "max_accelerate": 24.8},
            ],
            "execution_accuracy": True,
            "note": "This is a one-collection, no-join, no-cast aggregation -- the kind of case where even "
                    "zero-shot baseline is reliable. Good example of 'RAG/fine-tuning matter most on "
                    "harder, multi-table questions, not every question.'",
        },
        "rag": {
            "query": "db.cars_data.aggregate([{'$group': {'_id': '$Cylinders', 'max_accelerate': {'$max': "
                      "'$Accelerate'}}}, {'$project': {'_id': 0, 'Cylinders': '$_id', 'Max_Accelerate': "
                      "'$max_accelerate'}}])",
            "status": "PASS",
            "result": [
                {"Cylinders": 5, "Max_Accelerate": 20.1}, {"Cylinders": 4, "Max_Accelerate": 24.8},
                {"Cylinders": 8, "Max_Accelerate": 22.2}, {"Cylinders": 3, "Max_Accelerate": 13.5},
                {"Cylinders": 6, "Max_Accelerate": 21},
            ],
            "execution_accuracy": True,
            "note": "Correct, just a differently-capitalized field name than gold (Max_Accelerate vs "
                    "max_accelerate) -- the scorer's key-tolerant matching treats this as correct since "
                    "row content matches.",
        },
        "finetuned": {
            "query": "db.cars_data.aggregate([{'$group': {'_id': '$Cylinders', 'max_accelerate': {'$max': "
                      "'$Accelerate'}}}, {'$project': {'_id': 0, 'Cylinders': '$_id', 'max_accelerate': 1}}])",
            "status": "PASS",
            "result": [
                {"max_accelerate": 20.1, "Cylinders": 5}, {"max_accelerate": 21, "Cylinders": 6},
                {"max_accelerate": 13.5, "Cylinders": 3}, {"max_accelerate": 22.2, "Cylinders": 8},
                {"max_accelerate": 24.8, "Cylinders": 4},
            ],
            "execution_accuracy": True,
            "note": "Matches gold field naming exactly; row order differs from gold but the scorer treats "
                    "aggregation result sets as order-independent for unsorted pipelines like this one.",
        },
    },
]


def get_example_by_id(example_id: str):
    for ex in CURATED_EXAMPLES:
        if ex["id"] == example_id:
            return ex
    logger.warning("curated example id not found: %s", example_id)
    return None


logger.info("Loaded %d curated demo examples: %s", len(CURATED_EXAMPLES), [e["id"] for e in CURATED_EXAMPLES])
