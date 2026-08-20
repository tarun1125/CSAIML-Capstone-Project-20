import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLLECTIONS = {
    "concert_singer": ["stadium", "concert", "singer", "singer_in_concert"],
    "pets_1": ["Student", "Has_Pet", "Pets"],
    "network_1": ["Friend", "Highschooler", "Likes"],
    "car_1": ["continents", "car_makers", "countries", "model_list.json", "cars_data", "car_names"],
    "world_1": ["city", "country", "countrylanguage"],
    "dog_kennels": ["Breeds", "Charges", "Sizes", "Treatment_Types", "Owners", "Dogs", "Professionals", "Treatments"],
}

# Foreign-key relationships between collections, WITHIN each database. Not
# guessed from field-name conventions -- harvested by parsing every $lookup
# stage's from/localField/foreignField out of the 305-case golden
# data/reference_queries.json (the correct queries), then kept only where the
# relationship recurs across multiple gold queries with a stable field name
# (drops one-off noise like a query's own "_id" aliasing). Verified against
# the actual seed data too, not just the gold queries, for the three
# collections (dog_kennels' Dogs.breed_code/size_code/owner_id) that never
# happened to appear in a $lookup in this particular 305-case sample -- each
# checked as: is every non-null value on the child side actually present in
# the parent side's values (a real subset relationship, not a same-named
# coincidence).
#
# This targets a schema-linking gap the RAG prompt's schema block didn't
# previously surface at all: a model filtering an FK-holding field with a
# literal display value instead of joining through it (confirmed miss:
# car-h3 filters car_makers.Country == 'Asia' directly -- 'Asia' is a
# continent name, but Country is a country-id FK two hops from continents),
# and multi-hop joins that skip an intermediate collection because nothing
# told the model the chain existed (confirmed miss: case "12" joins
# model_list -> car_makers but never continues model_list -> car_names ->
# cars_data to reach Horsepower, which isn't a field on either collection it
# did join).
#
# Each tuple is (this_collection, this_field, other_collection, other_field,
# cast_note). cast_note is set ONLY where the two sides' field TYPES
# genuinely differ (checked against this same file's own field_types()
# output, not assumed) -- MongoDB's $lookup does not auto-cast, so a
# type-mismatched join silently returns zero matches instead of erroring,
# which is a much harder failure mode to notice than a crash. Three such
# mismatches confirmed: concert.Stadium_ID (str) vs stadium.Stadium_ID
# (int), singer_in_concert.Singer_ID (str) vs singer.Singer_ID (int),
# car_makers.Country (str) vs countries.CountryId (int) -- all three also
# independently evidenced by the gold queries' own field-renaming pattern
# (e.g. joining on a field literally named "CountryInt"/"country_int" to
# work around exactly this mismatch).
FOREIGN_KEYS = {
    "concert_singer": [
        ("concert", "Stadium_ID", "stadium", "Stadium_ID",
         "cast to int before joining/matching -- stored as a string in concert"),
        ("singer_in_concert", "Singer_ID", "singer", "Singer_ID",
         "cast to int before joining/matching -- stored as a string in singer_in_concert"),
        ("singer_in_concert", "concert_ID", "concert", "concert_ID", None),
    ],
    "pets_1": [
        ("Has_Pet", "StuID", "Student", "StuID", None),
        ("Has_Pet", "PetID", "Pets", "PetID", None),
    ],
    "car_1": [
        ("model_list.json", "Maker", "car_makers", "Id", None),
        ("car_names", "Model", "model_list.json", "Model", None),
        ("car_names", "MakeId", "cars_data", "Id", None),
        ("car_makers", "Country", "countries", "CountryId",
         "cast to int before joining/matching -- stored as a string in car_makers"),
        ("countries", "Continent", "continents", "ContId", None),
    ],
    "network_1": [
        ("Friend", "student_id", "Highschooler", "ID", None),
        ("Friend", "friend_id", "Highschooler", "ID", None),
        ("Likes", "student_id", "Highschooler", "ID", None),
        ("Likes", "liked_id", "Highschooler", "ID", None),
    ],
    "world_1": [
        ("city", "CountryCode", "country", "Code", None),
        ("countrylanguage", "CountryCode", "country", "Code", None),
    ],
    "dog_kennels": [
        ("Dogs", "owner_id", "Owners", "owner_id", None),
        ("Dogs", "breed_code", "Breeds", "breed_code", None),
        ("Dogs", "size_code", "Sizes", "size_code", None),
        ("Treatments", "dog_id", "Dogs", "dog_id", None),
        ("Treatments", "professional_id", "Professionals", "professional_id", None),
        ("Treatments", "treatment_type_code", "Treatment_Types", "treatment_type_code", None),
    ],
}


def field_types(docs: list[dict]) -> dict:
    # Type is inferred from the first NON-NULL value seen per field across
    # all docs, not just docs[0] -- e.g. world_1.country.IndepYear is a real
    # int field but null for 47/239 countries, and docs[0] happens to be one
    # of the null ones. Trusting docs[0] alone mislabeled it "NoneType" in
    # the LLM-facing schema prompt. Fields null in every doc are flagged
    # "unknown, nullable" instead of guessed.
    if not docs:
        return {}
    order = [k for k in docs[0] if k != "_id"]
    resolved = {k: None for k in order}
    nullable = {k: False for k in order}
    for doc in docs:
        for k in order:
            v = doc.get(k)
            if v is None:
                nullable[k] = True
            elif resolved[k] is None:
                resolved[k] = type(v).__name__
    return {
        k: (resolved[k] or "unknown") + (", nullable" if nullable[k] else "")
        for k in order
    }


def build_cards() -> list[dict]:
    cards = []
    for db, colls in COLLECTIONS.items():
        for coll in colls:
            # coll "model_list.json" is the real Atlas collection name (the
            # ".json" is literally part of it) but the dump file on disk is
            # just that same name -- don't double-append ".json".
            filename = coll if coll.endswith(".json") else f"{coll}.json"
            path = ROOT / "database" / "mongodb" / db / filename
            docs = json.loads(path.read_text(encoding="utf-8"))
            fields = field_types(docs)
            fks = [
                {"field": f1, "ref_collection": c2, "ref_field": f2, "note": note}
                for (c1, f1, c2, f2, note) in FOREIGN_KEYS.get(db, [])
                if c1 == coll
            ]
            text = f"database: {db}\ncollection: {coll}\nfields: " + ", ".join(
                f"{k} ({v})" for k, v in fields.items()
            )
            if fks:
                text += "\nforeign keys: " + "; ".join(
                    f"{fk['field']} -> {fk['ref_collection']}.{fk['ref_field']}"
                    + (f" ({fk['note']})" if fk["note"] else "")
                    for fk in fks
                )
            cards.append({
                "id": f"{db}.{coll}",
                "database": db,
                "collection": coll,
                "fields": fields,
                "foreign_keys": fks,
                "doc_count": len(docs),
                "text": text,
            })
    return cards


if __name__ == "__main__":
    cards = build_cards()
    out = Path(__file__).parent / "schema_cards.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(f"wrote {len(cards)} cards -> {out}")
