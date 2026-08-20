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
            text = f"database: {db}\ncollection: {coll}\nfields: " + ", ".join(
                f"{k} ({v})" for k, v in fields.items()
            )
            cards.append({
                "id": f"{db}.{coll}",
                "database": db,
                "collection": coll,
                "fields": fields,
                "doc_count": len(docs),
                "text": text,
            })
    return cards


if __name__ == "__main__":
    cards = build_cards()
    out = Path(__file__).parent / "schema_cards.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(f"wrote {len(cards)} cards -> {out}")
