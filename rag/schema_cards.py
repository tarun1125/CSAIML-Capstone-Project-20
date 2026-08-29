import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("schema_cards")

ROOT = Path(__file__).resolve().parents[1]

# COLLECTIONS used to be a hand-maintained dict listing 6 databases. It
# drifted out of sync with reality twice: round 2 added 7 more databases'
# local dumps without this dict being permanently updated to match (a
# rag/schema_cards.json with 13 databases exists on disk as evidence someone
# DID expand it temporarily -- that edit was never committed), and round 3
# added 10 more on top with the same dict still stuck at 6. Rather than
# patch the dict a third time and set up a fourth drift for round 4, this
# now auto-discovers collections directly from what's actually on disk
# under database/mongodb/ -- there is no separate list left to fall out of
# sync. Run dump_atlas_to_local.py (repo root) after any future
# dataset-expansion round to populate local dumps for new databases; this
# file picks them up automatically, no code change needed here.
_MONGODB_ROOT = ROOT / "database" / "mongodb"

# One known exception: car_1's real Atlas collection name is literally
# "model_list.json" (the ".json" is part of the name, not a file-extension
# artifact) -- Path.stem would strip it like a normal suffix and get the
# collection name wrong. This is the only case found so far; if a future
# database has the same quirk, add it here rather than guessing.
_COLLECTION_NAME_OVERRIDES = {
    ("car_1", "model_list"): "model_list.json",
}


def _discover_collections() -> dict[str, list[str]]:
    if not _MONGODB_ROOT.exists():
        log.warning("%s does not exist -- no local dumps to discover collections from. "
                    "Run dump_atlas_to_local.py first.", _MONGODB_ROOT)
        return {}
    discovered: dict[str, list[str]] = {}
    for db_dir in sorted(p for p in _MONGODB_ROOT.iterdir() if p.is_dir()):
        colls = []
        for f in sorted(db_dir.glob("*.json")):
            colls.append(_COLLECTION_NAME_OVERRIDES.get((db_dir.name, f.stem), f.stem))
        if colls:
            discovered[db_dir.name] = colls
    log.info("Auto-discovered %d database(s) with local dumps: %s",
              len(discovered), sorted(discovered))
    return discovered


COLLECTIONS = _discover_collections()

# Finding 5: Foreign-key (join) relationships between collections, derived
# from the 305-case gold query corpus by parsing every $lookup stage's
# from/localField/foreignField triple. Only canonical edges are kept --
# intermediate field names created by aggregation workarounds (e.g.
# "CountryInt", "sid_int") are replaced with the real schema field name.
# Type-mismatch annotations (str->int) flag joins that need $toInt/$toDouble
# conversion; these are the same 3 confirmed mismatches documented in the
# project status doc (concert.Stadium_ID, singer_in_concert.Singer_ID,
# car_makers.Country).
#
# NOTE: only covers the original 6 databases so far -- extending this to
# the 17 newer databases (by parsing their own gold queries' $lookup stages
# the same way) is still open, tracked separately in the project status doc.
# Not required for schema/field-type coverage, only for FK annotations.
#
# Format: FK_EDGES[database] = [(local_coll, local_field, foreign_coll, foreign_field, note), ...]
# where note is "" or a type-mismatch warning.
FK_EDGES = {
    "concert_singer": [
        ("concert", "Stadium_ID", "stadium", "Stadium_ID", "str->int, use $toInt"),
        ("singer_in_concert", "concert_ID", "concert", "concert_ID", ""),
        ("singer_in_concert", "Singer_ID", "singer", "Singer_ID", "str->int, use $toInt"),
    ],
    "car_1": [
        ("car_makers", "Country", "countries", "CountryId", "str->int, use $toInt"),
        ("car_makers", "Id", "model_list.json", "Maker", ""),
        ("model_list.json", "Model", "car_names", "Model", ""),
        ("car_names", "MakeId", "cars_data", "Id", ""),
        ("continents", "ContId", "countries", "Continent", ""),
    ],
    "pets_1": [
        ("Has_Pet", "PetID", "Pets", "PetID", ""),
        ("Has_Pet", "StuID", "Student", "StuID", ""),
    ],
    "network_1": [
        ("Friend", "student_id", "Highschooler", "ID", ""),
        ("Highschooler", "ID", "Likes", "student_id", ""),
    ],
    "dog_kennels": [
        ("Dogs", "dog_id", "Treatments", "dog_id", ""),
        ("Dogs", "owner_id", "Owners", "owner_id", ""),
        ("Dogs", "breed_code", "Breeds", "breed_code", ""),
        ("Professionals", "professional_id", "Treatments", "professional_id", ""),
        ("Treatments", "treatment_type_code", "Treatment_Types", "treatment_type_code", ""),
    ],
    "world_1": [
        ("country", "Code", "city", "CountryCode", ""),
        ("country", "Code", "countrylanguage", "CountryCode", ""),
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

            # Build FK annotation lines for this collection
            fk_lines = []
            db_edges = FK_EDGES.get(db, [])
            for lcoll, lfield, fcoll, ffield, note in db_edges:
                if lcoll == coll:
                    annotation = f"{lfield} -> {fcoll}.{ffield}"
                    if note:
                        annotation += f" ({note})"
                    fk_lines.append(annotation)
            if fk_lines:
                log.info("[%s.%s] FK edges: %s", db, coll, "; ".join(fk_lines))

            text = f"database: {db}\ncollection: {coll}\nfields: " + ", ".join(
                f"{k} ({v})" for k, v in fields.items()
            )
            if fk_lines:
                text += "\nFK: " + "; ".join(fk_lines)

            cards.append({
                "id": f"{db}.{coll}",
                "database": db,
                "collection": coll,
                "fields": fields,
                "fk_edges": fk_lines,
                "doc_count": len(docs),
                "text": text,
            })
    return cards


if __name__ == "__main__":
    cards = build_cards()
    out = Path(__file__).parent / "schema_cards.json"
    out.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    dbs = sorted(set(c["database"] for c in cards))
    print(f"wrote {len(cards)} cards across {len(dbs)} database(s) -> {out}")
    print(f"databases covered: {dbs}")
