# One-time (well, one-per-round) local export of every collection in every
# test-slice database out of Atlas, so rag/schema_cards.py's build_cards()
# can generate REAL, field-typed schema for all 23 databases the way it
# already does for the original 6 -- instead of the rag_prompts.json
# extraction workaround generate_baseline_mlx.py has been leaning on.
#
# WHY THIS EXISTS: database/mongodb/{db}/*.json currently only has dumps for
# the original 6 databases (car_1, concert_singer, dog_kennels, network_1,
# pets_1, world_1) -- confirmed by directly listing the directory this
# session. The 17 databases round 2 and round 3 added were never locally
# dumped, which is the actual root cause of two things: (1)
# rag/schema_cards.py's COLLECTIONS dict has repeatedly drifted out of sync
# with reality (it's back down to 6 databases right now, even though a
# stale rag/schema_cards.json sitting on disk has 13 -- someone expanded it
# temporarily at some point and that edit was never committed), and (2)
# generate_baseline_mlx.py has to reach for real schema wherever it can find
# it (rag_prompts.json) instead of the normal, local, reproducible path.
#
# THIS SCRIPT FIXES THE ROOT CAUSE, NOT ANOTHER SYMPTOM: once every
# database's collections are dumped locally, rag/schema_cards.py auto-
# discovers COLLECTIONS directly from what's on disk under database/mongodb/
# (see that file's own module comment) -- so there is no dict left to drift
# out of sync. Re-run this script after any future dataset-expansion round
# and the schema pipeline just picks up whatever's new, no manual dict edit
# required ever again.
#
# SAMPLING, NOT A FULL DUMP: unlike the original 6 databases' dumps (full
# collections, no cap), this script caps each collection at SAMPLE_CAP docs
# by default. These local files are used ONLY for field-type inference
# (schema_cards.py's field_types()) -- gold-query execution always runs
# against live Atlas via evaluation/execute_queries.py, never against these
# files. A representative sample is exactly as good as a full dump for
# inferring "what type is this field", and capping keeps this fast/light
# even for wta_1.rankings (510,437 rows) -- pulling that in full would be
# real time and disk cost for zero additional schema-inference value. Pass
# --sample-cap 0 to force a full, uncapped dump if you ever want one anyway.
#
# NEEDS A REAL ATLAS CONNECTION -- run this locally in your own Terminal
# (same as execute_queries.py/score_rag.py), not through a sandboxed tool
# environment with no route to Atlas.
#
# Usage:
#   python dump_atlas_to_local.py                # all 23, 5000-doc sample cap, skips existing files
#   python dump_atlas_to_local.py --sample-cap 0  # full, uncapped dump
#   python dump_atlas_to_local.py --databases college_3 chinook_1   # just the two known gap databases
#   python dump_atlas_to_local.py --overwrite     # re-dump even files that already exist

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "evaluation"))
from execute_queries import connect  # noqa: E402  reuse, don't reimplement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dump_atlas_to_local")

# The 23 databases the current 304-case rag/data/rag_test.json actually
# spans -- confirmed directly from that file's real "database" field values
# this session (not guessed, not copied from an older doc). Original 6 +
# round 2's 7 + round 3's 10.
DATABASES = [
    "car_1", "concert_singer", "dog_kennels", "network_1", "pets_1", "world_1",
    "apartment_rentals", "bike_1", "college_1", "college_2", "flight_2", "hr_1", "store_1",
    "baseball_1", "chinook_1", "college_3", "csu_1", "flight_4", "formula_1",
    "inn_1", "sakila_1", "wine_1", "wta_1",
]

DEFAULT_SAMPLE_CAP = 5000


def local_filename(collection_name: str) -> str:
    # car_1's "model_list.json" is a real Atlas collection name that
    # genuinely contains ".json" -- don't double-append. Every other
    # collection name gets a plain ".json" suffix for its dump file.
    return collection_name if collection_name.endswith(".json") else f"{collection_name}.json"


def dump_collection(client, db_name: str, coll_name: str, sample_cap: int, overwrite: bool) -> str:
    out_path = ROOT / "database" / "mongodb" / db_name / local_filename(coll_name)
    if out_path.exists() and not overwrite:
        log.info("[%s.%s] already dumped at %s -- skipping (pass --overwrite to refresh)",
                  db_name, coll_name, out_path)
        return "skipped"

    cursor = client[db_name][coll_name].find({}, {"_id": 0})
    if sample_cap:
        cursor = cursor.limit(sample_cap)
    docs = list(cursor)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(docs, indent=2, default=str), encoding="utf-8")
    log.info("[%s.%s] dumped %d doc(s) -> %s", db_name, coll_name, len(docs), out_path)
    return "dumped"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--databases", nargs="+", default=DATABASES,
                         help="Subset of databases to dump (default: all 23)")
    parser.add_argument("--sample-cap", type=int, default=DEFAULT_SAMPLE_CAP,
                         help="Max docs per collection, 0 = uncapped full dump (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-dump collections that already have a local file")
    args = parser.parse_args()

    log.info("Target databases (%d): %s", len(args.databases), args.databases)
    log.info("Sample cap per collection: %s", args.sample_cap or "UNCAPPED (full dump)")

    client = connect()

    summary = {"dumped": 0, "skipped": 0, "errors": 0}
    for db_name in args.databases:
        try:
            coll_names = sorted(
                c for c in client[db_name].list_collection_names()
                if not c.startswith("system.")
            )
        except Exception as e:
            log.error("[%s] could not list collections: %s", db_name, e)
            summary["errors"] += 1
            continue

        if not coll_names:
            log.warning("[%s] Atlas reports ZERO collections -- database name typo, or "
                        "this database genuinely isn't loaded on this cluster?", db_name)
            continue

        log.info("[%s] %d real collection(s) in Atlas: %s", db_name, len(coll_names), coll_names)
        for coll_name in coll_names:
            try:
                result = dump_collection(client, db_name, coll_name, args.sample_cap, args.overwrite)
                summary[result] += 1
            except Exception as e:
                log.error("[%s.%s] dump failed: %s", db_name, coll_name, e)
                summary["errors"] += 1

    log.info("=" * 70)
    log.info("DONE. dumped=%d skipped=%d errors=%d", summary["dumped"], summary["skipped"], summary["errors"])
    log.info("Next: python rag/schema_cards.py   (regenerates rag/schema_cards.json "
              "for every database that now has a local dump -- auto-discovered, "
              "no dict to edit by hand)")


if __name__ == "__main__":
    main()
