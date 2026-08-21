import argparse
import json
import logging
import sys
from pathlib import Path

from pymongo import MongoClient
from pymongo.server_api import ServerApi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rag"))
from schema_cards import COLLECTIONS  # noqa: E402  single source of truth -- see rag/schema_cards.py

log = logging.getLogger("atlas_verify_and_load")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mask_uri(uri: str) -> str:
    """mongodb+srv://user:password@host/... -> mongodb+srv://user:***@host/...
    so connection details stay traceable in logs without leaking the secret."""
    if "@" not in uri:
        return uri
    creds, rest = uri.split("@", 1)
    if ":" in creds:
        user, _password = creds.rsplit(":", 1)
        return f"{user}:***@{rest}"
    return f"{creds}@{rest}"


def connect() -> MongoClient:
    env_file = ROOT / "atlas-credentials.env"
    if not env_file.exists():
        raise RuntimeError(f"{env_file} not found -- this script must run where that file lives")

    env_values = load_env_file(env_file)
    uri = env_values.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(f"MONGODB_URI not found in {env_file}")

    log.info("Connecting to %s", mask_uri(uri))
    client = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    log.info("Connected OK")
    return client


def verify(client: MongoClient) -> dict[str, list[str]]:
    """Logs a full per-db/per-collection report and returns {db: [missing
    collection names]}. A collection that exists but has 0 documents counts
    as missing -- an empty collection fails a query exactly like a missing
    one (returns an empty cursor, not an error), so silently ignoring it
    would just reproduce BACKLOG.md #1 with extra steps."""
    missing_by_db: dict[str, list[str]] = {}
    for db_name, expected_colls in COLLECTIONS.items():
        real_colls = set(client[db_name].list_collection_names())
        missing = [c for c in expected_colls if c not in real_colls]
        present = [c for c in expected_colls if c in real_colls]

        for coll in present:
            cnt = client[db_name][coll].count_documents({})
            log.info("[%s.%s] %d docs", db_name, coll, cnt)
            if cnt == 0:
                log.warning("[%s.%s] exists but is EMPTY -- treating as missing", db_name, coll)
                missing.append(coll)

        status = "OK" if not missing else "INCOMPLETE"
        log.info("[%s] status=%s  %d/%d collections OK%s",
                  db_name, status, len(expected_colls) - len(missing), len(expected_colls),
                  f"  MISSING: {missing}" if missing else "")

        if missing:
            missing_by_db[db_name] = missing
    return missing_by_db


def load_collection(client: MongoClient, db_name: str, coll: str, force: bool) -> bool:
    # coll "model_list.json" (car_1) is the real Atlas collection name --
    # the ".json" is literally part of it, not a file extension to strip.
    # See rag/schema_cards.py for the same handling.
    filename = coll if coll.endswith(".json") else f"{coll}.json"
    path = ROOT / "database" / "mongodb" / db_name / filename
    if not path.exists():
        log.error("[%s.%s] no local dump at %s -- run `python sqlite_to_mongo.py %s` first",
                   db_name, coll, path, db_name)
        return False

    docs = json.loads(path.read_text(encoding="utf-8"))
    if not docs:
        log.warning("[%s.%s] local dump at %s is empty, skipping", db_name, coll, path)
        return False

    target = client[db_name][coll]
    if force:
        deleted = target.delete_many({}).deleted_count
        if deleted:
            log.info("[%s.%s] --force: cleared %d existing doc(s)", db_name, coll, deleted)

    result = target.insert_many(docs)
    log.info("[%s.%s] inserted %d doc(s) from %s", db_name, coll, len(result.inserted_ids), path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", action="store_true",
                         help="Load any missing/empty collections from database/mongodb/*.json")
    parser.add_argument("--force", action="store_true",
                         help="With --load: drop and reload every expected collection, not just missing ones")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    client = connect()
    missing_by_db = verify(client)

    if not missing_by_db and not args.force:
        log.info("All expected collections present and non-empty. Atlas is ready -- nothing to do.")
        return

    if not args.load:
        total_missing = sum(len(v) for v in missing_by_db.values())
        log.warning("%d collection(s) missing/empty across %d database(s). "
                    "Re-run with --load to fix, or --load --force to reload everything.",
                    total_missing, len(missing_by_db))
        for db_name, colls in missing_by_db.items():
            log.warning("  %s: %s", db_name, colls)
        return

    targets = COLLECTIONS if args.force else missing_by_db
    loaded, failed = 0, 0
    for db_name, colls in targets.items():
        for coll in colls:
            if load_collection(client, db_name, coll, args.force):
                loaded += 1
            else:
                failed += 1

    log.info("Load pass complete: %d collection(s) loaded, %d failed. Re-verifying...", loaded, failed)
    remaining = verify(client)
    if remaining:
        log.error("Still missing after load: %s -- check the errors above", remaining)
        sys.exit(1)
    else:
        log.info("All expected collections now present and non-empty. Atlas is ready.")


if __name__ == "__main__":
    main()
