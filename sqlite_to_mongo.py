import json
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sqlite_to_mongo")

ROOT = Path(__file__).resolve().parent

db_name = sys.argv[1] if len(sys.argv) > 1 else "pets_1"
db_path = ROOT / "database" / "sqlite" / db_name / f"{db_name}.sqlite"

# sqlite3.connect() does NOT error on a missing file -- it silently creates a
# new, empty database there instead. Without this check, a wrong db_name or
# running this from the wrong working directory produces a silent no-op: no
# tables, zero .json files written, and no error telling you why.
if not db_path.exists():
    raise FileNotFoundError(
        f"{db_path} not found -- check the database name and that you're "
        f"running this from the repo root (or pass a valid name, e.g. "
        f"`python sqlite_to_mongo.py concert_singer`)"
    )

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
).fetchall()
log.info("[%s] found %d table(s) in %s", db_name, len(tables), db_path)

output_dir = ROOT / "database" / "mongodb" / db_name
output_dir.mkdir(parents=True, exist_ok=True)

written = 0
for (table,) in tables:
    if table == "sqlite_sequence":
        continue  # internal SQLite bookkeeping table, not real data
    rows = cursor.execute(f"SELECT * FROM {table}").fetchall()

    data = [dict(row) for row in rows]

    out_path = output_dir / f"{table}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("[%s.%s] wrote %d doc(s) -> %s", db_name, table, len(data), out_path)
    written += 1

conn.close()
log.info("[%s] done -- %d collection(s) written to %s", db_name, written, output_dir)
