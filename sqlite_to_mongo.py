import sqlite3
import json
import sys
from pathlib import Path

db_name = sys.argv[1] if len(sys.argv) > 1 else "pets_1"
db_path = f"database/sqlite/{db_name}/{db_name}.sqlite"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
).fetchall()

output_dir = Path("database/mongodb") / db_name
output_dir.mkdir(parents=True, exist_ok=True)

for (table,) in tables:
    if table == "sqlite_sequence":
        continue  # internal SQLite bookkeeping table, not real data
    rows = cursor.execute(f"SELECT * FROM {table}").fetchall()

    data = [dict(row) for row in rows]

    with open(output_dir / f"{table}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

conn.close()