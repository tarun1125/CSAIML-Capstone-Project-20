import sqlite3
import json
from pathlib import Path

db_path = "database\\sqlite\\pets_1\\pets_1.sqlite"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
).fetchall()

output_dir = Path("pets_1")
output_dir.mkdir(exist_ok=True)

for (table,) in tables:
    rows = cursor.execute(f"SELECT * FROM {table}").fetchall()

    data = [dict(row) for row in rows]

    with open(output_dir / f"{table}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

conn.close()