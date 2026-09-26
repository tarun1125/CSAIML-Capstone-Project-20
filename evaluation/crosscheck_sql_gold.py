# Cross-checks this project's hand-written PyMongo GOLD queries against
# Spider's own gold SQL, by executing the SQL on Spider's original SQLite
# databases and comparing the two answers.
#
# What this verifies, and what it does not
# ----------------------------------------
# data/reference_queries.json holds 1,517 hand-written NL -> PyMongo pairs.
# 1,396 of them are Spider-derived, and every one of those matches a Spider
# example on (db_id, question text) exactly -- so Spider's gold SQL for the
# same question is recoverable. Running that SQL against Spider's SQLite and
# comparing to the stored Mongo gold result is an INDEPENDENT check of the
# ground truth: a different query language, a different engine, a different
# author.
#
# It answers "is the gold PyMongo query asking the same question as the NL?"
# It does NOT audit the SCORER -- a permissive result comparison letting a
# wrong MODEL query through is a separate problem, covered by
# evaluation/audit_false_positives.py. The two are complementary and neither
# substitutes for the other.
#
# Nothing here touches Atlas. The Mongo side is read from the already-executed
# data/gold_results.json; only the SQL side is executed, and SQLite is local.
#
# A disagreement is a case to READ, not automatically a broken gold query.
# The two stores are not identical by construction -- the SQLite -> MongoDB
# import (sqlite_to_mongo.py) preserved SQLite's column types, so a column
# SQLite stores as TEXT arrives in Mongo as a string, and SQL's arithmetic on
# it can differ from Mongo's. Real, benign sources of disagreement:
#   - SQL AVG/SUM over a TEXT column coerces; $avg over a string yields null
#     unless the gold query added an explicit $toDouble.
#   - SQL NULL handling in aggregates vs Mongo's.
#   - Spider gold SQL is itself known to contain some wrong queries.
# The comparison below is therefore deliberately value-based and tolerant of
# column naming, row order, and float representation, so that what remains is
# a genuine semantic difference rather than a formatting one.
#
# Usage:
#   python evaluation/crosscheck_sql_gold.py
#   python evaluation/crosscheck_sql_gold.py --spider-root ../spider_data --limit 200
#   python evaluation/crosscheck_sql_gold.py --databases college_1,college_2

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPIDER_ROOT = REPO_ROOT.parent / "spider_data"

# Spider ships its examples split across three files and its databases across
# two directories; a given db can live in either, so both are searched.
SPIDER_EXAMPLE_FILES = ["train_spider.json", "train_others.json", "dev.json"]
SPIDER_DB_DIRS = ["database", "test_database"]

FLOAT_TOLERANCE = 1e-6


def norm_question(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_spider_sql(spider_root: Path) -> dict:
    """(db_id, normalized question) -> gold SQL. Where a question appears more
    than once with textually different SQL, the first is kept; the variants
    seen in this dataset are cosmetic (aliased vs unaliased column refs)."""
    out = {}
    for name in SPIDER_EXAMPLE_FILES:
        path = spider_root / name
        if not path.exists():
            continue
        for ex in json.loads(path.read_text(encoding="utf-8")):
            out.setdefault((ex["db_id"], norm_question(ex["question"])), ex["query"])
    return out


def find_sqlite(spider_root: Path, db: str) -> Path | None:
    for d in SPIDER_DB_DIRS:
        p = spider_root / d / db / f"{db}.sqlite"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Value-based comparison
#
# The two sides have genuinely different shapes: SQLite gives tuples of
# scalars with no column names attached, Mongo gives dicts (or a bare scalar
# for count_documents). Comparing them therefore has to go through values.
# This is more permissive than the scorer used for model output, and
# deliberately so -- the goal is to surface cases where the two queries answer
# DIFFERENT QUESTIONS, not cases where they label the same answer differently.
# ---------------------------------------------------------------------------

def scalarize(v):
    """One value in a comparable canonical form. Numbers become floats rounded
    to FLOAT_TOLERANCE so 3 == 3.0 == 3.0000001; numeric strings become
    numbers, because a column SQLite typed TEXT is a string on the Mongo side
    but a number after SQL arithmetic; everything else becomes a stripped
    string."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s = v.strip()
        try:
            return round(float(s), 6)
        except ValueError:
            return s
    return str(v)


def cell(v) -> str:
    """One value as a comparable string. Both sides go through this, so a
    SQL integer 122 and a Mongo integer 122 land on the same token."""
    return str(scalarize(v))


def mongo_dicts(result) -> list[dict]:
    """Gold Mongo result -> a uniform list of dict rows.

    Three shapes have to be flattened first, or the comparison measures
    representation rather than meaning:

      * a bare scalar (count_documents returns an int, not a cursor)
      * a $facet set-difference result, which is ONE document holding an
        ARRAY of the answers -- e.g. [{"StuID": [1009, 1016, ...]}] where the
        SQL side returns 33 separate rows. Several gold queries use $facet to
        express SQL's EXCEPT / INTERSECT, which Mongo has no direct operator
        for, so this shape is common rather than exotic.
      * a list of bare scalars (distinct() returns one).
    """
    if result is None:
        return []
    if isinstance(result, (int, float, str, bool)):
        return [{"value": result}]
    if not isinstance(result, list):
        return []

    # $facet / $setDifference: a single document whose values are all lists.
    if (len(result) == 1 and isinstance(result[0], dict) and result[0]
            and all(isinstance(v, list) for v in result[0].values())):
        rows = []
        for values in result[0].values():
            for v in values:
                rows.append(v if isinstance(v, dict) else {"value": v})
        return rows

    return [r if isinstance(r, dict) else {"value": r} for r in result]


def as_multiset(rows: list[tuple]) -> Counter:
    return Counter(rows)


def mongo_tuples(rows: list[dict], drop: frozenset = frozenset()) -> list[tuple]:
    """Dict rows -> sorted value-tuples, ignoring column NAMES and column
    ORDER (the SQL and PyMongo authors chose both independently), dropping
    the `_id: null` aggregation artifact and any keys named in `drop`."""
    out = []
    for r in rows:
        vals = [v for k, v in r.items()
                if k not in drop and not (k == "_id" and v is None)]
        out.append(tuple(sorted(cell(v) for v in vals)))
    return out


def sql_tuples(rows) -> list[tuple]:
    return [tuple(sorted(cell(v) for v in row)) for row in rows]


def compare(sql_result, mongo_result) -> tuple[str, str]:
    """Returns (verdict, detail). Row multisets are compared unordered --
    neither store guarantees an order the other must match."""
    m_rows = mongo_dicts(mongo_result)
    a = as_multiset(sql_tuples(sql_result))
    b = as_multiset(mongo_tuples(m_rows))

    if a == b:
        return "AGREE", ""
    if not a and not b:
        return "AGREE", "both empty"

    # The gold PyMongo query commonly carries an extra COMPUTED column the SQL
    # doesn't project -- "grades with >= 4 students" returns {_id, count} in
    # Mongo but just `grade` in SQL. That is the same answer with more detail
    # attached, not a disagreement. Test it exactly rather than by loosening
    # the comparison: drop each candidate key (or pair of keys) from every
    # Mongo row and see whether an exact match falls out.
    keys = sorted({k for r in m_rows for k in r})
    sql_arity = len(next(iter(sql_result), ()))
    if m_rows and 0 < sql_arity < len(keys):
        from itertools import combinations
        for size in range(1, len(keys) - sql_arity + 1):
            for drop in combinations(keys, size):
                if as_multiset(mongo_tuples(m_rows, frozenset(drop))) == a:
                    return "AGREE_SQL_SUBSET", (
                        f"identical once Mongo's extra column(s) {list(drop)} "
                        f"are dropped -- SQL projects fewer columns")

    if not a or not b:
        return "DISAGREE", (f"one side empty (sql={sum(a.values())} rows, "
                            f"mongo={sum(b.values())} rows)")
    if sum(a.values()) != sum(b.values()):
        # Same distinct content but a different number of rows is almost always
        # a DISTINCT/GROUP BY difference -- worth separating from a real
        # content mismatch, since it points at a specific, common defect.
        if set(a) == set(b):
            return "DISAGREE_CARDINALITY", (f"same distinct rows, "
                                            f"sql={sum(a.values())} vs mongo={sum(b.values())}")
        return "DISAGREE", f"row count sql={sum(a.values())} vs mongo={sum(b.values())}"
    return "DISAGREE", f"same row count ({sum(a.values())}), different values"


# ---------------------------------------------------------------------------
# Triage
#
# A flat list of disagreements is not actionable -- these fall into a handful
# of causes with very different implications, and only some of them are gold
# defects. Tagging each one says which pile it belongs in.
# ---------------------------------------------------------------------------

CAUSE_NOTES = {
    "TIE_OR_RANK": (
        "'ORDER BY x LIMIT 1' with a TIE at the top. Both answers are equally "
        "valid and the two engines break the tie differently -- the QUESTION is "
        "under-determined, so the gold answer is one arbitrary pick among "
        "several. e.g. spider-network_1-7 asks which grade has the most high "
        "schoolers when grades 9/10/11/12 all have exactly 4."
    ),
    "JOIN_FANOUT": (
        "Same distinct rows, different row COUNTS -- one side's join fans out "
        "and duplicates. Usually a real defect in whichever side duplicates."
    ),
    "SET_OP": (
        "SQL uses EXCEPT/INTERSECT/UNION, expressed in Mongo via $facet. Check "
        "the $facet actually implements the set operation it stands in for."
    ),
    "TEXT_TYPED_NUMERIC": (
        "The column is TEXT in SQLite and therefore a string in Mongo. SQL "
        "compares/aggregates it numerically; Mongo needs an explicit "
        "$toInt/$toDouble, and a plain $gt on a string silently matches "
        "nothing. A row-count gap of a few rows on a range filter is this."
    ),
    "AGGREGATE_OVER_TEXT": (
        "SQL AVG/SUM coerces TEXT to a number; Mongo's $avg/$sum over a string "
        "yields null unless converted first."
    ),
    "SUBQUERY": "SQL uses a correlated/nested SELECT; check the Mongo equivalent.",
    "UNCLASSIFIED": "No pattern matched -- read this one.",
}

_SET_OP = re.compile(r"\b(EXCEPT|INTERSECT|UNION)\b", re.I)
_AGG = re.compile(r"\b(AVG|SUM)\s*\(", re.I)


def classify(sql: str, pymongo: str, verdict: str, detail: str) -> str:
    if verdict == "DISAGREE_CARDINALITY" or "same distinct rows" in detail:
        return "JOIN_FANOUT"
    if _SET_OP.search(sql):
        return "SET_OP"
    if re.search(r"\bLIMIT\s+1\b", sql, re.I) and detail.startswith("same row count (1)"):
        return "TIE_OR_RANK"
    if "$toInt" in pymongo or "$toDouble" in pymongo or '"null"' in pymongo:
        return "TEXT_TYPED_NUMERIC"
    if _AGG.search(sql):
        return "AGGREGATE_OVER_TEXT"
    if sql.upper().count("SELECT") > 1:
        return "SUBQUERY"
    if detail.startswith("row count"):
        # A small row-count gap on a plain range filter is the string-typed
        # column problem showing up without any conversion operator present to
        # give it away.
        return "TEXT_TYPED_NUMERIC"
    return "UNCLASSIFIED"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spider-root", type=Path, default=DEFAULT_SPIDER_ROOT,
                    help="Spider dataset root (holds train_spider.json, database/, ...)")
    ap.add_argument("--databases", help="comma-separated subset of databases to check")
    ap.add_argument("--limit", type=int, help="stop after N cases (for a quick smoke run)")
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "outputs/sql_crosscheck.csv")
    args = ap.parse_args()

    if not args.spider_root.exists():
        raise SystemExit(
            f"Spider root not found: {args.spider_root}\n"
            f"Pass --spider-root pointing at the extracted Spider dataset "
            f"(the directory containing train_spider.json and database/)."
        )

    ref = json.loads((REPO_ROOT / "data/reference_queries.json").read_text(encoding="utf-8"))
    gold = {str(r["id"]): r for r in json.loads(
        (REPO_ROOT / "data/gold_results.json").read_text(encoding="utf-8"))}
    test_ids = {str(r["id"]) for r in json.loads(
        (REPO_ROOT / "rag/data/rag_test.json").read_text(encoding="utf-8"))}

    sql_by_question = load_spider_sql(args.spider_root)
    print(f"[spider] loaded {len(sql_by_question)} unique (db, question) -> SQL pairs "
          f"from {args.spider_root}")

    wanted = set(args.databases.split(",")) if args.databases else None
    conns: dict[str, sqlite3.Connection] = {}
    verdicts = Counter()
    test_verdicts = Counter()
    causes = Counter()
    rows_out = []

    checked = 0
    for case in ref:
        cid = str(case["id"])
        db = case.get("database")
        if wanted and db not in wanted:
            continue
        if args.limit and checked >= args.limit:
            break

        sql = sql_by_question.get((db, norm_question(case.get("question"))))
        if sql is None:
            verdicts["NO_SQL"] += 1
            continue  # hand-written case with no Spider counterpart -- expected

        g = gold.get(cid)
        if g is None or g.get("status") != "PASS":
            verdicts["NO_MONGO_GOLD"] += 1
            continue

        if db not in conns:
            path = find_sqlite(args.spider_root, db)
            if path is None:
                verdicts["NO_SQLITE"] += 1
                continue
            conn = sqlite3.connect(path)
            # Spider databases contain values that are not valid UTF-8; without
            # this they raise on fetch and would be miscounted as SQL errors.
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            conns[db] = conn

        checked += 1
        try:
            got = conns[db].execute(sql).fetchall()
            verdict, detail = compare(got, g.get("result"))
        except Exception as e:                      # noqa: BLE001  report, don't abort the sweep
            verdict, detail = "SQL_ERROR", str(e)

        verdicts[verdict] += 1
        if cid in test_ids:
            test_verdicts[verdict] += 1
        if not verdict.startswith("AGREE"):
            cause = ("SQL_ERROR" if verdict == "SQL_ERROR"
                     else classify(sql, g["query"], verdict, detail))
            causes[cause] += 1
            rows_out.append({
                "id": cid, "database": db, "in_test_set": cid in test_ids,
                "verdict": verdict, "likely_cause": cause, "detail": detail,
                "question": case.get("question"),
                "gold_sql": " ".join(sql.split()),
                "gold_pymongo": g["query"],
            })

    for c in conns.values():
        c.close()

    EXECUTED = ("AGREE", "AGREE_SQL_SUBSET", "DISAGREE", "DISAGREE_CARDINALITY", "SQL_ERROR")
    total = sum(v for k, v in verdicts.items() if k in EXECUTED)
    print(f"\n=== SQL vs PyMongo gold, {total} case(s) executed on both sides ===")
    for k, v in verdicts.most_common():
        pct = f"  ({v / total * 100:5.1f}%)" if total and k in EXECUTED else ""
        print(f"  {k:22s} {v:5d}{pct}")

    t_total = sum(test_verdicts.values())
    if t_total:
        agree = test_verdicts["AGREE"] + test_verdicts["AGREE_SQL_SUBSET"]
        print(f"\n--- restricted to the 304-case held-out test set "
              f"({t_total} checkable) ---")
        for k, v in test_verdicts.most_common():
            print(f"  {k:22s} {v:5d}  ({v / t_total * 100:5.1f}%)")
        print(f"  gold agreement on the test set: {agree}/{t_total} "
              f"({agree / t_total * 100:.1f}%)")

    if causes:
        print("\n--- likely cause of each disagreement ---")
        for k, v in causes.most_common():
            print(f"  {k:22s} {v:5d}")
        print("\nWhat each means:")
        for k, _ in causes.most_common():
            note = CAUSE_NOTES.get(k)
            if note:
                print(f"  {k}:\n    " + note.replace(" -- ", " -- ").strip())

    if rows_out:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n[crosscheck] wrote {len(rows_out)} non-agreeing case(s) -> {args.csv}")
        print("[crosscheck] A disagreement is a case to read, not a verdict on the gold "
              "query -- see this file's header for the benign causes (TEXT-typed numeric "
              "columns, NULL handling, and Spider's own known-bad SQL).")


if __name__ == "__main__":
    main()
