# Audits execution_accuracy for FALSE POSITIVES: cases where a generated
# query is semantically wrong but still scored "correct" because its RESULT
# happened to match gold.
#
# Why this exists
# ---------------
# execution_accuracy is this project's primary metric, and it is the right
# primary metric -- it asks whether the query returns the correct answer
# rather than whether it looks like the reference. But results_match() in
# execute_queries.py compares RESULT SETS, and four properties of that
# comparison let a wrong query through:
#
#   1. empty == empty. A gold query returning [] or 0 matches ANY other
#      query that also returns nothing.
#   2. scalar == scalar. A count is one number; a different, wrong count
#      can coincide with it, especially on small Spider databases.
#   3. Order is never compared. sorted(map(_canonical, ...)) on both sides
#      is deliberate (Mongo has no inherent document order), but it means a
#      reversed or missing .sort() is invisible -- including on questions
#      that explicitly ask for an order.
#   4. Low cardinality. "top 3 X" over a collection with exactly 3 distinct
#      X values returns all of them no matter how (or whether) they are
#      ranked.
#
# What this script does NOT do: it does not re-score anything, does not
# touch the published numbers, and does not need an Atlas connection. It
# reads the already-executed result JSONs and the gold results off disk.
#
# It reports in two layers, deliberately kept separate:
#
#   * CANDIDATES -- flagged mechanically by the heuristics below. A flag is
#     a reason to LOOK, not a verdict. Most flagged cases turn out to be
#     legitimate equivalences (a reversed $lookup join direction, a
#     different field alias, aggregate-vs-find phrasing of the same query).
#   * CONFIRMED -- the subset a human actually read and judged wrong, kept
#     in REVIEWED_FALSE_POSITIVES below with the specific reason. Anything
#     flagged but not in that registry is reported as UNREVIEWED, so a
#     future run over new arms/ids cannot quietly inherit an old verdict.
#
# The same pass also reports FALSE NEGATIVES -- cases scored wrong that a
# stricter reading of the same results says are right (most commonly: the
# model simply didn't project _id away).
#
# Usage:
#   python evaluation/audit_false_positives.py
#   python evaluation/audit_false_positives.py --csv outputs/false_positive_audit.csv

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from execute_queries import (  # noqa: E402  reuse the REAL scorer, don't reimplement it
    _canonical,
    _strip_null_id,
    _values_canonical,
    results_match,
)

# ---------------------------------------------------------------------------
# The three canonical 304-case arms. See CANONICAL_ARTIFACTS.md for why these
# specific files and not their same-named siblings.
# ---------------------------------------------------------------------------

ARMS = {
    "baseline": REPO_ROOT / "rag/data/qwen_baseline_testslice_execution_results_mlx.json",
    "rag": REPO_ROOT / "rag/data/qwen_rag_execution_results_mlx.json",
    "finetuned": REPO_ROOT / "data/finetuned_full304_23db_1000iter_execution_results.json",
}

# ---------------------------------------------------------------------------
# Human-reviewed verdicts
#
# Keyed by (arm, id), NOT by id alone. A false positive is a property of a
# specific generated query, not of a test case -- two arms answering the same
# question produce different queries, and one can be right while the other is
# wrong. Two cases here prove the point directly: on spider-flight_2-75 and
# spider-apartment_rentals-29, fine-tuning's query is defective while RAG's is
# correct. An id-keyed registry would have blamed RAG for both.
#
# Every entry was read side-by-side against its gold query and the question
# text before being listed. The reason is the specific semantic defect, not
# the flag that surfaced it -- a flag is only what made someone look.
# ---------------------------------------------------------------------------

REVIEWED_FALSE_POSITIVES = {
    ("finetuned", "spider-flight_2-75"): (
        "Filters DestAirport where the question and gold filter SourceAirport "
        "('flights DEPARTING from APG'). Both return 0, and empty == empty."
    ),
    ("finetuned", "spider-pets_1-20"): (
        "Drops the PetType == 'cat' filter entirely and matches on pet_age == 3 "
        "alone. The question asks for the student with a CAT aged 3."
    ),
    ("finetuned", "spider-hr_1-65"): (
        "Drops the COMMISSION_PCT != null filter. The question restricts to "
        "employees who HAD a commission; this counts all employees per department."
    ),
    ("finetuned", "spider-college_1-142"): (
        "Counts all documents matching CRS_CODE instead of counting DISTINCT "
        "CLASS_SECTION values. The question asks how many different sections."
    ),
    ("finetuned", "spider-college_2-46"): (
        "Counts every document in `department` instead of distinct dept_name in "
        "`course`. The question asks how many departments OFFER COURSES."
    ),
    ("finetuned", "spider-college_2-124"): (
        "Groups `section` (course sections offered) instead of `takes` (student "
        "enrolments). The question asks which semester had the fewest STUDENTS."
    ),
    ("finetuned", "spider-apartment_rentals-29"): (
        "Averages room_count as a STRING (no $toDouble), so the ranking it sorts "
        "on is meaningless, and it returns the three type codes in a different "
        "order than gold. Passes because apartment_rentals has exactly 3 distinct "
        "apt_type_code values, making 'top 3' return all of them regardless, and "
        "because comparison is order-insensitive. RAG's query for this same id "
        "keeps the $toDouble and is correct."
    ),
    ("finetuned", "spider-store_1-25"): (
        "Sorts hire_date ascending where gold sorts birth_date descending. "
        "'Youngest employee' is about date of birth, not length of service."
    ),
    ("baseline", "spider-store_1-25"): (
        "Same defect as the fine-tuned arm's: sorts hire_date ascending instead "
        "of birth_date descending for 'youngest employee'."
    ),
    ("rag", "spider-store_1-26"): (
        "Sorts hire_date DESCENDING where gold sorts ascending -- the reverse "
        "ranking for 'worked longest'. Passes only because the collection holds "
        "no more than 10 employees, so top-10 is every employee either way."
    ),
    ("baseline", "spider-store_1-26"): (
        "Same reversed ranking as the RAG arm's, expressed as an aggregate "
        "pipeline: $sort hire_date -1 where gold sorts ascending."
    ),
    ("rag", "spider-wine_1-30"): (
        "Sorts on Price AFTER a $project that removed it, so output order is "
        "arbitrary -- and the question explicitly asks for wines ordered by price. "
        "Invisible to an order-insensitive comparison."
    ),
    ("rag", "spider-college_1-140"): (
        "Counts distinct CRS_CODE in COURSE where gold counts distinct CRS_CODE in "
        "CLASS -- 'courses that exist' vs 'courses that have scheduled classes'. "
        "Different questions that happen to return the same number."
    ),
    ("finetuned", "spider-college_1-140"): (
        "Identical defect to the RAG arm's on this id: COURSE instead of CLASS."
    ),
}

# Flagged, read, and judged NOT a false positive -- recorded so a reviewer
# doesn't re-litigate them every run. These are genuine equivalences.
REVIEWED_ACCEPTED = {
    ("rag", "spider-flight_2-75"): "Filters SourceAirport, matching gold exactly. "
                                   "Correct despite the empty-gold flag.",
    ("rag", "spider-apartment_rentals-29"): "Keeps the $toDouble conversion and "
                                            "matches gold's ranking; correct.",
    ("rag", "spider-college_2-59"): "Reversed $lookup join direction; same result set.",
    ("finetuned", "spider-college_2-59"): "Reversed join direction; equivalent.",
    ("rag", "spider-college_2-149"): "Reversed $lookup join direction; same result set.",
    ("finetuned", "spider-college_2-149"): "Reversed join direction; equivalent.",
    ("rag", "spider-store_1-49"): "Joins tracks->genres instead of genres->tracks; equivalent.",
    ("finetuned", "spider-store_1-49"): "Reversed join direction; equivalent.",
    ("rag", "spider-store_1-103"): "Reversed join direction; equivalent.",
    ("finetuned", "spider-store_1-105"): "Reversed join direction; equivalent.",
    ("rag", "spider-college_1-39"): "Groups on the joined side; arithmetically equivalent.",
    ("rag", "spider-college_1-123"): "Reversed join direction; equivalent.",
    ("rag", "spider-college_1-6"): "Reversed join direction plus a field alias; equivalent.",
    ("finetuned", "spider-chinook_1-15"): "Reversed join direction; equivalent.",
    ("finetuned", "spider-hr_1-21"): "$ne:[] existence test instead of group-by; equivalent.",
    ("finetuned", "spider-dog_kennels-76"): "Counts dogs having >=1 treatment instead of "
                                            "distinct dog_id in Treatments; equivalent.",
    ("finetuned", "spider-college_2-62"): "Starts from classroom rather than course, but the "
                                          "later $match on sec.building/semester/year "
                                          "constrains it identically.",
    ("finetuned", "spider-world_1-2"): "Counts DISTINCT languages per continent where gold "
                                       "counts language rows. Arguably MORE correct than gold "
                                       "for 'most diverse'; a gold-quality question, not a "
                                       "model error.",
    ("rag", "cs-m5"): "find().sort(Age,1).limit(1) is equivalent to $min over Age.",
    ("finetuned", "3"): "find().sort(weight,-1).limit(1) is equivalent to $max over weight.",
}


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def parse(query: str):
    try:
        return ast.parse(query.strip(), mode="eval")
    except SyntaxError:
        return None


def same_query(gold_q: str, pred_q: str) -> bool:
    """AST-identical ignoring formatting/quote style. These cases need no
    review at all -- the model reproduced the reference query."""
    g, p = parse(gold_q), parse(pred_q)
    return g is not None and p is not None and ast.dump(g) == ast.dump(p)


def collections_touched(query: str) -> set:
    tree = parse(query)
    if tree is None:
        return set()
    return {
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "db"
    }


def is_empty_result(v) -> bool:
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v == 0
    return not v


def has_sort(q: str) -> bool:
    return ".sort(" in q or "$sort" in q


def has_limit(q: str) -> bool:
    return ".limit(" in q or "$limit" in q


def match_tier(pred, gold) -> str:
    """Which tier of results_match() accepted this -- 'exact' (keys and values
    both matched) or 'key-tolerant' (Finding 3's values-only fallback). The
    key-tolerant tier is strictly more permissive, so it is worth surfacing."""
    if isinstance(pred, list) and isinstance(gold, list):
        try:
            if sorted(map(_canonical, pred)) == sorted(map(_canonical, gold)):
                return "exact"
        except TypeError:
            if pred == gold:
                return "exact"
        if (pred and gold and len(pred) == len(gold)
                and all(isinstance(x, dict) for x in pred)
                and all(isinstance(x, dict) for x in gold)):
            try:
                a = sorted(_values_canonical(_strip_null_id(d)) for d in pred)
                b = sorted(_values_canonical(_strip_null_id(d)) for d in gold)
                if a == b:
                    return "key-tolerant"
            except TypeError:
                pass
        return "no-match"
    return "exact" if pred == gold else "no-match"


def flags_for(record: dict, gold_record: dict, question: str) -> list:
    """Mechanical risk flags. Presence of a flag is a reason to READ the case,
    never a verdict on its own."""
    gold_q, pred_q = gold_record["query"], record["query"]
    gold_res, pred_res = gold_record.get("result"), record.get("result")
    out = []

    if is_empty_result(gold_res):
        out.append("GOLD_EMPTY")            # empty == empty matches anything empty
    if match_tier(pred_res, gold_res) == "key-tolerant":
        out.append("KEY_TOLERANT")         # matched only via the permissive tier
    if isinstance(gold_res, (int, float)) and not isinstance(gold_res, bool):
        out.append("SCALAR_GOLD")          # one number; coincidence is cheap
        if gold_res in (0, 1, 2, 3):
            out.append("SMALL_SCALAR")
    if isinstance(gold_res, list) and len(gold_res) == 1 and isinstance(gold_res[0], dict) \
            and len(gold_res[0]) <= 2:
        out.append("SINGLE_ROW")           # one narrow row; coincidence is cheap

    gc, pc = collections_touched(gold_q), collections_touched(pred_q)
    if gc and pc and gc != pc:
        out.append("DIFF_COLLECTIONS")     # answered from different collections

    if has_sort(gold_q) and not has_sort(pred_q):
        out.append("NO_SORT_IN_PRED")
    if has_limit(gold_q) != has_limit(pred_q):
        out.append("LIMIT_MISMATCH")
    if ORDER_WORDS.search(question or "") and has_sort(gold_q) and has_sort(pred_q):
        # Both sort, but comparison can't see the direction or the key.
        if _sort_specs(gold_q) != _sort_specs(pred_q):
            out.append("SORT_SPEC_DIFFERS")
    return out


import re  # noqa: E402  (used only by the two order helpers below)

ORDER_WORDS = re.compile(
    r"\b(order(ed)?\s+by|sort(ed)?|ascending|descending|alphabetical|in order|"
    r"top\s+\d+|highest|lowest|largest|smallest|most|least|youngest|oldest|longest|shortest)\b",
    re.I,
)
_SORT_SPEC = re.compile(r'"?\$?sort"?\s*[:(]\s*(\{[^}]*\})')


def _sort_specs(q: str) -> list:
    """Sort specs with quoting normalized, so {"a": 1} and {'a': 1} compare
    equal and only a genuinely different key or direction registers."""
    return [s.replace("'", '"') for s in _SORT_SPEC.findall(q)]


# ---------------------------------------------------------------------------
# False negatives -- the other direction
# ---------------------------------------------------------------------------

def false_negative_reason(pred, gold) -> str | None:
    """Cases scored WRONG that a stricter reading says are right. Returns the
    specific over-strictness, or None."""
    if not (isinstance(pred, list) and isinstance(gold, list) and pred and gold):
        return None
    if not (all(isinstance(x, dict) for x in pred) and all(isinstance(x, dict) for x in gold)):
        return None

    def drop_id(rows):
        return [{k: v for k, v in d.items() if k != "_id"} for d in rows]

    a, b = drop_id(pred), drop_id(gold)
    if results_match(a, b):
        return "ONLY_ID_DIFFERS: identical once the un-projected _id is dropped"
    try:
        if set(map(_canonical, a)) == set(map(_canonical, b)) and len(a) != len(b):
            return "SET_EQ_DUPES_DIFFER: same distinct rows, different duplicate counts"
    except TypeError:
        return None
    try:
        va = sorted(_values_canonical(d) for d in a)
        vb = sorted(_values_canonical(d) for d in b)
        if va == vb:
            return "VALUES_EQ_KEYS_DIFFER: same values under different key names"
        if set(map(tuple, va)) == set(map(tuple, vb)):
            return "VALUES_SET_EQ: same value set, different multiplicity"
    except TypeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "outputs/false_positive_audit.csv")
    args = ap.parse_args()

    gold = {str(r["id"]): r for r in json.loads(
        (REPO_ROOT / "data/gold_results.json").read_text(encoding="utf-8"))}
    ref = {str(r["id"]): r for r in json.loads(
        (REPO_ROOT / "data/reference_queries.json").read_text(encoding="utf-8"))}

    # --- gold health first: a false positive caused by broken gold is a
    # --- different problem from one caused by a permissive comparison.
    bad_gold = [i for i, r in gold.items() if r.get("status") != "PASS"]
    empty_gold = [i for i, r in gold.items()
                  if r.get("status") == "PASS" and is_empty_result(r.get("result"))]
    print(f"[gold] {len(gold)} gold results: {len(bad_gold)} failed to execute, "
          f"{len(empty_gold)} returned empty")
    if empty_gold:
        print(f"[gold] empty (any arm returning nothing scores correct here): {sorted(empty_gold)}")

    rows = []
    identical_keys = set()   # (arm, id) pairs that reproduce gold exactly
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"\n[{arm}] SKIPPED: {path} not found")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        correct = [r for r in records if r.get("execution_accuracy") is True]

        identical = confirmed = unreviewed = accepted = 0
        flag_counts = Counter()
        for r in correct:
            cid = str(r["id"])
            g = gold.get(cid)
            if g is None or g.get("status") != "PASS":
                continue
            question = (ref.get(cid) or {}).get("question", "")

            if same_query(g["query"], r["query"]):
                identical += 1
                identical_keys.add((arm, cid))
                continue

            fl = flags_for(r, g, question)
            flag_counts.update(fl)
            key = (arm, cid)
            if key in REVIEWED_FALSE_POSITIVES:
                verdict, note = "CONFIRMED_FP", REVIEWED_FALSE_POSITIVES[key]
                confirmed += 1
            elif key in REVIEWED_ACCEPTED:
                verdict, note = "REVIEWED_OK", REVIEWED_ACCEPTED[key]
                accepted += 1
            elif fl:
                verdict, note = "UNREVIEWED_CANDIDATE", ""
                unreviewed += 1
            else:
                continue  # differs from gold but trips no heuristic

            rows.append({
                "arm": arm, "id": cid, "database": r.get("database"),
                "verdict": verdict, "flags": "|".join(fl),
                "tier": match_tier(r.get("result"), g.get("result")),
                "question": question, "gold_query": g["query"], "pred_query": r["query"],
                "note": note,
            })

        n = len(correct)
        adj = n - confirmed
        print(f"\n=== {arm} ===")
        print(f"  scored correct           : {n}")
        print(f"  AST-identical to gold    : {identical}  (no review needed)")
        print(f"  confirmed FALSE POSITIVES: {confirmed}  ({confirmed / n * 100:.1f}% of correct)"
              if n else "  confirmed FALSE POSITIVES: 0")
        print(f"  reviewed & accepted      : {accepted}")
        print(f"  UNREVIEWED candidates    : {unreviewed}")
        print(f"  corrected accuracy       : {adj}/{len(records)} "
              f"({adj / len(records) * 100:.1f}%)  [published: {n}/{len(records)} "
              f"({n / len(records) * 100:.1f}%)]")
        if flag_counts:
            print(f"  flags raised: {dict(flag_counts.most_common())}")

        # --- false negatives
        fns = []
        for r in records:
            if r.get("status") != "PASS" or r.get("execution_accuracy") is True:
                continue
            g = gold.get(str(r["id"]))
            if g is None or g.get("status") != "PASS":
                continue
            reason = false_negative_reason(r.get("result"), g.get("result"))
            if reason:
                fns.append((str(r["id"]), reason))
                rows.append({
                    "arm": arm, "id": str(r["id"]), "database": r.get("database"),
                    "verdict": "FALSE_NEGATIVE_CANDIDATE", "flags": reason.split(":")[0],
                    "tier": "no-match",
                    "question": (ref.get(str(r["id"])) or {}).get("question", ""),
                    "gold_query": g["query"], "pred_query": r["query"], "note": reason,
                })
        print(f"  false-NEGATIVE candidates: {len(fns)}")
        for cid, reason in fns:
            print(f"      {cid}: {reason}")

    if rows:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[audit] wrote {len(rows)} row(s) -> {args.csv}")

    # A verdict that no longer lands on a scored-correct case is stale -- the
    # arm was regenerated, or the id moved out of the test split. Say so rather
    # than silently carrying a judgement about a query that no longer exists.
    # An AST-identical case never reaches the registry (it needs no review), so
    # a verdict on one is redundant rather than stale -- don't warn about those.
    seen = {(r["arm"], r["id"]) for r in rows} | identical_keys
    stale = sorted((set(REVIEWED_FALSE_POSITIVES) | set(REVIEWED_ACCEPTED)) - seen)
    if stale:
        print(f"\n[audit] WARNING: {len(stale)} reviewed verdict(s) no longer match a "
              f"scored-correct case -- the underlying query has changed or the id left "
              f"the test split. Re-review before trusting them: {stale}")

    unrev = [r for r in rows if r["verdict"] == "UNREVIEWED_CANDIDATE"]
    if unrev:
        print(f"\n[audit] {len(unrev)} flagged case(s) have no human verdict yet. A flag is "
              f"a reason to read the case, not a finding -- most are legitimate "
              f"equivalences. Review them and add each to REVIEWED_FALSE_POSITIVES or "
              f"REVIEWED_ACCEPTED at the top of this file.")


if __name__ == "__main__":
    main()
