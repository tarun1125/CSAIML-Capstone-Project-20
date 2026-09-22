# Relevance labels for the reranking experiment (docs/EXPERIMENT-reranking.md §2).
#
# ONE definition each, here and nowhere else. The whole point of the experiment
# is to check whether ranking metrics predict execution accuracy; if two modules
# disagreed about what "relevant" means, that check would be measuring the
# disagreement instead.
#
# Neither label needs new annotation -- both are already in the data:
#
#   binary  -- exemplar["database"] == case["database"]. This is the same notion
#              build_prompts.py's db_match already uses, generalised from "did
#              the right database appear at all" to "where in the ranking".
#   graded  -- 0-3, from how much the exemplar's collection set overlaps the
#              case's gold_collections. This is the label that makes nDCG mean
#              something; a flat binary label makes nDCG a re-weighted recall.
#
# LIMITATION, stated here rather than buried in the write-up: same-database is a
# PROXY for relevance, not relevance itself. A cross-database exemplar that
# demonstrates the right aggregation shape may condition the generator better
# than a same-database one that does not. The graded label narrows the gap but
# does not close it. The execution oracle is what actually settles it -- see
# eval_retrieval.py's per-case correlation.

import ast

# Aggregation stages that name a collection other than the pipeline's root.
# $graphLookup/$out/$merge appear zero times in this corpus (checked against all
# 1,517 normalized queries in fewshot_metadata.json + rag_test.json) but cost
# nothing to handle, and a future dataset expansion may introduce them.
_FROM_STAGES = {"$lookup", "$graphLookup"}
_COLL_STAGES = {"$unionWith"}


def collections_in(normalized_query: str) -> set[str]:
    """Collections a normalized query reads: the `db.<coll>` root plus every
    nested stage that names another collection.

    PARSED, NOT REGEXED, and that is load-bearing. 27 queries in this corpus put
    a $lookup inside a $facet; a shallow regex that scans for the first "from"
    after a "$lookup" would still find those, but one that anchors on the
    top-level pipeline shape would not. ast.walk() descends the whole expression,
    so nesting depth stops being something this function has to reason about.

    All 1,517 queries in the corpus parse as Python expressions. A query that
    does not parse returns whatever collections were unambiguous -- an empty set
    -- rather than raising: a single malformed exemplar should degrade one
    relevance grade, not kill a 15,200-pair evaluation run.
    """
    try:
        tree = ast.parse(normalized_query, mode="eval")
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        # db.<collection>.<method>(...)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "db":
                found.add(node.attr)
        # db['<collection>'] -- the escape hatch for a collection whose name is
        # not a Python identifier. `model_list.json` is the one such collection
        # here; it currently only ever appears as a $lookup `from`, but prompt
        # rule 2 tells the model to use this form, so generated queries can.
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "db":
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found.add(key.value)
        elif isinstance(node, ast.Dict):
            for key_node, val_node in zip(node.keys, node.values):
                if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                    continue
                stage = key_node.value
                if stage in _FROM_STAGES:
                    found |= _dict_string_field(val_node, "from")
                elif stage in _COLL_STAGES:
                    # {"$unionWith": "coll"} and {"$unionWith": {"coll": ...}}
                    # are both legal Mongo; only the second shape appears here.
                    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                        found.add(val_node.value)
                    else:
                        found |= _dict_string_field(val_node, "coll")
    return found


def _dict_string_field(node: ast.AST, field: str) -> set[str]:
    """{"from": "Dogs"} -> {"Dogs"}. Empty set if the field is absent or is not
    a string literal (a computed collection name is not resolvable statically,
    and guessing at one would be worse than admitting it)."""
    if not isinstance(node, ast.Dict):
        return set()
    for key_node, val_node in zip(node.keys, node.values):
        if (
            isinstance(key_node, ast.Constant)
            and key_node.value == field
            and isinstance(val_node, ast.Constant)
            and isinstance(val_node.value, str)
        ):
            return {val_node.value}
    return set()


def gold_collections_of(case: dict) -> set[str]:
    """`case["gold_collections"]` as bare collection names.

    NECESSARY because the field is inconsistently qualified: 331 of the 450
    entries across the 304 test cases are written `car_1.cars_data` and the
    other 119 are written bare (`Department`). collections_in() always returns
    bare names, so without this the two sides would never compare equal for
    two-thirds of the set and every graded label would collapse to 1.

    The prefix is stripped only when it actually matches the case's database --
    split on the FIRST dot, not the last. `car_1.model_list.json` must become
    `model_list.json`; rsplit would turn it into `car_1.model_list`, a
    collection that does not exist.
    """
    database = case.get("database")
    out = set()
    for entry in case.get("gold_collections") or []:
        head, sep, tail = entry.partition(".")
        out.add(tail if sep and head == database else entry)
    return out


def binary_relevance(exemplar: dict, case: dict) -> int:
    """1 if same database, else 0. The existing db_match notion."""
    return int(exemplar["database"] == case["database"])


def graded_relevance(exemplar: dict, case: dict) -> int:
    """0-3, for nDCG.

      0  different database
      1  same database, no collection overlap
      2  same database, partial overlap with case["gold_collections"]
      3  same database, exemplar's collection set == gold_collections

    Grade 3 is set EQUALITY, not containment: an exemplar that touches every
    gold collection plus three others is demonstrating a different join shape,
    which is exactly the thing the grade is meant to distinguish.
    """
    if not binary_relevance(exemplar, case):
        return 0
    gold = gold_collections_of(case)
    exemplar_colls = collections_in(exemplar["normalized_query"])
    if not gold or not exemplar_colls:
        # No gold collections recorded (or an unparseable exemplar) -- there is
        # nothing to grade the overlap against, so fall back to the weakest
        # same-database grade rather than inventing a 2 or a 3.
        return 1
    if exemplar_colls == gold:
        return 3
    if exemplar_colls & gold:
        return 2
    return 1
