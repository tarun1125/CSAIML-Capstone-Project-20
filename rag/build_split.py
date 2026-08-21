# Splits data/reference_queries.json (the golden set -- shared with the
# baseline arm, stays in the top-level data/ folder) into:
#   - rag/data/rag_fewshot_pool.json  -- the retrieval corpus RAG's FAISS
#     index is built from. Never touched at evaluation time.
#   - rag/data/rag_test.json          -- held out, never embedded into the
#     index, used only to measure whether retrieval helps.
# Both outputs are RAG-specific, so they live under rag/data/ (everything
# RAG touches is grouped under rag/ -- code at the top level, generated
# data/evaluations/results under rag/data/ and rag/outputs/).
#
# Stratified by `database` so every one of the 6 databases is represented
# in the test set in roughly the same proportion it appears in the full
# golden set, instead of leaving that to chance the way a plain random
# split would. Not stratified by database x complexity too -- that would
# create cells too small to honor proportionally at this corpus size, so
# it'd be stratification in name only.
#
# TEST_FRACTION is a proportion, not a fixed count, on purpose: this split
# was originally 100/21 (~17%) when the golden set was 121 cases. Once the
# Spider-derived candidates are translated and merged in, the golden set
# grows to 300+ cases -- a hardcoded TEST_SIZE=21 left over from the old
# count would silently stop being an 80/20 split at all. A fraction scales
# correctly with whatever the golden set's current size is.
#
# random_state is fixed so re-running this script (e.g. after the golden
# set grows) reproduces the exact same split for a given TEST_FRACTION --
# change RANDOM_STATE deliberately, not by re-running until you like the
# numbers.

import json
import logging
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_split")

RANDOM_STATE = 42
TEST_FRACTION = 0.2  # 80/20 split -- was TEST_SIZE=21 (a fixed count) before the dataset expansion


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"  # shared with baseline -- reference_queries.json stays here
    rag_data_dir = root / "rag" / "data"
    rag_data_dir.mkdir(parents=True, exist_ok=True)
    ref_file = data_dir / "reference_queries.json"

    cases = json.loads(ref_file.read_text(encoding="utf-8"))
    log.info("Loaded %d cases from %s", len(cases), ref_file)

    databases = [c["database"] for c in cases]
    log.info("Full distribution by database: %s", dict(Counter(databases)))

    fewshot_pool, test_set = train_test_split(
        cases,
        test_size=TEST_FRACTION,
        stratify=databases,
        random_state=RANDOM_STATE,
    )

    log.info("Few-shot pool: %d cases -- by database: %s",
              len(fewshot_pool), dict(Counter(c["database"] for c in fewshot_pool)))
    log.info("Held-out test: %d cases -- by database: %s",
              len(test_set), dict(Counter(c["database"] for c in test_set)))

    # Sanity check: no id should appear in both halves. This is guaranteed
    # by train_test_split (it partitions, doesn't sample with replacement),
    # but a leaked few-shot example landing in the test set would silently
    # make the RAG number look better than it is -- worth asserting, not
    # just trusting sklearn blindly, since this specific failure mode is
    # the one thing that would invalidate the whole comparison.
    fewshot_ids = {str(c["id"]) for c in fewshot_pool}
    test_ids = {str(c["id"]) for c in test_set}
    overlap = fewshot_ids & test_ids
    assert not overlap, f"LEAK: {overlap} appear in both fewshot pool and test set"
    assert len(fewshot_ids) == len(fewshot_pool), "duplicate ids in fewshot pool"
    assert len(test_ids) == len(test_set), "duplicate ids in test set"
    log.info("No id overlap between the two halves -- confirmed leakage-safe.")

    fewshot_path = rag_data_dir / "rag_fewshot_pool.json"
    test_path = rag_data_dir / "rag_test.json"
    fewshot_path.write_text(json.dumps(fewshot_pool, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_set, indent=2), encoding="utf-8")

    log.info("Saved %d -> %s", len(fewshot_pool), fewshot_path)
    log.info("Saved %d -> %s", len(test_set), test_path)


if __name__ == "__main__":
    main()
