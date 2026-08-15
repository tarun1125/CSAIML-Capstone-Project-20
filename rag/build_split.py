import json
import logging
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_split")

RANDOM_STATE = 42
TEST_SIZE = 21  # 121 - 100, matches the agreed 100/~20 split exactly


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    ref_file = data_dir / "reference_queries.json"

    cases = json.loads(ref_file.read_text(encoding="utf-8"))
    log.info("Loaded %d cases from %s", len(cases), ref_file)

    databases = [c["database"] for c in cases]
    log.info("Full distribution by database: %s", dict(Counter(databases)))

    fewshot_pool, test_set = train_test_split(
        cases,
        test_size=TEST_SIZE,
        stratify=databases,
        random_state=RANDOM_STATE,
    )

    log.info("Few-shot pool: %d cases -- by database: %s",
              len(fewshot_pool), dict(Counter(c["database"] for c in fewshot_pool)))
    log.info("Held-out test: %d cases -- by database: %s",
              len(test_set), dict(Counter(c["database"] for c in test_set)))


    fewshot_ids = {str(c["id"]) for c in fewshot_pool}
    test_ids = {str(c["id"]) for c in test_set}
    overlap = fewshot_ids & test_ids
    assert not overlap, f"LEAK: {overlap} appear in both fewshot pool and test set"
    assert len(fewshot_ids) == len(fewshot_pool), "duplicate ids in fewshot pool"
    assert len(test_ids) == len(test_set), "duplicate ids in test set"
    log.info("No id overlap between the two halves -- confirmed leakage-safe.")

    fewshot_path = data_dir / "rag_fewshot_pool.json"
    test_path = data_dir / "rag_test.json"
    fewshot_path.write_text(json.dumps(fewshot_pool, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_set, indent=2), encoding="utf-8")

    log.info("Saved %d -> %s", len(fewshot_pool), fewshot_path)
    log.info("Saved %d -> %s", len(test_set), test_path)


if __name__ == "__main__":
    main()
