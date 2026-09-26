# Cross-encoder reranking of FAISS candidates (docs/EXPERIMENT-reranking.md §4).
#
#   retrieve top-N with FAISS (N=50)  ->  cross-encoder scores each
#   (question, exemplar) pair jointly  ->  re-sort  ->  take top-K
#
# WHY A CROSS-ENCODER CAN BEAT THE BI-ENCODER HERE, specifically: the FAISS
# index embeds the exemplar's `question` text ONLY -- a deliberate, documented
# choice in build_retrieval_index.py, on the reasoning that mixing query syntax
# into the embedded string "blurs the semantic signal". That leaves the ranking
# resting on a single 384-dim vector per side, compared once. A cross-encoder
# sees both texts jointly and attends across them, so it can use signal the
# bi-encoder's single-vector bottleneck throws away.
#
# If reranking does NOT help, that is not a null result to bury -- it is
# evidence the bi-encoder was already saturating this pool, which is a finding
# about the corpus. Report it as one.
#
# SECOND ARM: rerank on `question + normalized_query` instead of question alone.
# This directly tests the choice build_retrieval_index.py documents. A
# cross-encoder may not suffer the blurring a bi-encoder does, because it is not
# forced to compress both signals into one vector. One extra run, and it either
# confirms the original reasoning or corrects it.
#
# NOT sentence-transformers, deliberately. That package is not in this venv and
# adding it would pull in a FOURTH copy of libomp.dylib alongside torch's,
# faiss's and sklearn's -- the exact hazard the import-order comment below
# exists because of. embed_utils.py already made this call for the bi-encoder
# ("the repo already depends on transformers/torch... this adds zero new
# dependencies"); the same reasoning applies unchanged here.
#   sentence-transformers' CrossEncoder is a thin wrapper over exactly this:
#   AutoModelForSequenceClassification with num_labels=1, raw logit as the
#   score, no activation. Scores from this module are comparable to anything
#   published with that class.
#
# Usage:
#   python rag/rerank.py                 # question-only arm, N=50
#   python rag/rerank.py --include-query # question + normalized_query arm
#   python rag/rerank.py --top-n 50 --batch-size 64

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# IMPORT ORDER IS LOAD-BEARING -- torch BEFORE faiss. This venv ships three
# copies of libomp.dylib (torch/lib, faiss/.dylibs, sklearn/.dylibs); whichever
# loads first wins the process, and if faiss's wins, the first torch forward
# pass segfaults with SIGSEGV, exit 139, no traceback and nothing catchable.
# Same bug documented at length in build_prompts.py. Ruff/isort will want to
# merge these blocks back together alphabetically. Do not let it -- the blank
# line and this comment are the only things keeping them apart.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from embed_utils import embed  # noqa: E402
# faiss now arrives through FaissRetriever, which imports it lazily when it is
# constructed -- still strictly after torch above.
from retrievers import FaissRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DATA = REPO_ROOT / "rag" / "data"
CACHE_PATH = RAG_DATA / "rerank_scores.json"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 50
# 512 is the model's own position-embedding limit. It matters for the
# include-query arm: question + a 400-character aggregation pipeline is long
# enough that truncation is a real effect, not a formality. Truncation is
# applied to the PAIR, longest-first, which is what the tokenizer's
# `truncation=True` does for a two-sequence encode -- so the question survives
# and the tail of the pipeline is what gets cut.
MAX_LENGTH = 512

log = logging.getLogger("rag.rerank")

_tokenizer = None
_model = None


def get_cross_encoder():
    """Lazy-loads once per process, mirroring embed_utils.get_embedder(). ~90MB,
    CPU-fine, no GPU required."""
    global _tokenizer, _model
    if _tokenizer is None:
        log.info("Loading cross-encoder %s (first call only)...", CROSS_ENCODER_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL).eval()
        log.info("Cross-encoder ready.")
    return _tokenizer, _model


def exemplar_text(exemplar: dict, include_query: bool) -> str:
    """The exemplar side of the pair. The question-only form is the direct
    analogue of what the FAISS index holds, which is what makes arm 2 a clean
    test of the include-the-query choice rather than a confound."""
    if not include_query:
        return exemplar["question"]
    return f"{exemplar['question']}\n{exemplar['normalized_query']}"


def variant_key(include_query: bool) -> str:
    return "question_plus_query" if include_query else "question_only"


def score_pairs(query: str, texts: list[str], batch_size: int = 64) -> list[float]:
    """Raw cross-encoder logits, one per text, in input order. Higher = more
    relevant. No sigmoid: the scores are only ever compared against each other
    within one case, and a monotone squashing changes nothing about the
    resulting order while making the numbers harder to read."""
    tokenizer, model = get_cross_encoder()
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            [query] * len(chunk), chunk,
            padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
        scores.extend(logits.squeeze(-1).tolist())
    return scores


def rerank_order(candidate_rows: list[int], scores: list[float]) -> list[int]:
    """Re-sort candidate row indices by cross-encoder score, best first.

    THE TIE-BREAK IS WRITTEN DOWN AND KEPT (spec §2): ties fall back to the
    candidate's ORIGINAL FAISS rank, not to its row index and not to whatever
    order Python's sort happens to produce. Two exemplars scoring identically
    then resolve to "the one the bi-encoder liked better", which is both a
    defensible prior and -- the part that matters -- reproducible across runs
    and machines.
    """
    order = sorted(range(len(candidate_rows)), key=lambda i: (-scores[i], i))
    return [candidate_rows[i] for i in order]


def load_cache() -> dict:
    """Cached scores keyed by variant -> "case_id||exemplar_id" -> float.

    Keyed by variant because the two arms score the same (case, exemplar) pairs
    against DIFFERENT exemplar text; a flat key would let arm 2 silently read
    arm 1's scores and produce a perfect, meaningless tie between the arms.
    """
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def build_candidates(top_n: int = DEFAULT_TOP_N, retriever=None) -> tuple[list[dict], list[dict], list[list[int]], list[list[float]]]:
    """Top-N candidate rows for all 304 test cases, plus their scores. Shared by
    both arms and by the baseline, so every arm reranks the SAME candidate set
    -- the only variable is the ordering.

    `retriever` (rag/retrievers.py) defaults to FAISS, which reproduces the
    original inline search exactly. rag/eval_retriever.py passes an Azure one
    to get ranking metrics for the Azure arms from this same code path."""
    if retriever is None:
        retriever = FaissRetriever()
    test_cases = json.loads((RAG_DATA / "rag_test.json").read_text(encoding="utf-8"))
    metadata = json.loads((RAG_DATA / "fewshot_metadata.json").read_text(encoding="utf-8"))
    log.info("retriever=%s, %d test cases, top_n=%d", retriever.describe(), len(test_cases), top_n)

    rows, sims = [], []
    for case in test_cases:
        qvec = embed([case["question"]])
        hits = retriever.search(case["question"], qvec, top_n)
        rows.append([r for r, _ in hits])
        sims.append([sc for _, sc in hits])
    return test_cases, metadata, rows, sims


def rerank_all(include_query: bool, top_n: int = DEFAULT_TOP_N, batch_size: int = 64) -> dict:
    """Score every (case, candidate) pair for one arm and cache the result.

    304 x 50 = 15,200 pairs. Cached to rag/data/rerank_scores.json so that
    re-running the metrics -- which is the thing you actually iterate on --
    never re-runs the model.
    """
    test_cases, metadata, rows, _sims = build_candidates(top_n)
    cache = load_cache()
    variant = variant_key(include_query)
    bucket = cache.setdefault(variant, {})

    started = time.time()
    computed = 0
    for n, (case, candidate_rows) in enumerate(zip(test_cases, rows), start=1):
        case_id = case["id"]
        missing_rows = [r for r in candidate_rows
                        if f"{case_id}||{metadata[r]['id']}" not in bucket]
        if missing_rows:
            texts = [exemplar_text(metadata[r], include_query) for r in missing_rows]
            scores = score_pairs(case["question"], texts, batch_size=batch_size)
            for row, score in zip(missing_rows, scores):
                bucket[f"{case_id}||{metadata[row]['id']}"] = score
            computed += len(missing_rows)
        if n % 25 == 0 or n == len(test_cases):
            log.info("[%s] %d/%d cases, %d pairs scored, %.1fs elapsed",
                     variant, n, len(test_cases), computed, time.time() - started)

    save_cache(cache)
    log.info("[%s] done: %d new pairs in %.1fs -> %s",
             variant, computed, time.time() - started, CACHE_PATH)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-query", action="store_true",
                        help="score against question + normalized_query (arm 2) "
                             "instead of question alone")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rerank_all(include_query=args.include_query, top_n=args.top_n, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
