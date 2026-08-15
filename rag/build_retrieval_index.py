# Embeds the 100-example few-shot pool (data/rag_fewshot_pool.json, produced
# by build_split.py) with all-MiniLM-L6-v2 and builds a flat FAISS index
# over the embeddings.
#
# IndexFlatIP (brute-force inner product over L2-normalized vectors, i.e.
# cosine similarity) is the right-fit choice at this scale -- 100 vectors is
# nowhere near where an approximate index (IVF/HNSW) starts paying for
# itself; those exist to avoid scanning millions of vectors, and here a
# full scan is a handful of milliseconds. Reaching for IVF/HNSW here would
# be over-engineering for a corpus two orders of magnitude too small to
# need it.
#
# Only embeds the NL `question` text -- not the query or schema -- since
# the retrieval job is "find past questions phrased like this one", and
# adding query/schema text into the embedded string would blur that
# semantic signal with syntax that has nothing to do with question
# similarity.

import json
import logging
from pathlib import Path

import faiss

from embed_utils import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.build_retrieval_index")


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    rag_dir = root / "rag"

    pool_file = data_dir / "rag_fewshot_pool.json"
    pool = json.loads(pool_file.read_text(encoding="utf-8"))
    log.info("Loaded %d few-shot pool cases from %s", len(pool), pool_file)

    questions = [c["question"] for c in pool]
    vecs = embed(questions)
    log.info("Embedded %d questions -> shape %s", len(questions), vecs.shape)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    log.info("Built FAISS IndexFlatIP, ntotal=%d, dim=%d", index.ntotal, vecs.shape[1])

    index_path = rag_dir / "fewshot.index"
    faiss.write_index(index, str(index_path))

    # Metadata is aligned by row position to the FAISS index (row i in the
    # index == metadata[i]) -- this is the only thing that lets a FAISS
    # search result (a list of row indices) be turned back into an actual
    # question/query/database.
    metadata = [
        {
            "id": c["id"],
            "question": c["question"],
            "database": c["database"],
            "complexity": c.get("complexity"),
            "normalized_query": c["normalized_query"],
        }
        for c in pool
    ]
    metadata_path = rag_dir / "fewshot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Saved FAISS index -> %s", index_path)
    log.info("Saved metadata (row-aligned) -> %s", metadata_path)


if __name__ == "__main__":
    main()
