# Shared embedding helper for the RAG pipeline.
#
# Deliberately kept identical in approach to the earlier rag/build_index.py
# (mean-pooled all-MiniLM-L6-v2 via raw transformers, not the higher-level
# sentence-transformers wrapper) -- the repo already depends on
# transformers/torch for the Qwen arm, so this adds zero new dependencies
# for anyone running the pipeline locally. Both build_retrieval_index.py
# (embeds the 100-example few-shot pool once) and build_prompts.py (embeds
# each new test question at retrieval time) import embed() from HERE so a
# query vector always lands in the exact same space as the indexed
# few-shot vectors -- two slightly-different embedding implementations
# would silently misalign the space and make FAISS's nearest-neighbor
# search meaningless without ever raising an error.

import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

log = logging.getLogger("rag.embed_utils")

_tokenizer = None
_model = None


def get_embedder():
    """Lazy-loads the tokenizer/model once per process. This is a small
    (~90MB, 384-dim) CPU-friendly model -- no GPU required, safe to run
    locally in VS Code, not just on Colab."""
    global _tokenizer, _model
    if _tokenizer is None:
        log.info("Loading embedder %s (first call only)...", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).eval()
        log.info("Embedder ready.")
    return _tokenizer, _model


def _mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(texts: list[str]) -> np.ndarray:
    """texts -> L2-normalized float32 (n, 384) array, ready for FAISS
    IndexFlatIP (inner product on L2-normalized vectors == cosine
    similarity)."""
    tokenizer, model = get_embedder()
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
    return vecs.numpy().astype("float32")
