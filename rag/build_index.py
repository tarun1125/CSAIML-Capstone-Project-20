import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HERE = Path(__file__).parent


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(texts: list[str], tokenizer, model) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    vecs = mean_pool(out.last_hidden_state, enc["attention_mask"])
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
    return vecs.numpy().astype("float32")  # FAISS requires float32


def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


if __name__ == "__main__":
    cards = json.loads((HERE / "schema_cards.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

    vecs = embed([c["text"] for c in cards], tokenizer, model)
    index = build_faiss_index(vecs)
    faiss.write_index(index, str(HERE / "schema.index"))
    (HERE / "schema_card_ids.json").write_text(json.dumps([c["id"] for c in cards]))

    print(f"embedded {len(cards)} cards -> schema.index {vecs.shape}")
