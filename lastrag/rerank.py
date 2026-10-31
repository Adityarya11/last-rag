"""Cross-encoder rerankers: for scoring (question, chunk) pairs read together."""

from __future__ import annotations

from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",  # ~90 MB, web-search trained
    "bge": "BAAI/bge-reranker-v2-m3",  # ~2.3 GB download, ~1.1 GB VRAM in bf16
}


@lru_cache
def model(name: str) -> CrossEncoder:
    return CrossEncoder(
        MODELS[name],
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=512,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def rerank(name: str, query: str, rows: list[dict]) -> list[dict]:
    """Rows (each with 'title' and 'text'), most relevant first."""
    pairs = [(query, f"{r['title']}\n{r['text']}") for r in rows]
    scores = model(name).predict(pairs, batch_size=16)
    ranked = sorted(zip(scores, rows), key=lambda p: p[0], reverse=True)
    return [row for _, row in ranked]
