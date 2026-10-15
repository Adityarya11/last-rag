"""embeddinggemma-2 (text-only), with document embeddings cached on disk.

Model card rules followed here:
  - bf16 or fp32, never fp16 (fp16 silently returns NaN / degraded vectors)
  - queries:   "task: search result | query: {q}"   (prompt "SearchQuery")
  - documents: "title: {title} | text: {body}"     (formatted by hand)
  - Matryoshka truncation to 512/256/128 must be re-normalised (encode() does it)
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = "google/embeddinggemma-2"
MAX_TOKENS = 2048  # the model allows 8k, but for 4 GB of VRAM its optimal
CACHE = Path(__file__).resolve().parent.parent / "indexes" / "embeddings"


@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    m = SentenceTransformer(
        MODEL,
        device="cuda" if cuda else "cpu",
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    m.max_seq_length = MAX_TOKENS
    return m


def doc_text(title: str, body: str) -> str:
    return f"title: {title or 'none'} | text: {body}"


def embed_queries(
    queries: list[str], prompt: str = "SearchQuery", dim: int = 768
) -> np.ndarray:
    return (
        model()
        .encode(
            queries,
            prompt_name=prompt,
            truncate_dim=dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )


def embed_docs(texts: list[str], dim: int = 768, batch_size: int = 8) -> np.ndarray:
    """Embeds documents, reusing a cached .npy when the exact same texts were seen before."""
    key = hashlib.sha256(
        "\x00".join([MODEL, str(MAX_TOKENS), *texts]).encode()
    ).hexdigest()[:16]
    path = CACHE / f"{key}.npy"
    if path.exists():
        full = np.load(path)
    else:
        full = (
            model()
            .encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            .astype(np.float32)
        )
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, full)
    if dim == full.shape[1]:
        return full
    cut = full[:, :dim]
    return cut / np.linalg.norm(cut, axis=1, keepdims=True)
