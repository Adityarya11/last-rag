"""LanceDB tables: one per (text source, chunker), holding vectors and a BM25 index."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from .chunking import Chunk, Chunker, chunk_corpus
from .contextual import load_cache, with_context
from .data import TextSource, load_corpus
from .embed import doc_text, embed_docs

DB_PATH = Path(__file__).resolve().parent.parent / "indexes" / "lancedb"


def table_name(
    source: TextSource, chunker: Chunker, contextual: bool = False, title: bool = True
) -> str:
    name = f"{source}_{chunker}"
    if contextual:
        name += "_ctx"
    if not title:
        name += "_notitle"
    return name


def indexed_texts(chunks: list[Chunk], contexts: dict[str, str] | None) -> list[str]:
    """The text each chunk is embedded and BM25-indexed on.

    Without contexts it is the chunk itself. With them, the generated note goes in
    front; a chunk with no context raises KeyError instead of being indexed bare.
    """
    if contexts is None:
        return [c.text for c in chunks]
    return [with_context(c, contexts) for c in chunks]


def index_fields(
    chunks: list[Chunk], texts: list[str], title: bool = True
) -> tuple[list[str], list[str]]:
    """(text to embed, text for BM25), with or without the article title in front."""
    heading = [c.title if title else "" for c in chunks]
    embedded = [doc_text(h, t) for h, t in zip(heading, texts)]
    searchable = [f"{h}\n{t}".strip() for h, t in zip(heading, texts)]
    return embedded, searchable


def open_table(
    source: TextSource,
    chunker: Chunker,
    rebuild: bool = False,
    contextual: bool = False,
    title: bool = True,
):
    db = lancedb.connect(DB_PATH)
    name = table_name(source, chunker, contextual, title)
    if not rebuild and name in db.list_tables().tables:
        return db.open_table(name)

    chunks = chunk_corpus(load_corpus(), chunker, source)
    contexts = (
        {chunk_id: r["context"] for chunk_id, r in load_cache().items()}
        if contextual
        else None
    )
    texts = indexed_texts(chunks, contexts)
    print(f"building {name}: {len(chunks):,} chunks")
    embedded, searchable = index_fields(chunks, texts, title)
    vecs = embed_docs(embedded)
    data = pa.table(
        {
            "chunk_id": [c.id for c in chunks],
            "article_id": [c.article_id for c in chunks],
            "article_type": [c.article_type for c in chunks],
            "title": [c.title for c in chunks],
            "text": [c.text for c in chunks],
            "search_text": searchable,
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vecs.ravel(), type=pa.float32()), vecs.shape[1]
            ),
        }
    )
    table = db.create_table(name, data, mode="overwrite")
    # LanceDB native BM25: lowercase, English stemming, stop words removed
    table.create_fts_index("search_text", replace=True)
    return table
