"""LanceDB tables: one per (text source, chunker), holding vectors and a BM25 index."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from .chunking import Chunker, chunk_corpus
from .data import TextSource, load_corpus
from .embed import doc_text, embed_docs

DB_PATH = Path(__file__).resolve().parent.parent / "indexes" / "lancedb"


def table_name(source: TextSource, chunker: Chunker) -> str:
    return f"{source}_{chunker}"


def open_table(source: TextSource, chunker: Chunker, rebuild: bool = False):
    db = lancedb.connect(DB_PATH)
    name = table_name(source, chunker)
    if not rebuild and name in db.list_tables().tables:
        return db.open_table(name)

    chunks = chunk_corpus(load_corpus(), chunker, source)
    print(f"building {name}: {len(chunks):,} chunks")
    vecs = embed_docs([doc_text(c.title, c.text) for c in chunks])
    data = pa.table(
        {
            "chunk_id": [c.id for c in chunks],
            "article_id": [c.article_id for c in chunks],
            "article_type": [c.article_type for c in chunks],
            "title": [c.title for c in chunks],
            "text": [c.text for c in chunks],
            "search_text": [f"{c.title}\n{c.text}" for c in chunks],
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vecs.ravel(), type=pa.float32()), vecs.shape[1]
            ),
        }
    )
    table = db.create_table(name, data, mode="overwrite")
    # LanceDB native BM25: lowercase, English stemming, stop words removed
    table.create_fts_index("search_text", replace=True)
    return table
