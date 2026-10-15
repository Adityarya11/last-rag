"""Retrievers: query in, ranked article ids out.

Search runs over chunks; a chunk's article takes the rank of its best chunk, so a
query is scored on which ARTICLES it found (that is what WixQA's gold labels name).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Literal

from .chunking import Chunker
from .data import TextSource
from .embed import embed_queries
from .index import open_table

Method = Literal["bm25", "dense", "rrf"]
METHODS: tuple[Method, ...] = ("bm25", "dense", "rrf")

FETCH_CHUNKS = (
    200  # enough chunks to fill the top 20 articles even when several share one
)


@dataclass(frozen=True)
class RetrieverConfig:
    method: Method = "dense"
    chunker: Chunker = "section"
    source: TextSource = "html"

    @property
    def name(self) -> str:
        return f"{self.method}|{self.chunker}|{self.source}"

    def as_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config

    @cached_property
    def table(self):
        return open_table(self.config.source, self.config.chunker)

    def search(self, query: str) -> list[str]:
        """Article ids, best first."""
        if self.config.method == "dense":
            return self._dense(query)
        if self.config.method == "bm25":
            return self._bm25(query)
        if self.config.method == "rrf":
            return rrf([self._bm25(query), self._dense(query)])
        raise ValueError(f"unknown method: {self.config.method}")

    def _dense(self, query: str) -> list[str]:
        q = embed_queries([query])[0]
        rows = (
            self.table.search(q, vector_column_name="vector")
            .distance_type("cosine")
            .limit(FETCH_CHUNKS)
            .select(["article_id"])
            .to_list()
        )
        return _articles(rows)

    def _bm25(self, query: str) -> list[str]:
        rows = (
            self.table.search(query, query_type="fts", fts_columns="search_text")
            .limit(FETCH_CHUNKS)
            .select(["article_id"])
            .to_list()
        )
        return _articles(rows)


def _articles(rows: list[dict]) -> list[str]:
    """Ranked chunks -> ranked article ids (an article keeps its best chunk's rank)."""
    return list(dict.fromkeys(r["article_id"] for r in rows))


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge several rankings, best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, article_id in enumerate(ranking, start=1):
            scores[article_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
