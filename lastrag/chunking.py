"""Split articles into the units we index and retrieve.

article  - one chunk per article (the WixQA paper's setup)
section  - split at ## / ### headings; tiny sections merge forward, huge ones
           split at line boundaries so no chunk exceeds MAX_WORDS
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .data import Article, TextSource

Chunker = Literal["article", "section"]
CHUNKERS: tuple[Chunker, ...] = ("article", "section")

MIN_WORDS = 40  # a heading plus one sentence is too little to embed on its own
MAX_WORDS = 400  # roughly 520 tokens; keeps the chunk about one topic


@dataclass(frozen=True)
class Chunk:
    id: str  # "<article_id>:<n>"
    article_id: str
    article_type: str
    title: str
    text: str


def chunk_corpus(
    articles: dict[str, Article], chunker: Chunker, source: TextSource
) -> list[Chunk]:
    chunks = []
    for a in articles.values():
        body = _strip_title(a.text(source), a.title)
        pieces = [body] if chunker == "article" else split_sections(body)
        chunks += [
            Chunk(f"{a.id}:{i}", a.id, a.type, a.title, text)
            for i, text in enumerate(pieces)
        ]
    return chunks


def _strip_title(text: str, title: str) -> str:
    """Both text sources start with the title line; the title is stored separately."""
    first, _, rest = text.partition("\n")
    return rest.strip() if first.lstrip("# ").strip() == title.strip() else text


def split_sections(body: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n+(?=#{2,3} )", body) if p.strip()]
    merged: list[str] = []
    carry = ""
    for p in parts:
        p = f"{carry}\n\n{p}" if carry else p
        if len(p.split()) < MIN_WORDS:
            carry = p
        else:
            merged.append(p)
            carry = ""
    if carry:  # a short tail joins the previous chunk
        if merged:
            merged[-1] += "\n\n" + carry
        else:
            merged.append(carry)
    return [piece for m in merged for piece in _cap(m)] or [body]


def _cap(text: str) -> list[str]:
    """Split an oversized section at line boundaries, repeating its heading."""
    if len(text.split()) <= MAX_WORDS:
        return [text]
    lines = text.split("\n")
    heading = lines[0] if lines[0].startswith("#") else ""
    out, cur, n = [], [], 0
    for line in lines:
        w = len(line.split())
        if cur and n + w > MAX_WORDS:
            out.append("\n".join(cur))
            cur, n = ([heading] if heading else []), len(heading.split())
        cur.append(line)
        n += w
    if cur:
        out.append("\n".join(cur))
    return out
