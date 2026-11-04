"""Contextual Retrieval: an LLM writes 1-2 sentences placing each chunk in its article"""

from __future__ import annotations

import argparse
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from .chunking import Chunk, chunk_corpus
from .data import Article, load_corpus

MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"
CACHE = (
    Path(__file__).resolve().parent.parent
    / "indexes"
    / "context"
    / f"{MODEL}__{PROMPT_VERSION}.jsonl"
)

# for usage report.
INPUT_PRICE = 0.10
CACHE_WRITE_X = 1.25
CACHE_READ_X = 0.10
OUTPUT_PRICE = 0.50

MAX_ATTEMPTS = 3
MAX_CONTEXT_WORDS = 80

PROMPT = (
    "Here is the chunk we want to situate within the whole article above.\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Write a short, succinct context (1-2 sentences) that situates this chunk "
    "within the article, to improve search retrieval of the chunk. "
    "Answer only with the context and nothing else."
)


class BadContext(ValueError):
    """this is only for the bad context or broken or empty response."""


def situate(client: anthropic.Anthropic, article: Article, chunk: Chunk) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        thinking={
            "type": "disabled",
        },
        output_config={
            "effort": "low",
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<article>\n{article.text('html')}\n</article>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": PROMPT.format(chunk=chunk.text),
                    },
                ],
            }
        ],
    )

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason != "end_turn":
        raise BadContext(f"stopped early: {response.stop_reason}")
    if not text:
        raise BadContext("empty answer")
    if len(text.split()) > MAX_CONTEXT_WORDS:
        raise BadContext("too long")

    u = response.usage
    return {
        "context": text,
        "usage": {
            "input": u.input_tokens,  # uncached input only
            "output": u.output_tokens,
            "cache_write": u.cache_creation_input_tokens or 0,
            "cache_read": u.cache_read_input_tokens or 0,
        },
    }


def load_cache(path: Path | None = None) -> dict[str, dict]:
    """chunk_id -> saved record, for every chunk already processed."""
    path = path or CACHE
    if not path.exists():
        return {}

    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["chunk_id"]] = r

    return records


def contextualise(
    chunks: list[Chunk], articles: dict[str, Article], workers: int = 8
) -> dict[str, str]:
    """A context for every chunk. API calls only for the non-cached ones"""
    cached = load_cache()
    todo: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.id not in cached:
            todo[c.article_id].append(c)

    if todo:
        client = anthropic.Anthropic(max_retries=5)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        with (
            CACHE.open("a", encoding="utf-8") as out,
            ThreadPoolExecutor(workers) as pool,
        ):
            futures = [
                pool.submit(_one_article, client, articles[a], cs, out, lock, cached)
                for a, cs in todo.items()
            ]
            for t in tqdm(as_completed(futures), total=len(futures), desc="articles"):
                t.result()

    return {c.id: cached[c.id]["context"] for c in chunks if c.id in cached}


# helper function for situate with retries.
def _situate_with_retries(client, article, chunk) -> dict | None:
    """A validated result, or None if this chunk could not be done."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return situate(client, article, chunk)
        except BadContext as e:
            print(f"{chunk.id}: attempt {attempt} / {MAX_ATTEMPTS}: {e}")
        except anthropic.APIError as e:
            print(f"{chunk.id}: API error, giving up on it: {e}")
            return None  # the SDK already retried 5 times
    return None


## sequential, first call is writing the prefix of the cache and then the rest is reading it.
def _one_article(client, article, chunks, out, lock, done) -> None:
    for c in chunks:
        r = _situate_with_retries(client, article, c)
        if r is None:
            continue
        record = {"chunk_id": c.id, "model": MODEL, "prompt": PROMPT_VERSION, **r}
        with lock:
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[c.id] = record


def with_context(chunk: Chunk, contexts: dict[str, str]) -> str:
    """The text to embed and BM25-index. Raises if a chunk has no context,
    so a half-finished run can never silently become an index."""
    return f"{contexts[chunk.id]}\n\n{chunk.text}"


def usage_report(records: list[dict], scale: float = 1.0) -> str:
    t: dict[str, int] = defaultdict(int)
    for r in records:
        for k, v in r["usage"].items():
            t[k] += v

    input_usd = (
        (t["input"] + CACHE_WRITE_X * t["cache_write"] + CACHE_READ_X * t["cache_read"])
        * INPUT_PRICE
        / 1e6
    )
    output_usd = None if OUTPUT_PRICE is None else t["output"] * OUTPUT_PRICE / 1e6
    prompt_tokens = t["input"] + t["cache_read"] + t["cache_write"]
    lines = [
        f"calls: {len(records):,}",
        (
            f"tokens: input {t['input']:,} | cache write {t['cache_write']:,} | "
            f"cache read {t['cache_read']:,} | output {t['output']:,}"
        ),
        (
            "share of prompt tokens read from cache: "
            f"{t['cache_read'] / max(prompt_tokens, 1):.1%}"
        ),
        f"input-side cost: ${input_usd:.4f}",
    ]

    if output_usd is None:
        lines.append("output cost: set OUTPUT_PRICE to compute")
    else:
        lines.append(f"total cost: ${input_usd + output_usd:.4f}")
        if scale != 1.0:
            total = (input_usd + output_usd) * scale
            lines.append(f"estimated full corpus (x{scale:.1f}): ${total:.2f}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=200, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show", type=int, default=5, help="contexts to print")
    args = ap.parse_args()
    load_dotenv()

    articles = load_corpus()
    all_chunks = chunk_corpus(articles, "section", "html")
    ids = sorted(articles)
    if args.articles:
        ids = random.Random(0).sample(ids, args.articles)

    chosen = set(ids)
    chunks = [c for c in all_chunks if c.article_id in chosen]

    contexts = contextualise(chunks, articles, args.workers)
    missing = [c for c in chunks if c.id not in contexts]
    print(f"{len(missing)} chunks failed and were not cached")
    cache = load_cache()
    records = [cache[c.id] for c in chunks if c.id in cache]
    print(f"\n{len(contexts):,} / {len(chunks):,} chunks have a context\n")
    print(usage_report(records, scale=len(all_chunks) / max(len(chunks), 1)))

    print("\nRead these before trusting the rest:")
    for c in random.Random(1).sample(chunks, min(args.show, len(chunks))):
        print(f"\n[{articles[c.article_id].title}]")
        print(f"  context: {contexts.get(c.id)}")
        print(f"  chunk:   {c.text[:160]!r}")


if __name__ == "__main__":
    main()
