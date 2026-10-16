"""Why did a reranker not help? Look inside the candidates instead of at averages.

    uv run python -m lastrag.diagnose --reranker minilm --qa expertwritten

For each question: take dense's top chunks, check whether the gold article is among
them (the reranker's ceiling), re-sort the first 30 with the cross-encoder, and
compare the gold article's rank before and after.
"""

from __future__ import annotations

import argparse
from collections import Counter

from .data import load_qa
from .rerank import model, rerank
from .retrieve import RERANK_CANDIDATES, Retriever, RetrieverConfig, _articles


def first_rank(ranked: list[str], gold: tuple[str, ...]) -> int | None:
    for i, a in enumerate(ranked, start=1):
        if a in gold:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", default="minilm")
    ap.add_argument("--qa", default="expertwritten")
    ap.add_argument("--show", type=int, default=5, help="worst failures to print")
    args = ap.parse_args()

    retriever = Retriever(RetrieverConfig())
    tok = model(args.reranker).tokenizer
    queries = load_qa(args.qa)
    n = len(queries)

    in_cands = Counter()  # gold inside the first K chunks
    outcome = Counter()  # rescued / buried / same, for gold inside the candidates
    top5 = Counter()
    trunc_gold = []  # gold chunk's token count when paired with the question
    fails = []

    for q in queries:
        rows = retriever._dense_rows(q.question, 100)
        for k in (10, 30, 50, 100):
            if first_rank(_articles(rows[:k]), q.gold):
                in_cands[k] += 1

        cands = rows[:RERANK_CANDIDATES]
        before = _articles(cands)
        after = _articles(rerank(args.reranker, q.question, cands))
        b, a = first_rank(before, q.gold), first_rank(after, q.gold)
        top5["dense"] += bool(b and b <= 5)
        top5["reranked"] += bool(a and a <= 5)

        if b is None:
            outcome["gold not in candidates"] += 1
            continue
        if a < b:
            outcome["rescued (rank improved)"] += 1
        elif a > b:
            outcome["buried (rank got worse)"] += 1
            fails.append((a - b, q, before, after, cands))
        else:
            outcome["unchanged"] += 1

        gold_chunk = next(r for r in cands if r["article_id"] in q.gold)
        n_tok = len(tok(q.question, f"{gold_chunk['title']}\n{gold_chunk['text']}")["input_ids"])
        trunc_gold.append(n_tok)

    print(f"\n## {args.qa} ({n} questions), reranker = {args.reranker}\n")
    print("Gold article inside dense's top-K chunks (the reranker's ceiling):")
    for k in (10, 30, 50, 100):
        print(f"  K={k:<3} {in_cands[k] / n:.3f}")
    print(f"\nTop-5 hit: dense {top5['dense'] / n:.3f} -> reranked {top5['reranked'] / n:.3f}")
    print("\nWhat the reranker did to the gold article's rank:")
    for k, v in outcome.most_common():
        print(f"  {k:<28} {v:>4}")
    over = sum(t > 512 for t in trunc_gold)
    print(f"\nGold chunks cut off at 512 tokens: {over}/{len(trunc_gold)} (max {max(trunc_gold)})")

    titles = {}
    for r in (r for _, _, _, _, cands in fails for r in cands):
        titles[r["article_id"]] = r["title"]
    print(f"\nWorst {args.show} cases where the reranker buried the gold article:")
    for drop, q, before, after, _ in sorted(fails, key=lambda f: -f[0])[: args.show]:
        print(f"\n  Q: {q.question[:150]}")
        print(f"  gold: {[titles.get(g, g[:8]) for g in q.gold]}  rank {first_rank(before, q.gold)} -> {first_rank(after, q.gold)}")
        print(f"  reranker's #1: {titles[after[0]]}")


if __name__ == "__main__":
    main()
