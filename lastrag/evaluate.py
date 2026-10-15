"""Run retrieval configs over WixQA query sets and save every run.

    uv run python -m lastrag.evaluate                              # default grid
    uv run python -m lastrag.evaluate --method dense --chunker section --source html
    uv run python -m lastrag.evaluate --compare "bm25|article|contents" "dense|section|html"

Each (config, query set) is saved to runs/retrieval/<set>/<config>.json with the
config, git commit, metric means + 95% CIs, and per-query ranks and latencies.
The metrics page and the README read ONLY from these files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .chunking import CHUNKERS
from .data import load_qa
from .metrics import KS, paired_bootstrap, per_query, summarize
from .retrieve import METHODS, Retriever, RetrieverConfig

RUNS = Path(__file__).resolve().parent.parent / "runs" / "retrieval"
HEADLINE = ["recall_any@5", "recall_all@5", "recall_any@20", "mrr", "ndcg@10"]
EVAL_SETS = ("expertwritten", "simulated")  # synthetic is for tuning, not headlines


def run_path(qa_set: str, config_name: str) -> Path:
    return RUNS / qa_set / (config_name.replace("|", "__") + ".json")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def evaluate(config: RetrieverConfig, qa_set: str) -> dict:
    retriever = Retriever(config)
    queries = load_qa(qa_set)
    retriever.search("warm up")  # model load and index open stay out of latency
    rows = []
    for q in queries:
        t = time.perf_counter()
        ranked = retriever.search(q.question)
        ms = (time.perf_counter() - t) * 1000
        m = per_query(ranked, q.gold)
        rows.append({"question": q.question, "gold": list(q.gold), "top5": ranked[:5], "ms": ms, **m})
    names = [f"recall_any@{k}" for k in KS] + [f"recall_all@{k}" for k in KS] + ["mrr", "ndcg@10"]
    lat = sorted(r["ms"] for r in rows)
    run = {
        "config": config.as_dict(),
        "config_name": config.name,
        "qa_set": qa_set,
        "n_queries": len(rows),
        "git": git_sha(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": summarize(rows, names),
        "latency_ms": {
            "p50": statistics.median(lat),
            "p95": lat[int(0.95 * (len(lat) - 1))],
        },
        "queries": [
            {**r, "first_gold_rank": None if r["first_gold_rank"] == float("inf") else r["first_gold_rank"]}
            for r in rows
        ],
    }
    path = run_path(qa_set, config.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=1), encoding="utf-8")
    return run


def table(runs: list[dict]) -> str:
    head = "| config | " + " | ".join(HEADLINE) + " | p50 ms |"
    lines = [head, "|" + "---|" * (len(HEADLINE) + 2)]
    for r in runs:
        cells = [
            f"{r['metrics'][m]['mean']:.3f} [{r['metrics'][m]['lo']:.2f}, {r['metrics'][m]['hi']:.2f}]"
            for m in HEADLINE
        ]
        lines.append(f"| {r['config_name']} | " + " | ".join(cells) + f" | {r['latency_ms']['p50']:.0f} |")
    return "\n".join(lines)


def compare(a_name: str, b_name: str, qa_set: str) -> str:
    a = json.loads(run_path(qa_set, a_name).read_text(encoding="utf-8"))
    b = json.loads(run_path(qa_set, b_name).read_text(encoding="utf-8"))
    out = [f"{qa_set}: {b_name}  minus  {a_name}", "| metric | diff | 95% CI | p |", "|---|---|---|---|"]
    for m in HEADLINE:
        d = paired_bootstrap([q[m] for q in a["queries"]], [q[m] for q in b["queries"]])
        out.append(f"| {m} | {d['diff']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['p']:.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs="+", default=list(METHODS))
    ap.add_argument("--chunker", nargs="+", default=list(CHUNKERS))
    ap.add_argument("--source", nargs="+", default=["contents", "html"])
    ap.add_argument("--qa", nargs="+", default=list(EVAL_SETS))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        for qa_set in args.qa:
            print(compare(*args.compare, qa_set), "\n")
        return

    for qa_set in args.qa:
        runs = [
            evaluate(RetrieverConfig(method, chunker, source), qa_set)
            for method, chunker, source in itertools.product(args.method, args.chunker, args.source)
        ]
        print(f"\n## {qa_set} ({runs[0]['n_queries']} queries)\n")
        print(table(runs))


if __name__ == "__main__":
    main()
