"""Load WixQA: the Wix help-center corpus and its three QA sets.

Source: Hugging Face `Wix/WixQA` (MIT, cite "Wix.com AI Research", arXiv 2505.08643).

The corpus `contents` field (what the paper's baselines use) is flattened text where
headings and list items run into each other: "...the invoice.In this article, learn
more about:\nEnabling auto-conversion Manually converting...". We rebuild clean
markdown from `html_content` instead. `Article.text(source)` picks which one, so the
difference is a toggle we can measure.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache, lru_cache
from html.parser import HTMLParser
from typing import Literal

from datasets import load_dataset

REPO = "Wix/WixQA"
QA_SETS = ("expertwritten", "simulated", "synthetic")
TextSource = Literal["contents", "html"]


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    url: str
    type: str  # article | feature_request | known_issue
    contents: str  # original flattened text from the dataset
    markdown: str  # rebuilt from html_content, structure kept

    def text(self, source: TextSource = "html") -> str:
        return self.markdown if source == "html" else self.contents


@dataclass(frozen=True)
class QA:
    question: str
    answer: str  # expert / reference answer, markdown
    gold: tuple[str, ...]  # article ids needed to answer


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, Article]:
    """All 6,221 help-center articles, keyed by id."""
    rows = load_dataset(REPO, "wix_kb_corpus", split="train")
    return {
        r["id"]: Article(
            id=r["id"],
            title=r["title"],
            url=r["url"],
            type=r["article_type"],
            contents=r["contents"],
            markdown=html_to_markdown(r["html_content"], r["title"]),
        )
        for r in rows
    }


@cache
def load_qa(name: str) -> tuple[QA, ...]:
    """One QA set: 'expertwritten' (200), 'simulated' (200) or 'synthetic' (6,221)."""
    if name not in QA_SETS:
        raise ValueError(f"unknown QA set {name!r}, expected one of {QA_SETS}")
    rows = load_dataset(REPO, f"wixqa_{name}", split="train")
    return tuple(
        QA(r["question"], r["answer"], tuple(_parse_ids(r["article_ids"])))
        for r in rows
    )


def _parse_ids(value) -> list[str]:
    # Hub copies have shipped this both as list[str] and as a stringified list.
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


# --- html_content -> markdown -------------------------------------------------


def html_to_markdown(html: str, title: str) -> str:
    parser = _MarkdownBuilder()
    parser.feed(html)
    parser.close()
    lines = [f"# {title}"]
    for line in parser.finish():
        if line.startswith("#"):
            lines.append("")  # blank line before headings
        lines.append(line)
    return "\n".join(lines).strip()


class _MarkdownBuilder(HTMLParser):
    """Turns Wix help-center HTML into markdown.

    Wix markup: <h2 level="3|4"> section headings; collapsibles ("Show me how") as
    <h4 class="collapsible-title">; editor tabs as a list of tab names followed by
    the tab bodies in the same order; tip boxes as <div class="info-title">.
    Images, videos, iframes and divider lines carry no text and are dropped.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buf: list[str] = []  # text of the line being built
        self.prefix = ""  # markdown prefix for that line ("## ", "- ", ...)
        self.divs: list[
            list[str]
        ] = []  # classes of each open <div>, to know which one closes
        self.lists: list[list] = []  # open lists as [ordered?, item counter]
        # Wix splits one numbered list around each screenshot; an <ol> that follows
        # another with nothing but images in between continues its numbering.
        self.last_ol: tuple[int, int] | None = (
            None  # (len(lines) when it closed, last number)
        )
        self.tab_names: list[str] = []
        self.tab_index = 0
        self.tab_name_buf: list[str] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.in_pre = False

    # line handling
    def flush(self) -> None:
        text = " ".join("".join(self.buf).split())
        if text:
            self.lines.append(self.prefix + text)
            self.prefix = ""
        self.buf = []

    def start_line(self, prefix: str) -> None:
        self.flush()
        self.prefix = prefix

    def finish(self) -> list[str]:
        self.flush()
        return self.lines

    # parser callbacks
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div":
            self.divs.append(cls)
            if "tabs-wrapper" in cls:
                self.tab_names, self.tab_index = [], 0
            elif "tab-heading" in cls:
                self.tab_name_buf = []
            elif "tab-content" in cls:
                name = (
                    self.tab_names[self.tab_index]
                    if self.tab_index < len(self.tab_names)
                    else ""
                )
                self.tab_index += 1
                self.start_line("#### ")
                self.buf.append(f"In {name}:" if name else "")
            elif self.cell is None:
                self.flush()
        elif tag == "h2":
            self.start_line("### " if a.get("level") == "4" else "## ")
        elif tag == "h4":
            self.start_line("#### ")
        elif tag in ("ul", "ol"):
            self.flush()
            counter = 0
            if tag == "ol" and a.get("start", "").isdigit():
                counter = int(a["start"]) - 1
            elif (
                tag == "ol"
                and not self.lists
                and self.last_ol
                and self.last_ol[0] == len(self.lines)
            ):
                counter = self.last_ol[1]
            self.lists.append([tag == "ol", counter])
        elif tag == "li":
            if self.lists:
                lst = self.lists[-1]
                lst[1] += 1
                marker = f"{lst[1]}. " if lst[0] else "- "
                self.start_line("   " * (len(self.lists) - 1) + marker)
            else:
                self.start_line("- ")
        elif tag == "tr":
            self.flush()
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "pre":
            self.flush()
            self.lines.append("```")
            self.in_pre = True
        elif tag == "br":
            if self.cell is not None:
                self.cell.append(" ")
            elif not self.in_pre:
                self.flush()
        elif tag == "p" and self.cell is None:
            self.flush()

    def handle_endtag(self, tag):
        if tag == "div":
            cls = self.divs.pop() if self.divs else []
            if "tab-heading" in cls and self.tab_name_buf is not None:
                self.tab_names.append(" ".join("".join(self.tab_name_buf).split()))
                self.tab_name_buf = None
            elif self.cell is None:
                self.flush()
        elif tag in ("h2", "h4", "li", "p"):
            if self.cell is None:
                self.flush()
        elif tag in ("ul", "ol"):
            self.flush()
            if self.lists:
                ordered, count = self.lists.pop()
                if ordered and not self.lists:
                    self.last_ol = (len(self.lines), count)
        elif tag in ("td", "th"):
            if self.row is not None and self.cell is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr":
            if self.row and any(self.row):
                self.lines.append("| " + " | ".join(self.row) + " |")
            self.row = None
        elif tag == "pre":
            self.lines.append("```")
            self.in_pre = False

    def handle_data(self, data):
        if self.tab_name_buf is not None:
            self.tab_name_buf.append(data)
        elif self.cell is not None:
            self.cell.append(data)
        elif self.in_pre:
            self.lines.extend(
                line.rstrip() for line in data.splitlines() if line.strip()
            )
        else:
            self.buf.append(data)
