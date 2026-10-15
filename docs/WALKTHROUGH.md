# How last-rag works so far

A plain-language tour of everything built up to the first retrieval results: what
happened, what data we use, and how each file works. Read it top to bottom once, then
keep it open next to the code.

---

## 1. The story so far

1. **Old project (2025 internship).** A company FAQ chatbot with 37 Q&A pairs, a weighted mix
   of TF-IDF + BM25 + embeddings, and GPT-4o. It was never measured. A quick test
   (36 hand-written questions) showed plain embeddings beat the "hybrid" mix. That was
   the first lesson: **measure, don't assume.**
2. **New project (last-rag).** Same idea, done properly. Every RAG technique becomes a
   *toggle*, and every toggle gets measured for quality, speed and cost. Company data
   can't be published, so we needed a public dataset.
3. **Dataset choice: WixQA** (section 2).
4. **Environment:** uv project, PyTorch with CUDA 13.0 so the RTX 3050 does the
   embedding work, and every cache on D:.
5. **Step 1:** load the data, look at it, and find that its main text field is damaged.
   We rebuilt clean text from the HTML (`data.py`, `stats.py`).
6. **Step 2:** build the measuring machine: chunk, embed, index, search, score, save
   (`chunking.py` → `embed.py` → `index.py` → `retrieve.py` → `metrics.py` →
   `evaluate.py`).
7. **First results:** `reports/retrieval_baselines.md`.

---

## 2. The data: WixQA

[Wix](https://www.wix.com) is a website builder. Its help center has thousands of
"how do I…" articles. Wix's AI research team published a snapshot of it, plus real
customer questions, as a benchmark: `Wix/WixQA` on Hugging Face, MIT license.

It lives in four tables (Hugging Face calls them *configs*):

| table | rows | what one row is |
|---|---|---|
| `wix_kb_corpus` | 6,221 | one help article: `id`, `title`, `url`, `contents` (plain text), `html_content`, `article_type` |
| `wixqa_expertwritten` | 200 | a real customer question, an answer written by a support expert, and `article_ids`, the articles needed to answer it |
| `wixqa_simulated` | 200 | a question distilled from a real support chat, plus answer and `article_ids` |
| `wixqa_synthetic` | 6,221 | an LLM-generated question for every article (for tuning; never used for headline numbers) |

The `article_ids` are the **gold labels**. They let us check retrieval automatically:
"did the search return the article an expert said is needed?"

### See it yourself

The dataset downloaded into `D:\coding\.hf-cache` the first time the code ran. To look
at it:

```powershell
cd D:\coding\last-rag
$env:HF_HOME = "D:\coding\.hf-cache"
uv run --no-sync python -c "from lastrag.data import load_corpus, load_qa; q = load_qa('expertwritten')[0]; print(q.question); print('gold:', q.gold); a = load_corpus()[q.gold[0]]; print(a.title, a.url); print(a.markdown[:1500])"
```

That prints one real question, the id of its gold article, and the first part of that
article. You can also browse it online at `huggingface.co/datasets/Wix/WixQA` (the
"Data Studio" tab).

### What we found when we looked (`reports/corpus_stats.md`)

- The `contents` field glues text together: `"invoice.In this article"`,
  `"invoiceEnabling automatic…"`. Headings and list steps lose their separators, which
  hurts keyword search and chunking. The `html_content` field still has the structure,
  so we rebuild clean markdown from it.
- The articles that real questions need are long (median 665 words, versus 222 for the
  corpus as a whole). That's why chunking matters.
- A third of the articles are "feature requests" that real questions almost never need.
  They're a future filter toggle.
- 26% of expert questions need 2–3 articles, so we score "found one" and "found all"
  separately.

---

## 3. The pipeline we built

```
 6,221 articles (Hugging Face)
        │  data.py         rebuild clean markdown from the HTML
        ▼
 Article objects
        │  chunking.py     whole article  OR  split at ## / ### headings
        ▼
 Chunks (6,221 or ~20k)
        │  embed.py        embeddinggemma-2 turns each chunk into 768 numbers (GPU)
        ▼
 Vectors (cached as .npy)
        │  index.py        store chunks + vectors in LanceDB, build a BM25 keyword index
        ▼
 LanceDB tables  (indexes/lancedb/<source>_<chunker>)
        │  retrieve.py     question → top chunks → their articles, best first
        ▼
 Ranked article ids
        │  metrics.py      compare with the gold article_ids → recall, MRR, nDCG
        ▼
 Scores per question
        │  evaluate.py     average over 200 questions, add confidence intervals, save JSON
        ▼
 runs/retrieval/<set>/<config>.json   →   reports/retrieval_baselines.md
```

**Two kinds of search:**
- **BM25 (keyword).** Scores chunks by shared words. Rare words count more, and very
  long chunks are penalised. It's fast (7 ms) but misses paraphrases: "money back" won't
  match "refund".
- **Dense (meaning).** An embedding model turns text into a vector (a list of 768
  numbers) so that texts with similar meaning point in similar directions. The search
  finds the chunk vectors closest to the question vector (cosine similarity). It's
  slower (about 140 ms, mostly embedding the question) but understands paraphrases.

---

## 4. File by file

### `lastrag/data.py`: loading and cleaning

- `Article` and `QA` are **dataclasses**: a quick way to define a "record" type with
  named fields. `frozen=True` makes them read-only, so nothing changes them by accident.
- `load_corpus()` downloads the corpus table (cached after the first time) and turns
  each row into an `Article`. `@lru_cache` means "remember the result". The second call
  returns instantly instead of re-parsing 6,221 articles (about 10 s).
- `load_qa(name)` does the same for a question set. `_parse_ids` exists because
  `article_ids` sometimes arrives as a real list and sometimes as the *string*
  `"['abc...']"`.
- `html_to_markdown` + `_MarkdownBuilder`: Python's built-in `HTMLParser` walks the HTML
  and calls `handle_starttag` / `handle_data` / `handle_endtag` for every piece. We
  build markdown lines as we go:
  - `<h2 level="3">` becomes `## heading`; `level="4"` becomes `### heading`
  - `<li>` inside `<ol>` becomes `1. step`. Wix splits one numbered list around every
    screenshot, so a new `<ol>` that directly follows another continues the count.
  - editor tabs become `#### In Wix Editor:` / `#### In Studio Editor:`
  - tables become `| cell | cell |`; code goes inside fenced code blocks; images are dropped
  - `Article.text("html")` returns this clean version and `Article.text("contents")`
    the original. That's our first toggle.

### `lastrag/stats.py`: looking before building

It counts words, sections, glued words and gold-label shapes, then writes
`reports/corpus_stats.md`. No AI here, just counting. It's how we decided chunk sizes.

### `lastrag/chunking.py`: cutting articles into pieces

- `Chunk` holds `id` (`"<article_id>:<n>"`), `article_id`, `article_type`, `title` and `text`.
- `chunker="article"`: one chunk per article (what the WixQA paper did).
- `chunker="section"`: `split_sections` cuts at lines starting with `## ` or `### `.
  - A section under 40 words (`MIN_WORDS`) is merged into the next one, since a heading
    plus one line is too thin to search well.
  - A section over 400 words (`MAX_WORDS`) is cut at line breaks by `_cap`, and each
    piece repeats the heading so it still says what it's about.
- `_strip_title` removes the title line from the body, because the title is stored and
  embedded separately.

### `lastrag/embed.py`: text to vectors

- `model()` loads embeddinggemma-2 once (`lru_cache` again):
  - **text-only** (`vision_config=None, audio_config=None`): 271M params instead of 740M
  - **bf16** on the GPU. The model card warns that fp16 produces broken (NaN) vectors.
  - `max_seq_length = 2048` tokens. The model doesn't set a limit, and long inputs
    would overflow 4 GB of VRAM.
- The model was trained with **prefixes**, and using them matters:
  - questions are sent as `task: search result | query: <question>` (`prompt_name="SearchQuery"`)
  - chunks are sent as `title: <title> | text: <chunk>` (`doc_text()`)
- `embed_docs` hashes all the input texts into a short key. If
  `indexes/embeddings/<key>.npy` exists, it loads the vectors instead of recomputing
  them. Same texts, same file, so re-runs take seconds instead of minutes.
- `normalize_embeddings=True` makes every vector length 1, so cosine similarity is just a
  dot product.

### `lastrag/index.py`: storing it in LanceDB

LanceDB is an embedded database: a folder on disk, no server. One table per
(text source × chunker), so 4 tables:
- columns: `chunk_id`, `article_id`, `article_type`, `title`, `text`, `search_text`,
  `vector`
- `create_fts_index("search_text")` builds the BM25 keyword index. `search_text` is
  title + text, so keyword search sees the title just as the embedding model does.
- If the table already exists it's opened rather than rebuilt (pass `rebuild=True` to force).

### `lastrag/retrieve.py`: asking questions

- `RetrieverConfig(method, chunker, source)` names a setup, e.g. `dense|section|html`.
- `Retriever.search(question)`:
  - **dense:** embed the question, then `table.search(vector).distance_type("cosine").limit(200)`
  - **bm25:** `table.search(question, query_type="fts").limit(200)`
  - both return up to 200 **chunks**. `dict.fromkeys(...)` keeps the first appearance of
    each article id, in order, which turns "ranked chunks" into "ranked articles" (an
    article ranks by its best chunk).

### `lastrag/metrics.py`: scoring

For one question, with `ranked` = our article ids and `gold` = the expert's:

- **recall_any@k:** 1 if any gold article is in the top k, else 0
- **recall_all@k:** 1 only if *every* gold article is in the top k
- **MRR:** 1 / position of the first gold article (1st = 1.0, 2nd = 0.5, 5th = 0.2, missing = 0)
- **nDCG@10:** like recall, but finding gold at position 1 earns more than at position 9

Averaging the 0s and 1s over 200 questions gives a rate like 0.83.

**Why confidence intervals:** with 200 questions, luck matters. The **bootstrap**
re-samples the 200 questions with replacement 10,000 times and recomputes the average
each time. The middle 95% of those averages is the interval. If two setups' intervals
overlap heavily, we can't claim one is better.

**Paired bootstrap** (`--compare`): for each question, take score(B) − score(A), then
bootstrap those differences. "Paired" means both setups are judged on the same
questions, which is much more sensitive than comparing two separate averages. The
p-value is roughly the chance of seeing this difference if there were no real one.

### `lastrag/evaluate.py`: running everything

- It builds every combination (method × chunker × source = 8) for each question set
  (ExpertWritten and Simulated).
- For each one: run all 200 questions, time each search, score it, then compute averages
  and intervals.
- It saves `runs/retrieval/<set>/<config>.json` with the config, git commit, metrics,
  p50/p95 latency, and every question's top 5 results, so you can inspect any failure.
- `--compare A B` prints the paired difference table.
- **Rule:** the README and the future metrics page only read these JSON files, so no
  number is ever typed by hand.

### `tests/`

`pytest` checks the HTML cleaner (headings, list numbering, tabs, tables), the metrics
(known inputs, known answers) and the chunker (merge small, cap large). Run
`uv run --no-sync pytest` after changing any of these.

---

## 5. The first results, in one paragraph

Meaning search (dense) beats keyword search (BM25) by about 20 points at about 20× the
latency. Cleaning the text and splitting by section adds another 8 points on expert
questions, a statistically real gain (p = 0.005). On simulated questions it's within
noise. Full tables: `reports/retrieval_baselines.md`.

---

## 6. Where things live on disk

| path | what | in git? |
|---|---|---|
| `lastrag/` | the code | yes |
| `tests/` | tests | yes |
| `reports/` | generated reports (public-data numbers) | yes |
| `docs/` | this file | yes |
| `indexes/embeddings/` | cached vectors (.npy) | no, rebuildable |
| `indexes/lancedb/` | LanceDB tables | no, rebuildable |
| `runs/` | raw run JSON | no (could be committed later as evidence) |
| `D:\coding\.hf-cache` | downloaded dataset + model | no, outside the repo |
