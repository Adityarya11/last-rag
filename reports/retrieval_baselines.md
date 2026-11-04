# Retrieval baselines on WixQA

Generated from `runs/retrieval/` (git f52086b-dirty, 2026-10-15). Every number below comes from a saved run.

**Setup:** embeddinggemma-2 (text-only, bf16, 768-d, max 2,048 tokens) on an RTX 3050 Laptop (4 GB); LanceDB brute-force cosine search; LanceDB native BM25 (English stemming, stop words removed) over title + text. Articles are ranked by their best chunk. Brackets are 95% bootstrap intervals; latency is p50 per query, including query embedding.

Config names are `method|chunker|text source`. `article` = one chunk per article (the paper's setup), `section` = split at headings. `contents` = the dataset's flattened text, `html` = markdown rebuilt from the HTML.

## Headline

| setup | any@5 (ExpertWritten) | time per query |
|---|---|---|
| keyword search (BM25), paper-style | 0.555 | 7 ms |
| meaning search (dense), paper-style | 0.750 | 131 ms |
| dense + sections + cleaned text | 0.830 | 148 ms |

- Dense beats keyword by about 20 points here, at about 20× the latency.
- Splitting into sections and cleaning the text added 8 points over the paper-style dense setup. The paired test says this is real, not luck (p = 0.005).
- On the Simulated set the same change gave +2.5 points, which is within noise (the confidence interval crosses zero). It depends on the queries, which is the point of measuring.

## expertwritten (200 queries)

| config | recall_any@5 | recall_all@5 | recall_any@20 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|
| bm25|article|contents | 0.555 [0.48–0.62] | 0.425 [0.36–0.49] | 0.765 [0.70–0.82] | 0.401 [0.35–0.46] | 0.429 [0.38–0.48] | 7 |
| bm25|article|html | 0.540 [0.47–0.61] | 0.405 [0.34–0.47] | 0.745 [0.69–0.81] | 0.387 [0.33–0.44] | 0.419 [0.37–0.47] | 7 |
| bm25|section|contents | 0.560 [0.49–0.63] | 0.430 [0.36–0.50] | 0.765 [0.70–0.82] | 0.398 [0.35–0.45] | 0.428 [0.38–0.48] | 8 |
| bm25|section|html | 0.595 [0.53–0.66] | 0.465 [0.40–0.54] | 0.805 [0.75–0.86] | 0.406 [0.36–0.46] | 0.439 [0.39–0.49] | 7 |
| dense|article|contents | 0.750 [0.69–0.81] | 0.635 [0.56–0.70] | 0.925 [0.89–0.96] | 0.535 [0.48–0.59] | 0.572 [0.52–0.62] | 131 |
| dense|article|html | 0.755 [0.69–0.81] | 0.635 [0.57–0.70] | 0.900 [0.85–0.94] | 0.557 [0.50–0.61] | 0.594 [0.55–0.64] | 137 |
| dense|section|contents | 0.745 [0.69–0.81] | 0.615 [0.55–0.68] | 0.945 [0.91–0.97] | 0.545 [0.49–0.60] | 0.584 [0.54–0.63] | 136 |
| dense|section|html | 0.830 [0.78–0.88] | 0.705 [0.64–0.77] | 0.955 [0.93–0.98] | 0.605 [0.55–0.66] | 0.647 [0.60–0.69] | 148 |

Paired difference, `dense|section|html` minus `dense|article|contents`:

| metric | diff | 95% CI | p |
|---|---|---|---|
| recall_any@5 | +0.080 | [+0.025, +0.135] | 0.005 |
| recall_all@5 | +0.070 | [+0.015, +0.125] | 0.014 |
| recall_any@20 | +0.030 | [+0.000, +0.060] | 0.069 |
| mrr | +0.070 | [+0.026, +0.117] | 0.002 |
| ndcg@10 | +0.074 | [+0.038, +0.112] | 0.000 | 

## simulated (200 queries)

| config | recall_any@5 | recall_all@5 | recall_any@20 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|
| bm25|article|contents | 0.455 [0.39–0.53] | 0.390 [0.33–0.46] | 0.760 [0.70–0.82] | 0.325 [0.28–0.38] | 0.356 [0.31–0.41] | 7 |
| bm25|article|html | 0.455 [0.39–0.53] | 0.390 [0.33–0.46] | 0.765 [0.70–0.82] | 0.321 [0.27–0.37] | 0.357 [0.31–0.41] | 7 |
| bm25|section|contents | 0.495 [0.42–0.56] | 0.410 [0.34–0.48] | 0.760 [0.70–0.81] | 0.352 [0.30–0.41] | 0.382 [0.33–0.43] | 7 |
| bm25|section|html | 0.540 [0.47–0.61] | 0.465 [0.40–0.54] | 0.760 [0.70–0.82] | 0.374 [0.32–0.43] | 0.406 [0.35–0.46] | 6 |
| dense|article|contents | 0.675 [0.61–0.74] | 0.595 [0.53–0.66] | 0.915 [0.88–0.95] | 0.461 [0.41–0.51] | 0.523 [0.48–0.57] | 122 |
| dense|article|html | 0.705 [0.64–0.77] | 0.640 [0.57–0.70] | 0.915 [0.88–0.95] | 0.515 [0.46–0.57] | 0.565 [0.51–0.61] | 136 |
| dense|section|contents | 0.680 [0.61–0.74] | 0.605 [0.54–0.68] | 0.910 [0.87–0.95] | 0.501 [0.45–0.56] | 0.544 [0.49–0.60] | 133 |
| dense|section|html | 0.700 [0.64–0.76] | 0.615 [0.55–0.68] | 0.915 [0.88–0.95] | 0.503 [0.45–0.56] | 0.550 [0.50–0.60] | 148 |

Paired difference, `dense|section|html` minus `dense|article|contents`:

| metric | diff | 95% CI | p |
|---|---|---|---|
| recall_any@5 | +0.025 | [-0.025, +0.075] | 0.362 |
| recall_all@5 | +0.020 | [-0.030, +0.070] | 0.501 |
| recall_any@20 | +0.000 | [-0.035, +0.035] | 1.000 |
| mrr | +0.042 | [-0.002, +0.085] | 0.063 |
| ndcg@10 | +0.028 | [-0.011, +0.065] | 0.160 | 

## Step 4: hybrid (RRF) and rerankers on top of `dense|section|html`

All four variants share the best baseline: section chunks, cleaned text. `rrf` fuses the BM25 and dense
rankings with Reciprocal Rank Fusion (k = 60). `+minilm` / `+bge` take dense's top 30 chunks and re-sort them
with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`, 22M params; `BAAI/bge-reranker-v2-m3`, 568M, bf16, max 512 tokens),
then append the rest of the dense ranking. Latencies are p50 / p95 per query on the RTX 3050.

### ExpertWritten

| config | any@5 | all@5 | any@20 | mrr | ndcg@10 | p50 / p95 ms |
|---|---|---|---|---|---|---|
| dense (baseline) | 0.830 [0.78, 0.88] | 0.705 [0.64, 0.77] | 0.955 [0.93, 0.98] | 0.605 [0.55, 0.66] | 0.647 [0.60, 0.69] | 123 / 154 |
| rrf | 0.780 [0.72, 0.83] | 0.675 [0.61, 0.74] | 0.945 [0.91, 0.97] | 0.567 [0.52, 0.62] | 0.605 [0.56, 0.65] | 166 / 191 |
| dense+minilm | 0.780 [0.72, 0.83] | 0.680 [0.61, 0.74] | 0.945 [0.91, 0.97] | 0.558 [0.51, 0.61] | 0.607 [0.56, 0.65] | 284 / 379 |
| dense+bge | 0.840 [0.79, 0.89] | 0.690 [0.62, 0.76] | 0.950 [0.92, 0.98] | 0.609 [0.56, 0.66] | 0.649 [0.60, 0.69] | 815 / 923 |

### Simulated

| config | any@5 | all@5 | any@20 | mrr | ndcg@10 | p50 / p95 ms |
|---|---|---|---|---|---|---|
| dense (baseline) | 0.700 [0.64, 0.76] | 0.615 [0.55, 0.68] | 0.915 [0.88, 0.95] | 0.503 [0.45, 0.56] | 0.550 [0.50, 0.60] | 146 / 185 |
| rrf | 0.690 [0.62, 0.76] | 0.625 [0.56, 0.69] | 0.880 [0.83, 0.93] | 0.516 [0.46, 0.57] | 0.554 [0.50, 0.61] | 163 / 206 |
| dense+minilm | 0.680 [0.61, 0.74] | 0.615 [0.55, 0.68] | 0.915 [0.88, 0.95] | 0.510 [0.46, 0.56] | 0.561 [0.51, 0.61] | 285 / 374 |
| dense+bge | 0.680 [0.61, 0.74] | 0.610 [0.55, 0.68] | 0.915 [0.88, 0.95] | 0.524 [0.47, 0.58] | 0.581 [0.53, 0.63] | 791 / 899 |

### Paired differences vs the dense baseline (variant minus dense)

| variant | set | any@5 | mrr | ndcg@10 |
|---|---|---|---|---|
| rrf | ExpertWritten | −0.050 [−0.110, +0.005], p 0.096 | −0.038, p 0.138 | **−0.042 [−0.083, −0.002], p 0.040** |
| rrf | Simulated | −0.010, p 0.781 | +0.013, p 0.581 | +0.003, p 0.876 |
| +minilm | ExpertWritten | −0.050 [−0.105, +0.005], p 0.105 | −0.047 [−0.099, +0.004], p 0.070 | −0.039 [−0.081, +0.000], p 0.051 |
| +minilm | Simulated | −0.020, p 0.574 | +0.007, p 0.800 | +0.010, p 0.637 |
| +bge | ExpertWritten | +0.010 [−0.040, +0.060], p 0.788 | +0.004, p 0.900 | +0.003, p 0.906 |
| +bge | Simulated | −0.020, p 0.607 | +0.022, p 0.412 | +0.031 [−0.011, +0.074], p 0.154 |

**Verdict: none of the three earns a place in the default config.**

- **RRF** makes expert-question ranking measurably worse (nDCG@10 −0.042, p = 0.04). BM25 is the weaker
  voter (any@5 0.54 vs 0.70 on Simulated), and RRF gives it an equal vote. This is the same lesson as the 2025 prototype.
- **MiniLM** (a small reranker trained on web search) trends worse on expert questions and doubles latency.
- **bge-reranker-v2-m3** is roughly neutral, with a small nDCG gain on Simulated that is within noise, at about 5.5× the latency.
- The gap the reranker was meant to close is real: the gold article sits in the top 20 for 91.5–95.5% of questions
  but in the top 5 for only 70–83%. Re-sorting 30 chunks with these off-the-shelf cross-encoders does not close it.
  Open checks: how often the gold article is inside the 30 candidates at all; truncation at 512 tokens; and the candidate
  count (30 is untested).

### Follow-up: is it the feature-request articles?

Diagnosis (`python -m lastrag.diagnose`, MiniLM on ExpertWritten): the gold article is inside dense's top 30 chunks for
95.5% of questions, so the candidates are not the limit. Of those, MiniLM moved the gold article down for 72
questions, left 71 unchanged and moved it up for 48. Several of its #1 picks were feature-request articles
("Wix X Request: ...") whose titles repeat the question's words, so we tested removing them
(`no_requests`, a LanceDB pre-filter on `article_type`). Feature requests are 33% of the corpus but only 3 of 258
ExpertWritten gold labels and 7 of 230 Simulated ones. (The synthetic set cannot be used to tune this filter:
a third of its gold articles are feature requests.)

| variant (vs plain dense, paired) | ExpertWritten any@5 | ExpertWritten ndcg@10 | Simulated any@5 | p50 ms |
|---|---|---|---|---|
| dense + filter | +0.010, p 0.54 | +0.009, p 0.29 | +0.015, p 0.33 | 175 |
| dense + filter + minilm | −0.050, p 0.11 | **−0.043, p 0.043** | −0.015, p 0.68 | 439 |
| dense + filter + bge | +0.005, p 0.93 | +0.007, p 0.76 | −0.015, p 0.71 | 928 |

**Result: the filter alone is neutral and does not rescue the rerankers.** Feature requests were part of the
story in individual examples but are not the cause. The latency of the filtered runs is higher than the
unfiltered baseline (175 vs 123 ms p50); that was measured in separate sessions, so treat it as unconfirmed.
Reranking stays off by default. Untested ideas: reranking at article level instead of per-section chunk;
more candidates; a reranker fine-tuned on this kind of data.

## Step 4b: contextual retrieval

For every `section` chunk (19,796 of them) `claude-haiku-5-5` wrote a 1-2 sentence note placing the chunk in its
article (`lastrag/contextual.py`, prompt `v1`, thinking off, the article as a cached prompt prefix). The note is put
in front of the chunk before it is embedded and BM25-indexed (`indexed_texts` in `index.py`); the stored `text`
column stays the original chunk. The tables are separate (`html_section_ctx`), and configs carry a `-ctx` suffix.

**Generation cost.** 19,796 calls, 0 failures, $2.35 in total (input side $1.53, output $0.83), with 68.5% of
prompt tokens read from the prompt cache. A 35-chunk sample had predicted $2.32. The notes have a median of 42 words
(p95 54, max 80); the prompt asked for 1-2 sentences, so they are longer than that wording suggests.

**Paired differences (with context minus without), 200 queries per set.** Each cell is the difference with its 95%
bootstrap interval.

| retriever | set | any@5 (without → with) | recall_any@5 | recall_all@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| dense | ExpertWritten | 0.830 → 0.790 | −0.040 [−0.090, +0.005] | **−0.050 [−0.095, −0.005], p 0.035** | −0.007 [−0.046, +0.031] | −0.019 [−0.049, +0.012] |
| dense | Simulated | 0.700 → 0.700 | +0.000 [−0.035, +0.035] | +0.005 [−0.030, +0.040] | +0.016 [−0.019, +0.053] | +0.026 [−0.003, +0.056] |
| bm25 | ExpertWritten | 0.595 → 0.585 | −0.010 [−0.055, +0.035] | −0.010 [−0.055, +0.035] | **+0.025 [+0.001, +0.051], p 0.043** | +0.020 [−0.001, +0.041] |
| bm25 | Simulated | 0.540 → 0.525 | −0.015 [−0.045, +0.015] | −0.010 [−0.045, +0.025] | −0.000 [−0.022, +0.022] | −0.006 [−0.026, +0.014] |
| rrf | ExpertWritten | 0.780 → 0.795 | +0.015 [−0.030, +0.060] | −0.005 [−0.045, +0.035] | +0.011 [−0.017, +0.038] | +0.016 [−0.006, +0.037] |
| rrf | Simulated | 0.690 → 0.675 | −0.015 [−0.060, +0.030] | −0.015 [−0.065, +0.035] | −0.009 [−0.039, +0.020] | +0.001 [−0.025, +0.026] |

recall_any@20 differences are all within ±0.02 (full output: `python -m lastrag.evaluate --compare A B`).

**Result: no detectable effect, and nothing here goes into the default config.**

- Of 30 paired comparisons (3 retrievers × 2 sets × 5 metrics), 2 have p < 0.05, and they point in opposite
  directions (dense ExpertWritten recall_all@5 worse, bm25 ExpertWritten MRR better). With 30 tests about 1.5 such
  results are expected by chance, and neither survives a multiple-comparison correction (0.05 / 30 ≈ 0.0017). The
  metrics are correlated, so the 30 are not independent, but the conclusion is the same.
- "No detectable effect" is not "no effect". The intervals are about ±0.04-0.05 wide on recall@5, so gains above
  roughly +0.06 are ruled out and smaller ones are not. recall_any@20 is already 0.90-0.95, which leaves little room.
- Untested explanations: the title is already part of the embedded text (`title: ... | text: ...`) and section chunks
  already begin with their heading, so the note may add little the retriever did not have. A title ablation
  (no title / title / title + note) would test it.
- Latency: p50 with context was higher in every pair (dense 225 vs 123 ms on ExpertWritten, 219 vs 146 on Simulated;
  bm25 8-9 vs 5-6 ms; rrf 203-207 vs 163-166 ms). The no-title arms in Step 4c, which carry no notes at all, show
  the same gap (dense 231 and 250 ms), so the notes are not the cause. The original baseline was measured in a
  different session (git `20ad595-dirty`); until it is re-run back to back with the others, latency differences between
  these rows are unconfirmed.

## Step 4c: is the title doing the note's job?

The embedded text of every chunk is `title: <article title> | text: <chunk>`. To see whether the generated note
added anything the title did not already give, two more index variants were built: the same chunks with the title
left out, with and without the note (`-notitle`, `-ctx-notitle`). Dense and BM25, both sets, 200 queries each.
Differences are B minus A with 95% bootstrap intervals; `*` marks p < 0.05.

**Dense, ExpertWritten**

| comparison (A → B) | recall_any@5 | MRR | nDCG@10 |
|---|---|---|---|
| title → no title | −0.105 [−0.155, −0.060]* | −0.124 [−0.162, −0.086]* | −0.119 [−0.152, −0.086]* |
| no title → no title + note | +0.075 [+0.015, +0.135]* | +0.096 [+0.052, +0.140]* | +0.086 [+0.048, +0.123]* |
| title → title + note | −0.040 [−0.090, +0.005] | −0.007 [−0.046, +0.031] | −0.019 [−0.049, +0.012] |
| title + note → note only | +0.010 [−0.035, +0.055] | −0.020 [−0.057, +0.016] | −0.014 [−0.042, +0.014] |

**Dense, Simulated**

| comparison (A → B) | recall_any@20 | MRR | nDCG@10 |
|---|---|---|---|
| title → no title | −0.065 [−0.105, −0.030]* | −0.059 [−0.099, −0.020]* | −0.050 [−0.086, −0.016]* |
| no title → no title + note | +0.065 [+0.030, +0.105]* | +0.028 [−0.016, +0.072] | +0.028 [−0.010, +0.066] |
| title → title + note | −0.015 [−0.045, +0.015] | +0.016 [−0.019, +0.053] | +0.026 [−0.003, +0.056] |
| title + note → note only | +0.015 [−0.010, +0.040] | −0.047 [−0.083, −0.012]* | −0.048 [−0.077, −0.020]* |

BM25 shows the same direction (title → no title: nDCG@10 −0.032 on ExpertWritten, −0.036 on Simulated, both with
p < 0.05); its other differences are small and mostly inside their intervals.

**Result.** The title carries real signal for dense retrieval, and the note replaces most of it when the title is
missing, but adds nothing on top of the title. That explains Step 4b: on this corpus the article titles already say what
the note says. It also suggests when the note would earn its cost: on documents with missing or uninformative titles.
That is a hypothesis for other data, not something these runs test.

Cautions: this step adds about 64 more paired comparisons, so isolated p-values just under 0.05 (for example BM25
ExpertWritten MRR, +0.033) should not be quoted; the large, consistent effects (dropping the title for dense) are
not at risk from that. The `-notitle` runs are labelled `-dirty` in their `git` field because they were run before
the code was committed.

## Metric definitions

- **recall_any@k**: share of questions with at least one gold article in the top k.
- **recall_all@k**: share with every gold article in the top k (26% of ExpertWritten questions need 2–3 articles).
- **MRR**: average of 1 / rank of the first gold article.
- **nDCG@10**: rewards gold articles near the top of the first 10 results.

These are not directly comparable to the WixQA paper's *context recall*, which is an LLM-judged metric (GPT-4o judge, top-5 retrieved passages).
