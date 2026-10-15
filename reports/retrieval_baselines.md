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

## Metric definitions

- **recall_any@k**: share of questions with at least one gold article in the top k.
- **recall_all@k**: share with every gold article in the top k (26% of ExpertWritten questions need 2–3 articles).
- **MRR**: average of 1 / rank of the first gold article.
- **nDCG@10**: rewards gold articles near the top of the first 10 results.

These are not directly comparable to the WixQA paper's *context recall*, which is an LLM-judged metric (GPT-4o judge, top-5 retrieved passages).
