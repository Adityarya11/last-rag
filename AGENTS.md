# AGENTS.md — last-rag

## What this project is

**last-rag** is an educational, measured RAG system. Every pipeline technique is a
toggle, and every toggle is measured for **quality, latency and cost**. Users flip
toggles in the UI and see what each one actually buys. The point is to show that
"turn everything on" is not automatically best: the right config depends on the
data, the use case and the budget.

Origin story (README tone): a 2025 internship FAQ bot sat on a laptop, got looked at
again, and was rebuilt properly.

The README must carry a disclaimer: this is the smallest honest version of a RAG system.
The dataset is clean and structured with little preprocessing, so results will not
transfer blindly to messy enterprise data. The toggles are there to explore that.

## Hard rules

- **No private data, ever.** The HireSmart/AmberFlux FAQs are an undisclosed company
  policy and must never enter this repo, in any form: no copies, no derived eval
  sets, no quoted answers.
- **No LangChain / LangGraph / LlamaIndex.** Each stage is a plain Python function
  with a typed input and output, so it can be toggled, timed and tested alone. The 2025
  version broke mostly at framework glue: stale cached memory, sync calls inside async
  code, deprecated memory classes.
- **Nothing ships unmeasured.** A stage enters the default config only if the eval
  shows it helps, with a confidence interval that excludes zero, for its latency and cost.
- **Never fabricate numbers.** Every number in the README or UI comes from a stored run
  in `runs/`.
- Don't commit, push or publish unless the user asks.

## Machine notes (Windows 11, user's laptop)

- **Disk space is tight.** C: has about 22 GB free, E: about 9 GB. Never put large
  artifacts on C:.
- Env vars already point caches off C:: `UV_CACHE_DIR=E:\AI\uv-cache`,
  `HF_HOME=E:\AI\huggingface`, `UV_PYTHON_INSTALL_DIR=D:\Dev\uv\python`.
  Keep the project on E: so uv can hardlink from its cache instead of copying.
- GPU: RTX 3050 Laptop, **4 GB VRAM**, driver CUDA 13.1, so use torch from the
  `cu130` index. Size models to fit: about 1 GB per 500M params in bf16, with headroom
  needed for batches.
- Python 3.14 through uv. Shells are PowerShell (primary) and Git Bash.
- Ask before any install over about 1 GB, and say where it will land.

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Web | FastAPI + Jinja2 + HTMX | Server-rendered. No JS framework. |
| Vector store | LanceDB (embedded) | Dense and full-text search in one place. Native hybrid search is one toggle; our own RRF is another. |
| Embeddings | `google/embeddinggemma-2` | Apache-2.0, text-only path about 270M params, 768-d with Matryoshka 512/256/128 (re-normalize after truncating), 8k ctx. Query prompt `task: search result \| query: {q}`, doc prompt `title: {title} \| text: {body}`. **bf16 or fp32, never fp16** (NaNs). Load text modality only. Dims are a toggle. |
| Reranker | local cross-encoder | Pick by benchmark. Candidates include `BAAI/bge-reranker-v2-m3`; check what's current and confirm it fits 4 GB. API rerankers (Voyage/Cohere) are optional comparison arms. |
| LLM | Claude Haiku (cheap per-query work), Sonnet (final answer) | **Load the `claude-api` skill before writing any Claude call**, then confirm model IDs and pricing there, not from memory. Use prompt caching for the static system and context layers. |
| Router | **Jev** (TypeSafe AI, early access) vs Claude Haiku | Jev is a non-generative "System One" model: you send a typed schema of questions and it returns typed answers with calibrated confidence, at a vendor-claimed 70–500 ms. API only. **Read docs.typesafe.ai before coding; the request schema is unknown so far.** Treat its speed and cost figures as vendor claims and measure them ourselves. |
| Deps / env | uv | `pyproject.toml` pins the torch source to the `pytorch-cu130` index (explicit). |

## Dataset: WixQA (primary)

Hugging Face `Wix/WixQA`, **MIT license** (cite "Wix.com AI Research"; paper arXiv 2505.08643).
It is a real company's public help center, which is why it was chosen over a toy FAQ set.

- `wix_kb_corpus`: **6,221 articles** with fields `id, url, contents, article_type`
  (`article` / `feature_request` / `known_issue`). Contents are HTML-stripped and **long**:
  the median source article behind an ExpertWritten answer is about 5.9k tokens. **Chunking matters here**,
  so contextual retrieval is a meaningful test on this data, unlike on the old FAQ set.
- `wixqa_expertwritten`: 200 real user queries with expert answers; 27% need more than one article.
- `wixqa_simulated`: 200 queries distilled from support chats, the hardest set; 14% need more than one article.
- `wixqa_synthetic`: 6,221 LLM-made pairs, one per article. Use these for tuning and
  dev, **never for headline numbers.**
- Fields: `question, answer (markdown), article_ids (gold)`.

**Published baselines to beat** (FlashRAG, top-k=5, GPT-4o as judge):

| Retriever | ExpertWritten context recall | Simulated context recall |
|---|---|---|
| BM25 | 0.73 | 0.55 |
| E5-large-v2 | 0.81 | 0.67 |

Factuality was about 0.72–0.85, and token F1 0.21–0.43, depending on generator.
Our judge differs from theirs (Claude, not GPT-4o), so say so whenever we compare.

**Gaps we fill ourselves.** WixQA is single-turn and has no unanswerable queries, so we build an
**adversarial / routing set** (Wix-grounded, generated, then hand-verified, stored in-repo):
- out of scope ("what's the weather")
- meta questions about the app ("what does this app do", "who built you")
- ambiguous queries needing clarification ("it's not working")
- false premises (contradicting a KB fact)
- multi-turn follow-ups ("and how do I undo that?")
- prompt injection planted in a document
- Optional out-of-scope source: CLINC150 OOS split (CC BY 4.0 per UCI).

Rejected: MultiDoc2Dial (license unclear, 488 docs), EnterpriseRAG-Bench (fictional company,
Slack/email sources), RAGBench subsets (not one company's help center).

## Pipeline: each stage is a toggle

1. **Route** (Jev | Haiku | off): classify `answerable / meta / out_of_scope / ambiguous / chitchat`
   with calibrated confidence **before** any retrieval. Non-answerable queries short-circuit
   with a canned or clarifying reply. Measure routing accuracy, latency saved and cost saved.
2. **Query transform**: off | rewrite (resolve multi-turn references) | HyDE | multi-query.
3. **Chunking** (index-time): fixed-size | structure-aware (headings/steps) | ± **contextual
   retrieval** (Haiku writes a 1–2 sentence prefix situating the chunk in its article, using prompt caching).
4. **Retrieve**: dense | BM25/FTS | hybrid (LanceDB native) | hybrid (our RRF, k tunable). top-k candidates.
5. **Rerank**: off | cross-encoder, keep top-n. This is the talk's "fetch 20–25, keep 5–10" idea,
   which we test rather than assume.
6. **Grade / correct (CRAG-lite)**: off | grade evidence; if weak → rewrite and retry once, or abstain.
7. **Generate**: three-layer prompt (system with faithfulness mandate → context with source tags → query last).
   Answers cite article ids or chunks.
8. **Verify** (stretch): claim-level check of the answer against cited chunks; drop or flag unsupported claims.

Every stage records `{latency_ms, input_tokens, output_tokens, cost_usd, model}` into a
per-request trace. The trace drives both the UI telemetry and the eval runs.

## Evaluation

- **Retrieval** (gold `article_ids`): recall@k, MRR, nDCG@10, at both article and chunk level.
- **Generation**: the RAG triad (context relevance, groundedness/faithfulness, answer relevance) via an
  LLM judge, plus factuality against the gold answer and token F1 so we're comparable to the paper.
- **Routing / abstention**: accuracy per class on the adversarial set, and the false-refusal rate on
  answerable queries.
- **Systems**: p50 and p95 latency per stage and end to end; $ per query; $ per 1k queries.
- **Statistics**: bootstrap 95% CIs on every metric and paired tests between configs. 200 queries
  is not many, so report intervals, never bare points.
- **Judge bias**: Claude judging Claude may favour its own output. Hand-label a subset (about 50) and
  report judge–human agreement.
- Runs are stored as JSON in `runs/` (config hash, git sha, per-query traces). The metrics page reads only from these.

## UI (HTMX)

- **Chat page**: toggle panel, answer with clickable citations, and a per-request telemetry strip
  (stage timings waterfall, tokens, $).
- **Metrics page**: config × metric table with CIs, a **Pareto frontier** (quality vs $/query vs p95
  latency), and "what this config would cost at N queries/day". It should make "everything on ≠ best"
  visible. It's also the source for the demo video.

## Lesson carried from the 2025 prototype (numbers only, no data)

On 37 FAQs with 36 paraphrased queries, MiniLM dense (question+answer indexed) scored
recall@1 0.83 / MRR 0.905. Hybrid RRF with BM25 was **worse** (0.67 / 0.772), because BM25 is weak on
paraphrases and RRF gives it an equal vote. Indexing question+answer beat question-only by a
wide margin. Takeaway: hybrid and fusion are hypotheses to test, not defaults.

## Build order

0. Env: uv project, CUDA torch, verify the GPU is visible.
1. Data loader for WixQA, plus a corpus stats script (article lengths, which shape chunking).
2. Eval harness first: retrieval metrics, bootstrap CIs, run storage.
3. Baselines: BM25 and dense with embeddinggemma-2. Reproduce roughly the paper's numbers.
4. Chunking variants, then hybrid/RRF, then reranker, then contextual retrieval. Measure each step.
5. Generation + LLM judge + hand-labeled judge check.
6. Router (Haiku first, then Jev) + adversarial set.
7. Query transforms, CRAG-lite, verify.
8. FastAPI/HTMX chat + telemetry, then metrics page.
9. README with results, disclaimer, demo video.
