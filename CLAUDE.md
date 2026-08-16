# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```powershell
# Run the full app (FastAPI + Streamlit)
.\run.ps1

# Run tests
pytest
pytest tests/test_pipeline.py::test_name  # Single test

# Rebuild everything from scratch
python -m cvscreener.generation.pipeline   # Generate 50 CVs (~55 min, ~65 s each)
python -m cvscreener.ingest.index          # Build search index (~11 min)

# Quick development
python -m cvscreener.generation.pipeline --count 5   # first 5 personas only
python -m cvscreener.generation.pipeline --only cv_07
python -m cvscreener.generation.pipeline --stage render --force
python -m cvscreener.ingest.index
pytest -v
```

Both pipelines are resumable: every stage caches its artifact and is skipped if
it already exists, so a renderer fix costs seconds, not another hour.

## Architecture Overview

This is a **local-only bilingual RAG system** over 50 synthetic CVs. No cloud APIs, no auth keys.

### Three main pipelines

1. **Generation** → `src/cvscreener/generation/`
   - `personas.py`: Diverse character matrix (age, role, language, seniority)
   - `profile.py`: LLM writes structured CV facts (JSON Schema constrained)
   - `photo.py`: AI-generated headshots (Pollinations.ai flux)
   - `render.py`: ReportLab renders 3 PDF layouts (Sidebar, Banner, Classic)
   - `pipeline.py`: Orchestrates all stages, resumable (caches each artifact)

2. **Ingestion** → `src/cvscreener/ingest/`
   - `extract.py`: Reads PDFs with column awareness (fixes interleaved columns)
   - `chunk.py`: Splits into sections (contact, summary, experience, etc.)
   - `enrich.py`: Extracts structured facts (names, emails, universities, seniority)
   - `index.py`: Builds dense vectors (bge-m3) + BM25 lexical index, writes parquet

3. **Query/RAG** → `src/cvscreener/rag/`
   - `router.py`: LLM classifier routes queries to `retrieve` | `aggregate` | `chart`
   - `retrieve.py`: Hybrid RRF (dense + BM25), with cross-lingual fairness fix
   - `aggregate.py`: Scans full table for counts/filters (universities, seniority, skills)
   - `answer.py`: Generates grounded response, checks keywords against corpus vocabulary
   - `keywords.py`: Detects absent terms (CNN → flags as not in CVs, suggests alternatives)

Shared, not owned by any pipeline:

- `config.py`: pydantic-settings; `chat_model` etc. are **defaults**, overridable via `.env`
- `branding.py`: the LeadTech palette — single source of truth for PDF, UI and charts
- `textutils.py`: `fold_accents`, `ascii_slug`, `WORD_RE`. The **only** module allowed to
  import `unicodedata`; a test enforces it, because three divergent copies of accent
  folding is how ingestion and retrieval quietly stop agreeing
- `logs.py`: rotating app log + the JSONL transcript
- `api/main.py`: FastAPI + SSE (`plan` → `token`* → `meta` → `done`/`error`)
- `ui/`: Streamlit, four tabs — Chat, Insights, Pipeline, Logs

### Why the router?

Top-k retrieval cannot count accurately. Asking *"How many candidates know Kubernetes?"*
with k=5 is unsolvable: 13 candidates match, and they span far more than 5 chunks.
The router sends:

- Semantic questions ("Who has Kubernetes?") → retrieve top-k chunks
- Aggregate questions ("Count candidates by seniority") → scan all 50 via pandas
- Chart questions ("Pie chart by language") → aggregate + Plotly

### The bilingual bug (solved)

**Dense retrieval** works across languages (embeddings are shared vector space).
**BM25** does not (lexical only). RRF fuses them, but this broke in bilingual:

A Spanish query would rank BM25-only as {ES: many, EN: 0}. An English chunk with the
best semantic match (0.5183) but no BM25 score gets fused to ~half the score of a
weaker Spanish chunk appearing in both rankings. **The best match was losing.**

**Fix:** Reserve final result slots for cross-language chunks if their dense similarity
≥ 0.90 of the best in index. This stays within one metric (no score-scale blending).
Tests: `test_cross_lingual_retrieval_crosses_the_language_barrier` and
`test_competitive_english_cvs_are_not_suppressed` in `test_pipeline.py`.

### Conversational context

Each question used to be routed alone, so *"chart those candidates by seniority"*
plotted all 50. The client sends the previous question and its **resolved** plan
(`previous_question`, `previous_plan` on `POST /chat`); the router inherits any
filter the new question leaves empty. `intent`/`chart_type`/`dimension` are never
inherited — a follow-up always supplies its own verb.

- The API stays stateless: no sessions, and "New query" is just not sending the fields.
- What carries over is the **filter**, not the answer, so "those" re-resolves
  against the whole table rather than the chunks that happened to rank.
- On the retrieve branch the previous question is also prepended to the search
  text — *"¿y dónde estudió?"* has nothing to match on its own.
- `follow_up` is trusted from the model here (unlike `dimension`) because it is a
  boolean: both truths are legal tokens, so the decoder is never cornered. A
  lexical check (`refers_back`) can force it **on**, never off.

## Key Design Decisions (Why, not How)

| What | Why |
|---|---|
| **Ollama + local models** | No API keys to manage, reproducible on any machine with GPU, instant latency feedback |
| **gemma2:9b, not 12b** | 9B fits entirely in GPU on RTX 4070 (18.3 tok/s). 12B spills to CPU (7.3 tok/s) — 2.5× slower |
| **bge-m3 embeddings** | Genuinely multilingual, one index for both ES/EN, no translation overhead |
| **RRF, not weighted fusion** | Scores from dense and BM25 are incomparable (different scales, corpus-dependent). RRF ranks documents instead, needs no tuning constants |
| **Constrained decoding** | Ollama enforces JSON Schema — no parsing bugs. But watch: if the right answer is not in the enum, the model picks the nearest legal value silently |
| **Regex + LLM extraction** | Email/phone/DOB are rigid patterns — regex is instant and exact. LLM only extracts judgement calls (seniority, skills) |
| **ReportLab, not HTML→PDF** | Pure Python, no Chrome/GTK dependency, works offline |
| **Parquet for candidates** | Structured facts (one row = one CV) live here, indexed by dense vectors and BM25 |

## Testing Philosophy

Tests assert **observable behaviour**, not implementation:

- Don't test "retrieval returns top-5 chunks" — test "the answer names the right candidates"
- Don't test "emoji renders" — test "the output contains no HTML escapes that reach the UI"
- Don't test "the model outputs JSON" — test the parsed result is valid and grounded
- When a model **can't express failure** (constrained enum), code must verify output against the literal question

Example: `test_absent_fields_match_no_dimension` and
`test_router_refuses_to_chart_a_field_it_does_not_have` assert that asking for "a pie
chart by gender" produces no gender chart (because gender ∉ CVs), even though the model
picks the nearest enum value. The check is `dimension_supported_by` in `router.py`; the
tests verify it stops wrong charts without blocking the eight legitimate dimensions.

## Notable Bugs Fixed (and Why They're Worth Knowing)

1. **Cross-lingual bias in RRF** (see above) — caused correct English matches to lose
2. **Semantic search too forgiving** — "CNN" → "Computer Vision" was plausible but false
   - Fix: Extract user's literal terms, check them against all 375 chunks, flag absences
3. **Constrained decoder picks wrong enum** — ask for "gender chart", model returns "language"
   - Fix: Verify the chosen dimension was mentioned in the user's question
4. **Model invents arithmetic** — reported 40%/50%/10% instead of 20%/28%/52% for seniority split
   - Fix: Pass exact counts as facts, never ask model to count a truncated table
5. **Chunks grouped wrong in context** — model saw flat list of 5 fragments, output flat list with repeats
   - Fix: Group chunks by candidate before prompt, prompt shows examples of the desired structure
6. **Global model state leaked** — `settings.chat_model = req.model` mutated process state for all requests
   - Fix: Thread model as per-request argument; global is only a default fallback
7. **Context lost between questions** — "chart those candidates" charted all 50
   - Fix: Carry the previous *resolved plan*, inherit only unstated filters (see above)
8. **Path traversal via `%5C`** — `/cv/..%5Cdecoy` served a file outside the corpus with 200
   - Not the obvious vector: Starlette normalises `../`, but a backslash is not a URL
     path separator, so it survives routing and `Path.glob` walks up on Windows
   - Fix: Validate `cv_id` against `[A-Za-z0-9_-]{1,64}` **and** re-check the resolved parent

## Logging Strategy

Two separate log files, each answers a different question:

- **`cvscreener.log`** (rotating, 1MB × 3 backups) — diagnostics (Ollama timeouts, tracebacks)
- **`chat.jsonl`** (one object per question) — analytics (intent distribution, error rate, exact Q&A)

Questions recorded in full because **corpus is synthetic**. Against real CVs: this
is a privacy decision, not code. Recruiter queries + candidate names = personal data.

## Code Style

- Docstrings are minimal; code structure should be self-explanatory
- Comments explain *why*, not *what* — the code says what
- Multi-step algorithms get one overview comment at the top
- Prompt engineering: comment why each line exists (prompts are fragile)
- Type hints are used (Python 3.11+)
- Tests use clear assertion messages, not just `assert x`

## When You're Debugging

1. **Check the logs first**: `data/logs/chat.jsonl` holds the exact Q&A for every request
2. **Use the Pipeline tab in Streamlit**: Shows dense rank, BM25 rank, RRF score per chunk
3. **The model does what you show it**: If output is flat, context was flat. Shape the context.
4. **Constrained decoding hides failures**: Check `router.py` for validation gates (dimension_supported_by, keywords check)
5. **Off-topic detection is intentional**: No separate detector. Bad questions go to retrieve, model says "not in CVs"

## Performance Notes

- Time to first token: ~7 s (route + embed happen in parallel)
- Full answer: ~12 s on RTX 4070 Laptop
- Architecture is model-agnostic (sidebar switches models live)
- Both pipelines are resumable — comment out stages you don't need
