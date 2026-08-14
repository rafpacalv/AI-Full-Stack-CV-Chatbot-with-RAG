# LeadTech · AI-Powered CV Screener

A bilingual (ES/EN) RAG system over a corpus of 28 synthetic CVs, running
**entirely on local models via Ollama** — no API keys, no cloud inference.

Ask in Spanish, get answers grounded in English CVs. Ask for a count, get an
exact figure computed over all 28 candidates rather than a guess from five
retrieved chunks. Ask for a histogram, get one.

![Chat](docs/img/ui-chart.png)

---

## What it does

| | |
|---|---|
| **Generate** | 28 unique fake CVs as PDFs — LLM-authored text, AI-generated headshots, 3 layouts, 14 Spanish + 14 English |
| **Ingest** | Column-aware PDF extraction → section chunking → structured facts → dense + lexical index |
| **Ask** | Routed queries: semantic retrieval, exact aggregates, or on-demand charts — all streamed, all cited |

---

## Quick start

Requires **Python 3.11+**, **[Ollama](https://ollama.com)**, and ~8 GB of disk for the models.

```bash
ollama pull gemma2:9b        # chat, routing, fact extraction
ollama pull bge-m3           # multilingual embeddings

python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
copy .env.example .env
```

The repository already ships the generated corpus and index, so you can go
straight to:

```powershell
.\run.ps1          # starts FastAPI (:8000) and Streamlit (:8501)
```

To rebuild everything from scratch:

```bash
python -m cvscreener.generation.pipeline   # 28 CVs   (~25 min)
python -m cvscreener.ingest.index          # search index (~13 min)
pytest                                     # 33 tests
```

Both pipelines are **resumable** — each stage caches its artefact and is skipped
when it already exists, so fixing the renderer costs seconds rather than another
full generation run.

---

## Architecture

Full diagram and rationale: **[docs/architecture.md](docs/architecture.md)**

```
Generation   persona matrix → gemma2:9b (JSON Schema) → ReportLab ⟶ 28 PDFs
                            → pollinations.ai (flux)  ↗

Ingestion    PDF → pdfplumber (column-aware) → section chunks
                 → gemma2:9b + regex ⟶ candidates.parquet
                 → bge-m3            ⟶ 208 × 1024 vectors + BM25

Query        question → router ┬─ retrieve  → dense ⊕ BM25 → RRF → grounded answer
                              ├─ aggregate → pandas over all 28 → exact count
                              └─ chart     → pandas → Plotly
```

### Three things worth pointing at

**1 · The router exists because top-*k* cannot count.**
"Who knows Kubernetes?" is a retrieval question. "*How many* candidates know
Kubernetes?" is not — with `k=5`, the correct answer is unreachable by
construction, however good the retriever. So a schema-constrained classifier
sends counting and charting questions to pandas over the full candidate table
instead. *8 of 28*, verified against the source data, not estimated.

**2 · Metadata is re-derived from the PDFs, never reused from the generator.**
`data/profiles/*.json` holds exactly the structured data the ingestion step
wants. It is never read. Everything is re-extracted from text pulled back out of
the PDFs, so the pipeline is doing the real job — reading documents — rather
than round-tripping its own JSON.

**3 · Hybrid retrieval had a bilingual failure mode, and fixing it needed measurement.**
See below.

---

## The most interesting bug

Hybrid retrieval fuses a dense retriever with BM25 using Reciprocal Rank Fusion,
which rewards documents appearing in *both* rankings. In a bilingual corpus that
quietly breaks:

```
Spanish query → BM25 scores:  { es: 56 non-zero,  en: 0 }
```

A lexical retriever **cannot match across a language boundary**. So an English
chunk can only ever collect rank mass from one of the two retrievers — it is
structurally capped at roughly half the fused score of a Spanish chunk, no
matter how relevant it is.

This suppressed correct answers. For *"¿Qué candidato se dedica al diseño de
producto y accesibilidad?"*:

| Chunk | Dense similarity | In results? |
|---|---|---|
| **Rasmus Lindqvist** (EN, Product Designer) | **0.5183** — best in the entire index | ❌ no |
| Adrián Quesada (ES) | 0.4667 (0.90 of best) | ✅ yes |
| Marta Sanchis (ES) | 0.4597 (0.89 of best) | ✅ yes |

The best match in the corpus was losing to two weaker ones because of a scoring
artefact.

**The fix** reserves final slots for other-language chunks whose *dense*
similarity stands on its own — at least 0.90 of the best in the index. The
threshold is relative and stays within a single metric, so it never blends the
two incomparable score scales that RRF was chosen to avoid. Three regression
tests pin the behaviour.

---

## Engineering decisions

| Decision | Why |
|---|---|
| **numpy, not a vector DB** | 208 vectors. Exact cosine is one matmul, sub-millisecond. An ANN index would be slower, approximate, and another dependency. |
| **gemma2:9b, not gemma4:12b** | Profiled on this 8 GB RTX 4070 Laptop: the 12B gets only 4.4 GB onto the GPU and runs the rest on CPU → **7.3 tok/s**. The 9B fits entirely → **18.3 tok/s**. 2.5× faster for comparable quality once prompts were tightened. |
| **bge-m3 embeddings** | Genuinely multilingual — one index serves both languages, no translation step. |
| **RRF, not weighted scores** | Measured: bge-m3 scores unrelated text at 0.475 and relevant text at 0.580. A high floor and a narrow band, against unbounded corpus-dependent BM25 scores. Rank-based fusion needs no constants that would break on a different corpus. |
| **Constrained decoding** | Ollama enforces the JSON Schema, so there is no output parsing or JSON-repair code anywhere. |
| **Few-shot in the router** | Schema fixes shape, not meaning. Zero-shot, gemma2 classified *"genera un histograma…"* as `retrieve`. The examples fixed it. |
| **Regex + LLM extraction** | E-mail, phone and date of birth have rigid forms; `re` handles them exactly and instantly. The LLM only does what needs judgement. |
| **Concurrent route + embed** | Independent, both models resident in VRAM. Overlapping halves time-to-first-token: ~13 s → ~7 s. |
| **ReportLab, not HTML→PDF** | Pure-Python wheels. No Chrome or GTK for a reviewer to install. |

### Two bugs that only surfaced by reading the extracted text

**Silent glyph corruption.** ReportLab draws bullets in the style's
`bulletFontName`, defaulting to Helvetica, where `•` has no mapping. Every
bullet extracted as `(cid:127)` — invisible in the rendered PDF, poisoning every
chunk and embedding downstream.

**Interleaved columns.** `pdfplumber` reads by vertical position across the full
page, so the sidebar layout came out with phone numbers spliced into the middle
of sentences. Fixed by detecting the whitespace gutter empirically rather than
hard-coding our own template's geometry — so it also works on CVs this pipeline
did not generate.

---

## The corpus

28 CVs, 3 layouts, 14 Spanish / 14 English, ages 23–40, 1–14 years of
experience, roles mirroring Leadtech's own departments.

| Sidebar (ES) | Banner (ES) | Classic (EN) |
|---|---|---|
| ![](docs/img/cv-sidebar-es.png) | ![](docs/img/cv-banner-es.png) | ![](docs/img/cv-classic-en.png) |

Diversity is engineered in `personas.py` rather than delegated to the model —
asked for "28 varied CVs" an LLM reliably collapses onto the same few archetypes.
A fixed seed makes the whole corpus reproducible.

**All data is synthetic.** Names are invented, e-mail addresses use the
RFC 2606 reserved `example.com` domains, and every PDF carries
`Subject: Synthetic CV generated for a technical assessment` in its metadata.
Photos come from a diffusion model; the people do not exist.

---

## Interface

Branding is taken from leadtech.com's production CSS and logo SVGs — mint
`#00FFC6`, deep indigo `#140C29`, with Comfortaa and Roboto (the fallbacks their
own stylesheet declares for their proprietary face).

The chart palette is **not** the brand palette. Brand mint sits at OKLCH
L 0.889, far above the band a dark chart surface needs, and brand mint/green and
blue/cyan are hue-twins that collapse under deuteranopia. The hues were
re-stepped in OKLCH and the two collisions dropped; the result passes all six
checks of the colour validator (lightness band, chroma floor, CVD separation,
normal-vision floor, contrast). Worst adjacent pair: ΔE 10.3 deuteranopia,
against a target of 8.

![Insights](docs/img/ui-insights.png)

| Tab | |
|---|---|
| **Chat** | Streamed answers, routing badge, citation chips, source PDF download, live model switch |
| **Insights** | Corpus-level charts and the full candidate table |
| **Pipeline** | Ingestion stats and the retrieval trace for the last question — dense rank, BM25 rank and fused score per chunk |

---

## Layout

```
src/cvscreener/
├── config.py · branding.py · llm.py · textutils.py
├── generation/   personas · profile · photo · render · pipeline
├── ingest/       extract · chunk · enrich · index
├── rag/          router · retrieve · aggregate · answer
└── api/          main.py            FastAPI + SSE
ui/               app.py · theme.py · charts.py
tests/            33 tests
data/             cvs/ · photos/ · profiles/ · index/
```

## API

| Endpoint | |
|---|---|
| `POST /chat` | SSE: `plan` → `token`* → `meta` (citations, chunks, chart) → `done` |
| `POST /search` | Raw retrieval with dense/BM25/RRF scores |
| `POST /route` | The query plan alone |
| `GET /candidates` | The full candidate table |
| `GET /cv/{cv_id}` | The original PDF |
| `GET /stats` · `GET /health` | Index stats · liveness |

Interactive docs at `http://127.0.0.1:8000/docs`.

## Known limitations

- **Latency.** ~7 s to first token, ~12 s total on this hardware. Everything runs
  locally on a laptop GPU; the architecture is model-agnostic and the sidebar
  switches models live.
- **Fact extraction is imperfect.** University names normalise inconsistently
  (`UPC` vs the full name), and "Lead" seniority is sometimes read as "Senior".
- **The cross-lingual ratio (0.90) is corpus-tuned.** It was measured on these
  28 CVs and would need re-checking on a materially different corpus.
