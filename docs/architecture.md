# Architecture

## Overview

The whole system on one page. The three workflows below expand each band.

```mermaid
flowchart TB
    subgraph GEN["1 · Generation — offline, once (~55 min)"]
        direction LR
        PM["Persona matrix<br/>50 personas, fixed seed<br/>role · seniority · age<br/>city · language · template"]
        LLM1["gemma2:9b<br/>JSON Schema constrained"]
        POLL["pollinations.ai<br/>flux · free, no API key"]
        RL["ReportLab<br/>3 distinct layouts"]
        PDF[("data/cvs/<br/>50 PDFs<br/>25 ES · 25 EN")]
        PM --> LLM1 --> RL
        PM --> POLL --> RL
        RL --> PDF
    end

    subgraph ING["2 · Ingestion — offline, re-runnable (~11 min)"]
        direction LR
        EX["pdfplumber<br/><b>column-aware</b><br/>gutter detection"]
        EN["gemma2:9b + regex<br/>hybrid fact extraction"]
        CH["Section-aware chunking<br/>bilingual headings"]
        EMB["bge-m3<br/>1024-d multilingual"]
        IDX[("data/index/<br/>375 chunks<br/>375×1024 vectors<br/>BM25 · parquet")]
        EX --> EN --> CH --> EMB --> IDX
    end

    subgraph QRY["3 · Query — online"]
        direction TB
        Q["Question<br/>ES or EN"]
        R{"Router<br/>gemma2:9b<br/>schema + few-shot"}
        subgraph PAR[" concurrent "]
            direction LR
            DENSE["Dense<br/>numpy matmul"]
            BM["BM25<br/>lexical"]
        end
        RRF["Reciprocal Rank Fusion<br/>k=60 · rank-based"]
        AGG["pandas over<br/><b>all 50</b> candidates"]
        ANS["Grounded answer<br/>+ citations · streamed"]
        CHART["Plotly<br/>validated palette"]

        Q --> R
        R -->|retrieve| PAR --> RRF --> ANS
        R -->|aggregate| AGG --> ANS
        R -->|chart| AGG --> CHART
    end

    PDF ==> EX
    IDX ==> PAR
    IDX ==> AGG

    UI["Streamlit · LeadTech theme"] <-->|SSE| API["FastAPI"]
    API <--> QRY
```

The three workflows are joined by two artifacts on disk. Generation ends at
`data/cvs/*.pdf`; the backend picks those up and ends at `data/index/`; the UI
never touches either, and reaches the corpus only through the API.

In the diagrams below, cylinders are files on disk, diamonds are decisions, and
everything else is code.

---

## 1 · CV generation — offline, run once (~55 min)

```mermaid
flowchart TB
    SEED(["build_personas()<br/>SEED fixed"])

    subgraph MATRIX["Persona matrix — diversity engineered, not hoped for"]
        direction LR
        S1["Stream 1 · Random(SEED)<br/>drives personas 0-27"]
        S2["Stream 2 · Random(SEED+1)<br/>drives personas 28-49"]
        P50["50 personas<br/>role · seniority · age · city<br/>language · template · gender"]
        S1 --> P50
        S2 --> P50
    end

    SEED --> MATRIX
    P50 --> LOOP{"for each persona:<br/>artifact already cached?"}

    LOOP -->|photo missing| POLL["pollinations.ai · flux<br/>portrait prompt per persona"]
    POLL -->|network error| MONO["_monogram fallback<br/>initials on a mint tile"]
    POLL --> JPG[("data/photos/id.jpg")]
    MONO --> JPG

    LOOP -->|profile missing| GEN["generate_profile<br/>gemma2:9b + JSON Schema<br/>~65 s — the expensive stage"]
    GEN --> CONTACT["_contact / _birth_date<br/>deterministic, seeded on cv_id<br/>example.com · day capped at 28"]
    CONTACT --> JSON[("data/profiles/id.json")]

    LOOP -->|cached| SKIP["skip — reuse the artifact"]

    JPG --> RENDER
    JSON --> RENDER
    SKIP --> RENDER

    RENDER["render_cv · ReportLab<br/>ALWAYS re-run — milliseconds<br/>3 layouts: Sidebar · Banner · Classic<br/>TrueType fonts, so 'ń' survives"]
    RENDER --> PDF2[("data/cvs/ — 50 PDFs<br/>25 ES · 25 EN")]
```

**Why two RNG streams.** The corpus grew from 28 CVs to 50 after the first batch
had already been generated. A single stream reseeded for 50 draws would have
reshuffled the original 28 — `cv_07` would silently become a different person,
invalidating half an hour of cached PDFs and photos. Stream 1 keeps its exact
call sequence; stream 2 handles everything appended afterwards.

**Why render is never cached.** Text costs ~65 s per CV; rendering costs
milliseconds. Caching the expensive stage and always redoing the cheap one is
what makes a layout bug cost one re-render instead of an hour of regeneration.

---

## 2 · Backend — ingestion offline (~11 min), query online

```mermaid
flowchart TB
    PDF3[("data/cvs/*.pdf")]

    subgraph ING2["Ingestion — python -m cvscreener.ingest.index"]
        direction LR
        EX2["extract.py · pdfplumber<br/>detect the whitespace gutter,<br/>read each column separately"]
        CH2["chunk.py · section-aware<br/>bilingual headings; every chunk<br/>prefixed 'Name — Section'"]
        EN2["enrich.py · hybrid<br/>regex: email, phone, DOB<br/>gemma2:9b: seniority, skills"]
        EMB2["index.py · bge-m3<br/>1024-d, L2-normalised"]
        EX2 --> CH2 --> EN2 --> EMB2
    end

    PDF3 ==>|never reads data/profiles/| EX2
    EMB2 --> IDX2[("data/index/<br/>embeddings.npy · chunks.jsonl<br/>bm25.pkl · candidates.parquet")]

    subgraph APIB["POST /chat — one SSE stream per question"]
        direction TB
        REQ(["question + previous_question<br/>+ previous_plan"])

        subgraph CONC["concurrent — halves time to first token"]
            direction LR
            EMBQ["embed_query · bge-m3"]
            ROUTE["router.py · gemma2:9b<br/>JSON Schema + few-shot"]
        end
        REQ --> CONC

        GATE{"deterministic gates<br/>applied to the model's plan"}
        ROUTE --> GATE
        GATE -->|dimension absent from the question| UNSUP["dimension := unsupported<br/>withhold the chart"]
        GATE -->|refers_back or follow_up| CARRY["carry_over — inherit<br/>the filters left unstated"]

        INTENT{intent}
        GATE --> INTENT
        CARRY --> INTENT
        UNSUP --> INTENT
        INTENT -->|retrieve| SEARCH
        INTENT -->|aggregate / chart| AGG2

        subgraph SEARCH["retrieve.py · hybrid"]
            direction TB
            DD["dense · one numpy matmul<br/>375 chunks, sub-millisecond"]
            BB["BM25 · exact tokens"]
            RRF2["RRF, k=60 — ranks only"]
            FAIR["cross-lingual reservation<br/>dense >= 0.90 x best in index"]
            DD --> RRF2
            BB --> RRF2
            RRF2 --> FAIR
        end

        AGG2["aggregate.py · pandas over<br/>ALL 50 rows — exact counts<br/>handed over as facts"]

        FAIR --> ANS2
        AGG2 --> ANS2
        ANS2["answer.py — context grouped<br/>BY CANDIDATE; reply in the<br/>question's language"]
        ANS2 --> SSE["SSE: plan → token* → meta → done"]
        SSE --> LOG2[("data/logs/chat.jsonl<br/>written on success AND on error")]
    end

    EMBQ -.->|query vector| SEARCH
    IDX2 ==> SEARCH
    IDX2 ==> AGG2
```

**The integrity constraint** is the `PDF ==> extract` edge. `data/profiles/*.json`
holds everything the generator produced, already parsed, sitting right next to
the PDFs — and ingestion never opens it. Otherwise the demo would prove the
generator can round-trip its own JSON, not that the system can read a CV.

**The gates exist because constrained decoding cannot fail loudly.** The schema
guarantees a well-formed plan; it cannot guarantee a true one. `dimension` is an
enum, so a request for a gender chart returns the nearest legal value with full
confidence — measured as `cv_language` in English and `seniority` in Spanish. So
the plan is checked against the question's own words before anything is plotted.

---

## 3 · User interface — Streamlit, four tabs

```mermaid
flowchart TB
    USER(["Recruiter types a question"])
    USER --> SEND["stream_chat()<br/>POST /chat over SSE"]

    SEND --> EV{"SSE event"}
    EV -->|plan| BADGE["routing_badge — intent,<br/>inherited filters,<br/>'not in any CV' warnings"]
    EV -->|token| BUBBLE["repaint the bubble<br/>+ block cursor"]
    EV -->|meta| SRC["citations · chunks<br/>chart spec"]
    EV -->|error| ERR["st.error — never hang"]

    BUBBLE --> TIDY["tidy_answer<br/>strip empty bullets"]
    TIDY --> RMD["render_markdown<br/>escape first, then<br/>bold / italic / lists"]

    BADGE --> COMMIT
    RMD --> COMMIT
    SRC --> COMMIT
    ERR --> COMMIT

    COMMIT["commit the turn to st.session_state<br/>messages[] · context · last_meta"]
    COMMIT --> STATE[("st.session_state<br/>messages[] — role · content · plan · citations · chart<br/>context — previous question + its RESOLVED plan<br/>last_meta — retrieval trace for the Pipeline tab")]

    STATE -.->|"context travels with the next question<br/>(New query clears it)"| SEND
```

| Tab | Accent | Reads |
|---|---|---|
| **Chat** | mint | `messages[]` and `context` — the answer, its routing badge, its sources, and the line naming the question being followed |
| **Insights** | sky | `GET /candidates` — corpus charts and the full table |
| **Pipeline** | amber | `last_meta` — dense rank, BM25 rank and fused score per chunk, plus `GET /stats` |
| **Logs** | pink | `GET /logs`, `DELETE /logs` — every exchange with latency and errors |

**Why the conversation lives in the client.** The API holds no sessions, so
there is no cache to expire, no key to leak between users, and every request is
reproducible from its own body. "New query" is simply not sending two fields.

**Why the plan is stored on each message.** The routing badge used to be written
straight into a live `st.empty()`, so the moment the transcript re-rendered, the
explanation of *how* an answer was reached vanished — leaving the answer with
nothing to justify it.

**Why `render_markdown` exists.** The bubbles are raw HTML `<div>`s, and
CommonMark does not process markdown inside a raw HTML block — a blank line
*terminates* the block. The grouped answer format was the first thing to put
blank lines inside one, so half an answer rendered as literal `**Name**`. The
converter escapes first, which also closed a hole: model output was reaching
`unsafe_allow_html=True` verbatim.

## Why the query layer is routed

A plain RAG pipeline retrieves the top *k* chunks and answers from them. That is
the right shape for "who has experience with Kubernetes?" and structurally wrong
for "how many candidates know Kubernetes?" — with `k=5`, the honest answer to a
counting question is unreachable no matter how good the retriever is.

So the router picks the machinery that can actually answer:

| Intent | Machinery | Example |
|---|---|---|
| `retrieve` | hybrid search → grounded generation | *¿Quién sabe aprendizaje automático?* |
| `aggregate` | pandas over the full candidate table | *How many candidates know Kubernetes?* → **exactly 13 of 50** |
| `chart` | same, plus a Plotly figure | *Histograma de las edades de los que sepan Python* |

## Why RRF, and not a weighted score blend

Measured on this corpus with `bge-m3`:

| Pair | Cosine |
|---|---|
| `"aprendizaje automático…"` ↔ `"machine learning…"` | **0.745** |
| Spanish question ↔ relevant English CV | **0.580** |
| Spanish question ↔ unrelated English text | **0.475** |

The signal is real, but the floor is high and the usable band is narrow — and
BM25 scores are unbounded and corpus-dependent. Mapping two distributions of
such different shape onto one scale means inventing constants that would not
survive a change of corpus.

Reciprocal Rank Fusion consumes only **ranks**:

$$\text{RRF}(d) = \sum_{r \in \{\text{dense},\ \text{bm25}\}} \frac{1}{k + \text{rank}_r(d)}$$

No thresholds, no normalisation, no tuning. Each retriever contributes what it
is good at: BM25 nails literal tokens (`UPC`, `PostgreSQL`, a surname), the
dense index crosses the language barrier.

## The integrity constraint

`data/profiles/*.json` holds the structured output the generator produced. The
ingestion pipeline **never reads it**. Everything — names, skills, ages,
employers — is re-derived from text pulled back out of the PDFs.

Without that rule the demo would be circular: it would prove the generator can
round-trip its own JSON, not that the system can read a CV. With it, the
pipeline treats each PDF as an opaque document, exactly as it would a résumé a
real candidate sent in.

## Design decisions

| Decision | Reason |
|---|---|
| **numpy, not a vector DB** | 375 vectors (chunks). Exact cosine is one matmul, sub-millisecond. An ANN index would be slower, approximate, and another dependency. |
| **gemma2:9b over gemma4:12b** | Profiled on this 8 GB RTX 4070: the 12B gets only 4.4 GB onto the GPU and runs the rest on CPU → 7.3 tok/s. The 9B fits → 18.3 tok/s, 2.5× faster at comparable quality. |
| **bge-m3 embeddings** | Genuinely multilingual, so one index serves both languages. A monolingual model would need either translation or two indexes. |
| **Constrained decoding** | Ollama enforces the JSON Schema, so no output parsing or JSON repair anywhere in the codebase. |
| **Few-shot in the router** | Schema guarantees shape, not meaning: zero-shot, gemma2 labelled the histogram request `retrieve`. The examples fixed it. |
| **ReportLab, not HTML→PDF** | Pure Python wheels. No Chrome, no GTK for a reviewer to install. |
| **Regex + LLM extraction** | E-mail, phone and date of birth have rigid surface forms; `re` solves them exactly and instantly. The LLM handles only what needs judgement. |
| **Concurrent route + embed** | Independent, and both models are resident in VRAM. Overlapping them halves time-to-first-token (~13 s → ~7 s). |

## Problems hit along the way

**Two-column extraction.** `pdfplumber` reads by vertical position across the
full page width, so the sidebar template came out interleaved — a phone number
landing mid-sentence in the professional summary. Fixed by detecting the
whitespace gutter empirically (scan x positions, keep those no word crosses,
require both sides to hold ≥12% of the words) and extracting each column
separately. Detecting it rather than hard-coding our own template's geometry
means it also works on a CV the pipeline did not generate.

**Silent glyph corruption.** ReportLab draws `Paragraph` bullets in the style's
`bulletFontName`, which defaults to Helvetica. "•" has no Helvetica mapping, so
every bullet extracted as `(cid:127)` — invisible in the rendered PDF, poisoning
every chunk and embedding downstream. Caught only by reading the extracted text.

**Latin-1 fonts.** ReportLab's built-in fonts cannot encode `ń`, and the corpus
contains *Katarzyna Wilczyńska*. Fixed by registering real TrueType families.
