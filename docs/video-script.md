# Video script — under 5 minutes

Rehearse once with a timer. Every figure below was checked against the built
index; if you rebuild the corpus, re-check them before recording.

**Before recording:**

1. Run `.\run.ps1` and wait for the sidebar to show `CONNECTED`.
2. Ask one throwaway question so both models are hot — the first call after a
   cold start pays ~31 s to load gemma2 and bge-m3.
3. **Clear the log** in the Logs tab. The tab then holds only the demo
   questions, which makes the closing shot legible.
4. Press **New query** so the chat starts with no conversational context.

**Answers take ~12 s.** Four questions is ~50 s of generation, so narrate *over*
the streaming rather than waiting for it. Each question below marks what to say
while it runs.

---

## 0:00 – 0:20 · What it is

> "An AI CV screener. Fifty fake CVs, half Spanish and half English, and a chat
> that answers questions about them — running entirely on local models through
> Ollama. No API keys, nothing leaves the laptop."

Show the UI briefly. Point at `CONNECTED`, `375 chunks`, `bge-m3`.

---

## 0:20 – 0:55 · The generation pipeline

Show `personas.py`, then a couple of PDFs.

> "The CVs come from a persona matrix — fifty personas with role, seniority,
> age, city, language and layout all fixed up front with a seed. That's
> deliberate: ask a model for fifty varied CVs and you get the same three
> archetypes over and over. Diversity is engineered, then the model writes prose
> inside those constraints."

Open two or three PDFs — one Spanish sidebar, one English classic.

> "Text from gemma2, headshots from a hosted diffusion model, PDFs from
> ReportLab in three layouts. All synthetic: invented names, reserved e-mail
> domains, and every PDF says so in its metadata."

---

## 0:55 – 1:30 · The ingest and RAG workflow

Show the **Pipeline** tab.

> "Ingestion reads only the PDFs. It never touches the JSON the generator
> produced, even though that file sits right next to them with the same data
> already parsed. If I fed that in, I'd be proving the generator can round-trip
> its own output — not that the system can read a CV.
>
> Extract, chunk by section, then two indexes: dense vectors from bge-m3, which
> is multilingual, and BM25 for exact tokens. Fused with Reciprocal Rank
> Fusion — and the trace here shows both ranks and the fused score per chunk."

---

## 1:30 – 3:00 · The demo

Four questions. The last one is the one to protect.

### Q1 — cross-lingual · `retrieve`

> `¿Quién tiene experiencia con aprendizaje automático?`

*While it streams:*

> "Asked in Spanish. Watch the sources — English CVs come back too. Nothing was
> translated; bge-m3 puts both languages in one vector space."

Five candidates, grouped one block per person.

### Q2 — the follow-up · `chart`

> `Ahora un diagrama sectorial de esos candidatos por seniority`

*Point at the badge before the text arrives:* **follows the previous question**,
and a **skill: machine learning** chip the question never mentioned.

> "'Those candidates.' It inherited the filter from the question before it, so
> the pie covers those five people — not all fifty. What carries over is the
> filter, not the answer: it re-runs against the whole table rather than the
> handful of chunks that happened to rank."

Then press **New query**.

> "And that drops the context."

### Q3 — the counting question · `aggregate`

> `How many candidates know Kubernetes?`

Point at the badge: **TABLE AGGREGATE**.

> "This one didn't go to the retriever. A classic RAG pipeline retrieves five
> chunks and counts those five — with top-k, a counting question is unanswerable
> by construction. So a router sends it to pandas over all fifty candidates
> instead. Thirteen. Exact, not estimated."

### Q4 — asking for a field that does not exist · **never cut**

> `Make me a pie chart of the candidates by gender`

*While it streams:*

> "It refuses, and says which fields it *can* plot.
>
> Here's why that's harder than it looks. The router's output is constrained to
> a JSON Schema, so the plan is always structurally valid — that's why there's
> no JSON-repair code anywhere in this project. But the dimension is an enum,
> and *gender is not one of its values*. The model literally cannot say
> 'gender'. So the decoder renormalises over the values that are legal and
> returns the nearest one — confidently, with no exception and no low-confidence
> signal.
>
> Measured: this exact question produced a **CV-language** chart in English and
> a **seniority** chart in Spanish, both sitting next to a perfectly correct
> text answer. The guarantee that removes the parsing code is the same mechanism
> that manufactures a plausible wrong answer.
>
> So the plan is checked against the question's own words before anything is
> plotted. A wrong chart beats no chart to the eye, which is exactly why it has
> to be the other way round."

---

## 3:00 – 4:30 · Technical highlight — the bilingual RRF bug

Have the README table on screen. This one is told, not shown.

> "The thing I'm most pleased with came out of a test I expected to pass.
>
> Hybrid retrieval fuses dense and BM25 with Reciprocal Rank Fusion, which
> rewards a document for appearing in both rankings. I asked a Spanish question
> and got only Spanish CVs back. So I measured it — and a Spanish query scores
> **zero** BM25 against every single English chunk. Obvious in hindsight:
> lexical matching cannot cross a language boundary.
>
> Which means an English chunk can only ever earn rank mass from one of the two
> retrievers. It is structurally capped at about half the fused score of a
> Spanish one, no matter how relevant it is.
>
> And it was suppressing correct answers. For the product-design question, the
> best dense match in the entire index was an English Product Designer — and he
> wasn't in the results at all. He lost to two Spanish chunks scoring 0.90 and
> 0.89 of his similarity.
>
> The fix reserves slots for other-language chunks whose dense similarity holds
> up on its own — at least 0.90 of the best in the index. A relative threshold
> inside a single metric, so it never blends the two incomparable score scales
> that RRF exists to avoid. Regression tests hold it in place.
>
> What I'd take from it: the hybrid retrieval everyone reaches for by default
> has an assumption baked in — that both retrievers can see the whole corpus. In
> a bilingual one, that's just false."

---

## 4:30 – 5:00 · Close

Switch to the **Logs** tab, now holding exactly the four demo questions.

> "Every exchange is recorded — the question, how it was routed, latency, and
> any error. Separate from the application log, because one is data and the
> other is diagnostics.
>
> Everything local. 166 tests, both pipelines resumable. The one thing I'd
> change with more time is latency — about twelve seconds an answer on a laptop
> GPU. Nothing is tied to a model, though: that selector switches it live."

---

## Cuts if running long

- **Q3 (Kubernetes)** — the counting argument can be made in one sentence over
  Q2's chart, since that is already an aggregate.
- Compress the generation section to 25 s: one PDF, not three.
- Drop the Pipeline tab walkthrough; keep the "reads only the PDFs" point, which
  is the part that shows judgement.
- **Never cut Q4 or the technical highlight.** Those two are what the AI-literacy
  and thought-process criteria are scored on.

## Do not say

- "28 CVs" — the corpus is 50.
- Any count from memory. Kubernetes is **13**, machine learning **5**, Python
  **17**, UPC graduates **6**, seniority **26 / 14 / 10**.
