# Video script — under 5 minutes

Rehearse once with a timer. The three demo questions are pre-warmed by the API's
startup routine, so the first one is not slow.

**Before recording:** run `.\run.ps1`, wait for the sidebar to show `CONNECTED`,
and ask one throwaway question so both models are hot.

---

## 0:00 – 0:25 · What it is

> "An AI CV screener. 28 fake CVs in Spanish and English, and a chat that
> answers questions about them — running entirely on local models through
> Ollama. No API keys, nothing leaves the laptop."

Show the UI briefly. Point out `CONNECTED`, `208 chunks`, `bge-m3`.

---

## 0:25 – 1:15 · The generation pipeline

Show `personas.py`, then a couple of PDFs.

> "The CVs are generated in three stages. First a persona matrix — 28 personas
> with role, seniority, age, city, language and layout all fixed up front with a
> seed. That's deliberate: if you just ask a model for 28 varied CVs, you get the
> same three archetypes over and over. Diversity is engineered, then the model
> writes prose inside those constraints."

Open two or three PDFs — one Spanish sidebar, one English classic.

> "Text from gemma2, headshots from a hosted diffusion model, PDFs from
> ReportLab in three layouts. All synthetic: invented names, reserved e-mail
> domains, and every PDF says so in its metadata."

---

## 1:15 – 2:00 · The ingest and RAG workflow

Show the **Pipeline** tab.

> "Ingestion reads only the PDFs. It never touches the JSON the generator
> produced, even though that file sits right next to them with all the same data
> already parsed. If I fed that in, I'd be proving the generator can round-trip
> its own output — not that the system can read a CV."

> "Extract, chunk by section, then two indexes: dense vectors from bge-m3, which
> is multilingual, and BM25 for exact tokens. Fused with Reciprocal Rank Fusion."

---

## 2:00 – 3:15 · The demo

**Q1 — cross-lingual.** Ask in Spanish:
> `¿Quién tiene experiencia con aprendizaje automático?`

> "Asked in Spanish. Look at the sources — English CVs come back too. Nothing was
> translated; bge-m3 puts both languages in one vector space."

**Q2 — the counting question.** 
> `How many candidates know Kubernetes?`

Point at the routing badge changing to **TABLE AGGREGATE**.

> "This one didn't go to the retriever. A classic RAG pipeline retrieves five
> chunks and counts those five — with top-k, a counting question is unanswerable
> by construction. So a router sends it to pandas over all 28 candidates instead.
> Eight. That's exact, not estimated."

**Q3 — the chart.**
> `Genera un histograma de las edades de los candidatos que sepan Python`

> "Same routing idea, one step further: the model emits a query plan — filter on
> Python, dimension age, chart type histogram — pandas runs it, and the UI draws
> it."

**Q4 — grounding (fast, ~10 s).**
> `¿Quién ha trabajado en la NASA?`

> "And it says nobody has, instead of inventing someone."

---

## 3:15 – 4:40 · Technical highlight — the bilingual RRF bug

This is the part to spend real time on. Have the README table on screen.

> "The thing I'm most pleased with came out of a test I expected to pass.

> Hybrid retrieval fuses dense and BM25 with Reciprocal Rank Fusion, which
> rewards a document for showing up in both rankings. I asked a Spanish question
> and got only Spanish CVs back. So I measured it — and a Spanish query scores
> **zero** BM25 against every single English chunk. Obvious in hindsight: lexical
> matching can't cross a language boundary.

> Which means an English chunk can only ever earn rank mass from one of the two
> retrievers. It's structurally capped at half the fused score of a Spanish one,
> no matter how relevant it is.

> And it was suppressing correct answers. For the product-design question, the
> best dense match in the entire index was an English Product Designer — and he
> wasn't in the results at all. He lost to two Spanish chunks scoring 0.90 and
> 0.89 of his similarity.

> The fix reserves slots for other-language chunks whose dense similarity holds
> up on its own — at least 0.90 of the best in the index. That's a relative
> threshold inside a single metric, so it never blends the two incomparable score
> scales that RRF exists to avoid. Three regression tests hold it in place."

> "What I'd take from it: the hybrid retrieval everyone reaches for by default
> has an assumption baked in — that both retrievers can see the whole corpus.
> In a bilingual one, that's just false."

---

## 4:40 – 5:00 · Close

> "Everything local, 33 tests, resumable pipelines. The one thing I'd change with
> more time is latency — about 12 seconds an answer on a laptop GPU. The
> architecture is model-agnostic, so a bigger machine or a hosted model is a
> config change."

---

## Cuts if running long

- Drop Q4 (grounding) — mention it instead of showing it.
- Compress the pipeline section to 30 s.
- **Never cut the technical highlight.** It is what the section is scored on.
