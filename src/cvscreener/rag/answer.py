"""Compose grounded answers from retrieved context, streaming as they generate.

Grounding is enforced by construction rather than requested politely: the model
only ever sees the retrieved chunks, and the prompt makes "the CVs do not say"
an explicitly correct answer. That matters more than usual with a 9B model,
which will otherwise fill a gap with something plausible.

The bilingual corpus adds a second rule. Retrieval is cross-lingual, so a
Spanish question routinely lands on English CVs. The answer must come back in
the *question's* language regardless of the sources' language, which the model
does not do reliably unless told.
"""

from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from ..config import settings
from ..llm import client
from ..textutils import fold_accents
from .aggregate import AggregateResult
from .keywords import warning_sentence
from .retrieve import RetrievedChunk
from .router import CHARTABLE_DIMENSIONS

MAX_CONTEXT_CHARS = 7000

SYSTEM_ES = """\
Eres el asistente de selección de LeadTech. Respondes preguntas sobre una base \
de datos de CVs.

Reglas estrictas:
1. Usa ÚNICAMENTE la información de los fragmentos de CV proporcionados.
2. Si los fragmentos no contienen la respuesta, dilo claramente en UNA sola \
frase y NO uses el formato de lista de la regla 5: no menciones a nadie. Esta \
regla tiene prioridad sobre todas las demás. Nunca inventes candidatos, \
empresas, fechas ni tecnologías.
3. Cita siempre a los candidatos por su nombre completo.
4. Algunos CVs están en inglés: tradúcelos al responder, pero no cambies los \
nombres propios ni los nombres de tecnologías.
5. Cuando SÍ haya candidatos que respondan a la pregunta, AGRUPA POR PERSONA. \
Cada candidato aparece UNA sola vez, con su nombre como encabezado, y debajo \
sus datos. Nunca repitas un nombre en varias entradas de la lista, ni siquiera \
si tiene varios proyectos. Usa exactamente este formato:

**Nombre Completo** — su puesto o resumen en una línea
- un logro o dato concreto
- otro logro o dato concreto

**Otro Candidato** — su puesto o resumen en una línea
- un logro o dato concreto

6. Si de un candidato solo consta su puesto y ningún detalle, escribe \
únicamente su línea de encabezado. NUNCA escribas una viñeta vacía.
7. Sé conciso y concreto.
8. Responde SIEMPRE en español."""

SYSTEM_EN = """\
You are LeadTech's recruitment assistant. You answer questions about a CV \
database.

Strict rules:
1. Use ONLY the information in the provided CV excerpts.
2. If the excerpts do not contain the answer, say so plainly in ONE sentence \
and do NOT use the list format from rule 5: name nobody. This rule overrides \
every other rule. Never invent candidates, companies, dates or technologies.
3. Always refer to candidates by their full name.
4. Some CVs are written in Spanish: translate when answering, but keep proper \
nouns and technology names unchanged.
5. When candidates DO answer the question, GROUP BY PERSON. Each candidate \
appears ONCE, with their name as a heading and their details underneath. Never \
repeat a name across several list entries, not even when they have several \
relevant projects. Use exactly this format:

**Full Name** — their role or a one-line summary
- one concrete achievement or fact
- another concrete achievement or fact

**Other Candidate** — their role or a one-line summary
- one concrete achievement or fact

6. If only a candidate's role is recorded and no details, write their heading \
line alone. NEVER write an empty bullet.
7. Be concise and specific.
8. ALWAYS answer in English."""

# Function words that are distinctive to one language and common in questions.
_ES_MARKERS = (
    "quien", "quienes", "cuantos", "cuantas", "cual", "cuales", "que ", "como",
    "donde", "tiene", "sabe", "saben", "candidato", "candidatos", "experiencia",
    "anos", "los ", "las ", "del ", "una ", "por ", "para ", "con ", "muestra",
    "dame", "genera", "grafico", "histograma", "edades",
)
_EN_MARKERS = (
    "who", "how many", "which", "what", "where", "show", "list", "give me",
    "candidate", "candidates", "experience", "years", "the ", "with ", "chart",
    "graph", "histogram", "ages", "summarize", "summarise",
)


def detect_question_language(question: str) -> str:
    """Pick the reply language from the question itself.

    Detected from the question, never from the sources: retrieval is
    cross-lingual, so a Spanish question routinely lands on English CVs, and
    answering in the sources' language would be wrong. Whoever asked gets an
    answer in the language they asked in.

    Wrapped in spaces so " the " matches the word and not the tail of "clothe".
    """
    folded = f" {fold_accents(question).lower()} "
    es = sum(folded.count(m) for m in _ES_MARKERS)
    en = sum(folded.count(m) for m in _EN_MARKERS)
    if es == en:
        # Tie-break on characters only Spanish uses. Note this reads the
        # ORIGINAL string, not the accent-folded one.
        return "es" if any(c in question for c in "áéíóúñ¿¡") else "en"
    return "es" if es > en else "en"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks grouped by candidate, one numbered block each.

    Grouping is not cosmetic. Listed flat, the model mirrors the shape it is
    given and emits one bullet per fragment - measured, a question about
    machine learning returned the same candidate five separate times, once for
    his profile and once for each achievement inside a single experience
    chunk, each prefixed with his name. The reader has to reassemble the person
    from the pieces.

    So the context is shaped like the answer that is wanted: one numbered
    entry per person, their sections nested underneath. Candidates keep the
    order they were retrieved in, so the best match is still first.
    """
    # dict preserves insertion order, which is RRF order: best candidate first.
    grouped: dict[str, list[RetrievedChunk]] = {}
    for item in chunks:
        grouped.setdefault(item.chunk.candidate, []).append(item)

    parts: list[str] = []
    total = 0
    for i, (candidate, items) in enumerate(grouped.items(), 1):
        # Every block is labelled with its candidate and file. Without that the
        # model sees a wall of anonymous text and cannot attribute anything
        # correctly - or cite it.
        sections = "\n".join(
            f"  [{item.chunk.section}] {item.chunk.text}" for item in items
        )
        block = f"[{i}] Candidate: {candidate} (file: {items[0].chunk.source_file})\n{sections}"

        # Stop at the budget rather than truncating mid-block: half a CV
        # excerpt is worse than one fewer excerpt. gemma2:9b has an 8k window
        # and this keeps well inside it.
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


_EMPTY_BULLET_RE = re.compile(r"^[ \t]*[-*•]+[ \t]*$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def tidy_answer(text: str) -> str:
    """Remove bullets the model opened and had nothing to put in.

    The grouped format asks for a heading per candidate and bullets beneath.
    When a candidate was retrieved on their headline alone there is nothing to
    put in a bullet, and gemma2:9b emits an empty one anyway - the prompt says
    not to, twice, in both languages, and it does it regardless. A 9B model
    reliably follows the shape of an example and unreliably follows a
    prohibition, so this is not worth another round of prompt wording: an empty
    list item is never valid output, which makes it a job for code.

    Applied where the finished answer exists rather than mid-stream, so
    token-by-token rendering is untouched.
    """
    return _BLANK_RUN_RE.sub("\n\n", _EMPTY_BULLET_RE.sub("", text)).strip()


def _table_context(frame: pd.DataFrame, limit: int = 40) -> str:
    columns = [
        c
        for c in ("full_name", "current_role", "seniority", "years_experience", "age", "city")
        if c in frame.columns
    ]
    if frame.empty:
        return "(no candidates matched)"
    body = frame[columns].head(limit).to_string(index=False)
    if len(frame) > limit:
        # Say so explicitly. An unannounced truncation invites the model to
        # count these rows and present the total as if it had seen them all.
        body += (
            f"\n(truncated: showing {limit} of {len(frame)} rows - do not count "
            "these rows, use the computed result above)"
        )
    return body


def stream_retrieval_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    missing_terms: list[str] | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Answer from retrieved chunks.

    ``missing_terms`` lists terms from the question that appear in no CV at
    all. Dense retrieval returns the nearest neighbours regardless, so without
    this the model sees plausible-looking context and infers the missing skill
    from an adjacent one - answering a question about "CNN" with a Computer
    Vision candidate. Telling it which terms are genuinely absent turns a
    silent overclaim into an explicit "no CV mentions this".
    """
    lang = detect_question_language(question)
    if not chunks:
        yield (
            "No he encontrado ningún CV relacionado con esa pregunta."
            if lang == "es"
            else "I could not find any CV related to that question."
        )
        return

    prompt = (
        f"{'Fragmentos de CV' if lang == 'es' else 'CV excerpts'}:\n\n"
        f"{build_context(chunks)}\n\n"
        f"{'Pregunta' if lang == 'es' else 'Question'}: {question}"
        f"{warning_sentence(missing_terms or [], lang)}"
    )
    yield from client.chat_stream(
        [
            {"role": "system", "content": SYSTEM_ES if lang == "es" else SYSTEM_EN},
            {"role": "user", "content": prompt},
        ],
        model=model or settings.chat_model,
        temperature=0.15,
    )


def chart_unavailable_sentence(lang: str) -> str:
    """Instruction appended when the requested breakdown cannot be plotted.

    Phrased as an instruction to the model rather than printed verbatim so it
    lands in the same voice as the rest of the answer - but the *decision* was
    made deterministically in the router, not by the model.
    """
    fields = ", ".join(CHARTABLE_DIMENSIONS)
    if lang == "es":
        return (
            "\n\nIMPORTANTE: no se puede generar el gráfico pedido porque ese "
            "campo no está en la base de datos. Empieza la respuesta diciéndolo "
            "claramente, y añade que sí se puede representar por: "
            f"{fields}. No inventes el dato ni lo sustituyas por otro campo."
        )
    return (
        "\n\nIMPORTANT: the requested chart cannot be produced because that "
        "field is not in the database. Open your answer by saying so plainly, "
        f"and add that these fields can be charted instead: {fields}. "
        "Do not invent the data or substitute a different field."
    )


def stream_aggregate_answer(
    question: str,
    result: AggregateResult,
    *,
    chart_unavailable: bool = False,
    model: str | None = None,
) -> Iterator[str]:
    """Narrate a computed result.

    The count is already exact, so the model's only job is to phrase it. The
    number is handed over as a fact and the prompt forbids recomputing it -
    letting a 9B model do arithmetic over a table is how wrong totals happen.

    ``chart_unavailable`` says the user asked to plot a field the table does not
    hold. The text answer is still right, so it is still streamed; what changes
    is that the answer has to lead with the absence instead of quietly arriving
    next to a chart of something else.
    """
    lang = detect_question_language(question)
    frame = result.matched

    # The UI already renders every match as a citation chip (and, for charts, as
    # a figure), so making the model recite all of them costs ~30s of streaming
    # to duplicate what is on screen. Keep the prose to the headline figure.
    many = len(frame) > 6

    if lang == "es":
        system = (
            "Eres el asistente de selección de LeadTech. Te doy un resultado ya "
            "calculado sobre la base de datos completa de CVs. Redáctalo en "
            "español en 1-2 frases. La cifra es correcta: NO la recalcules ni la "
            "pongas en duda. No inventes datos."
            + (
                " NO enumeres los candidatos: la interfaz ya muestra la lista completa."
                if many
                else " Menciona a los candidatos por su nombre."
            )
        )
        prompt = (
            f"Pregunta: {question}\n\n"
            f"Resultado calculado: {result.text}\n"
            f"Candidatos que cumplen ({len(frame)}):\n{_table_context(frame)}"
            f"{chart_unavailable_sentence('es') if chart_unavailable else ''}"
        )
    else:
        system = (
            "You are LeadTech's recruitment assistant. You are given a result "
            "already computed over the complete CV database. Phrase it in 1-2 "
            "sentences. The figure is correct: do NOT recompute or question it. "
            "Invent nothing."
            + (
                " Do NOT list the candidates: the interface already shows the full list."
                if many
                else " Name the candidates."
            )
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Computed result: {result.text}\n"
            f"Matching candidates ({len(frame)}):\n{_table_context(frame)}"
            f"{chart_unavailable_sentence('en') if chart_unavailable else ''}"
        )

    yield from client.chat_stream(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model or settings.chat_model,
        temperature=0.15,
        num_predict=200 if many else 500,
    )
