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

from typing import Iterator

import pandas as pd

from ..config import settings
from ..llm import client
from ..textutils import fold_accents
from .aggregate import AggregateResult
from .retrieve import RetrievedChunk

MAX_CONTEXT_CHARS = 7000

SYSTEM_ES = """\
Eres el asistente de selección de LeadTech. Respondes preguntas sobre una base \
de datos de CVs.

Reglas estrictas:
1. Usa ÚNICAMENTE la información de los fragmentos de CV proporcionados.
2. Si los fragmentos no contienen la respuesta, dilo claramente. Nunca inventes \
candidatos, empresas, fechas ni tecnologías.
3. Cita siempre a los candidatos por su nombre completo.
4. Algunos CVs están en inglés: tradúcelos al responder, pero no cambies los \
nombres propios ni los nombres de tecnologías.
5. Sé conciso y concreto. Usa listas cuando compares varios candidatos.
6. Responde SIEMPRE en español."""

SYSTEM_EN = """\
You are LeadTech's recruitment assistant. You answer questions about a CV \
database.

Strict rules:
1. Use ONLY the information in the provided CV excerpts.
2. If the excerpts do not contain the answer, say so plainly. Never invent \
candidates, companies, dates or technologies.
3. Always refer to candidates by their full name.
4. Some CVs are written in Spanish: translate when answering, but keep proper \
nouns and technology names unchanged.
5. Be concise and specific. Use lists when comparing several candidates.
6. ALWAYS answer in English."""

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
    """Pick the reply language from the question itself."""
    folded = f" {fold_accents(question).lower()} "
    es = sum(folded.count(m) for m in _ES_MARKERS)
    en = sum(folded.count(m) for m in _EN_MARKERS)
    if es == en:
        # Accented characters are a strong Spanish tell when counts are level.
        return "es" if any(c in question for c in "áéíóúñ¿¡") else "en"
    return "es" if es > en else "en"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, attributable context block."""
    parts: list[str] = []
    total = 0
    for i, item in enumerate(chunks, 1):
        block = (
            f"[{i}] Candidate: {item.chunk.candidate} "
            f"(file: {item.chunk.source_file}, section: {item.chunk.section})\n"
            f"{item.chunk.text}"
        )
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _table_context(frame: pd.DataFrame, limit: int = 40) -> str:
    columns = [
        c
        for c in ("full_name", "current_role", "seniority", "years_experience", "age", "city")
        if c in frame.columns
    ]
    if frame.empty:
        return "(no candidates matched)"
    return frame[columns].head(limit).to_string(index=False)


def stream_retrieval_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
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
    )
    yield from client.chat_stream(
        [
            {"role": "system", "content": SYSTEM_ES if lang == "es" else SYSTEM_EN},
            {"role": "user", "content": prompt},
        ],
        model=settings.chat_model,
        temperature=0.15,
    )


def stream_aggregate_answer(question: str, result: AggregateResult) -> Iterator[str]:
    """Narrate a computed result.

    The count is already exact, so the model's only job is to phrase it. The
    number is handed over as a fact and the prompt forbids recomputing it -
    letting a 9B model do arithmetic over a table is how wrong totals happen.
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
        )

    yield from client.chat_stream(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=settings.chat_model,
        temperature=0.15,
        num_predict=200 if many else 500,
    )
