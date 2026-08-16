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
from ..textutils import WORD_RE, fold_accents
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
    # `university` is here because it was asked for and could not be answered:
    # "dame el nombre de las universidades de los candidatos" reached the model
    # with a table holding names, roles and cities and no university anywhere in
    # it, so the only honest reply available was a non-answer. A field the router
    # can filter on has to be a field the narrator can see.
    columns = [
        c
        for c in (
            "full_name", "current_role", "seniority",
            "years_experience", "age", "city", "university",
        )
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


def _previous_turn_line(previous_question: str, lang: str) -> str:
    """Show the model what a follow-up is following up on.

    Only the question, never the previous answer. Feeding an answer back in
    would let one turn's mistake become the next turn's evidence, and the
    grounding rule - answer only from the CV excerpts - has to stay true of
    every turn independently. The question is safe: it is the user's own words,
    and all it does is resolve "those" to something nameable.
    """
    if not previous_question:
        return ""
    if lang == "es":
        return (
            f"Pregunta anterior del usuario: {previous_question}\n"
            "La pregunta siguiente se refiere a esos mismos candidatos.\n"
        )
    return (
        f"The user's previous question: {previous_question}\n"
        "The question below refers to those same candidates.\n"
    )


def stream_retrieval_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    missing_terms: list[str] | None = None,
    previous_question: str = "",
    model: str | None = None,
) -> Iterator[str]:
    """Answer from retrieved chunks.

    ``missing_terms`` lists terms from the question that appear in no CV at
    all. Dense retrieval returns the nearest neighbours regardless, so without
    this the model sees plausible-looking context and infers the missing skill
    from an adjacent one - answering a question about "CNN" with a Computer
    Vision candidate. Telling it which terms are genuinely absent turns a
    silent overclaim into an explicit "no CV mentions this".

    ``previous_question`` is set only when the router resolved this as a
    follow-up, and exists so a question like "¿y dónde estudió?" has a subject.
    """
    # From the current question, not the conversation: the reply goes back in
    # the language just used, so switching language mid-conversation works.
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
        f"{_previous_turn_line(previous_question, lang)}"
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


# Words that ask for the breakdown laid out as a table rather than a sentence.
#
# The same deliberately dumb lexical check used everywhere else here: it only
# confirms what the user literally wrote. Kept to the noun, because that is the
# only form that reliably means "lay it out in rows" - and the cost of a miss is
# a comma-separated list holding the same numbers, not a wrong answer.
_TABLE_WORDS = frozenset("tabla tablas table tables tabular tabulate".split())


def asks_for_a_table(question: str) -> bool:
    words = set(WORD_RE.findall(fold_accents(question).casefold()))
    return bool(words & _TABLE_WORDS)


def _artifact_rule(chart_included: bool, lang: str) -> str:
    """Tell the model what is on screen beside its words.

    Two observed failures, one cause - the prompt never said what accompanied
    the answer, so the model guessed:

      "se ha creado una tabla con las universidades"  (no table exists)
      "te recomiendo crear un gráfico de barras..."   (the bar chart was
                                                       already rendered above it)

    The second is the worse one: the figure the user asked for was sitting right
    there, and the prose told them to go and make it.

    Note what the chart wording does *not* invite. Told merely to describe the
    figure, gemma2 wrote "las barras están ordenadas alfabéticamente" - they are
    ordered by frequency. It cannot see the plot, so it furnishes the parts it
    was not given. It is pointed at the numbers instead, which it has.
    """
    if lang == "es":
        if chart_included:
            return (
                " Junto a tu respuesta se muestra ya el gráfico pedido: NO le "
                "sugieras al usuario que lo cree. Coméntalo a partir de las cifras "
                "exactas que se te dan, no de su aspecto: no puedes verlo, así que "
                "no describas colores, ejes ni en qué orden salen las barras."
            )
        return (
            " Tu respuesta es solo texto. No digas que has creado o adjuntado "
            "tablas, gráficos o ficheros."
        )
    if chart_included:
        return (
            " The chart that was asked for is already displayed alongside your "
            "answer: do NOT suggest the user create it. Comment on it from the "
            "exact figures you are given, not from its appearance: you cannot see "
            "it, so do not describe colours, axes or the order of the bars."
        )
    return (
        " Your answer is text only. Do not claim to have created or attached "
        "tables, charts or files."
    )


def stream_aggregate_answer(
    question: str,
    result: AggregateResult,
    *,
    chart_unavailable: bool = False,
    previous_question: str = "",
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

    # A breakdown is the answer, not a supporting detail: asked to name the
    # universities, "there are 50 candidates from 27 universities" is still a
    # refusal. So when one is present the no-enumerating rule is lifted for it -
    # it still applies to the candidate list, which is on screen either way.
    #
    # Not when a chart carries it, though. The figure already lists every value,
    # so asking for them in prose as well sets "list all of it" against "comment
    # on the chart" and the model has to pick one.
    breakdown = result.breakdown if result.chart is None else None

    # Asked for a table, lay it out as one. Only meaningful alongside a
    # breakdown: there is nothing to tabulate in a bare count.
    wants_table = bool(breakdown) and asks_for_a_table(question)

    if lang == "es":
        system = (
            "Eres el asistente de selección de LeadTech. Te doy un resultado ya "
            "calculado sobre la base de datos completa de CVs. Responde SIEMPRE "
            "en español: los datos que te paso están en inglés, pero tu respuesta "
            "no. La cifra es correcta: NO la recalcules ni la pongas en duda. No "
            "inventes datos."
            + (
                " Presenta el desglose exacto que se te da: es la respuesta a la "
                "pregunta, así que enuméralo entero y no lo resumas."
                + (
                    " Formatéalo como tabla markdown de dos columnas, con una fila "
                    "por valor."
                    if wants_table
                    else ""
                )
                if breakdown
                else " Redáctalo en 1-2 frases."
            )
            + (
                " NO enumeres los candidatos uno a uno: la interfaz ya muestra la "
                "lista completa."
                if many
                else " Menciona a los candidatos por su nombre."
            )
            + _artifact_rule(result.chart is not None, "es")
        )
        prompt = (
            f"{_previous_turn_line(previous_question, 'es')}"
            f"Pregunta: {question}\n\n"
            f"Resultado calculado: {result.text}\n"
            f"Candidatos que cumplen ({len(frame)}):\n{_table_context(frame)}"
            f"{chart_unavailable_sentence('es') if chart_unavailable else ''}"
        )
    else:
        system = (
            "You are LeadTech's recruitment assistant. You are given a result "
            "already computed over the complete CV database. The figure is "
            "correct: do NOT recompute or question it. Invent nothing."
            + (
                " Present the exact breakdown you are given: it is the answer to "
                "the question, so list all of it and do not summarise it."
                + (
                    " Format it as a two-column markdown table, one row per value."
                    if wants_table
                    else ""
                )
                if breakdown
                else " Phrase it in 1-2 sentences."
            )
            + (
                " Do NOT list the candidates one by one: the interface already "
                "shows the full list."
                if many
                else " Name the candidates."
            )
            + _artifact_rule(result.chart is not None, "en")
        )
        prompt = (
            f"{_previous_turn_line(previous_question, 'en')}"
            f"Question: {question}\n\n"
            f"Computed result: {result.text}\n"
            f"Matching candidates ({len(frame)}):\n{_table_context(frame)}"
            f"{chart_unavailable_sentence('en') if chart_unavailable else ''}"
        )

    yield from client.chat_stream(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model or settings.chat_model,
        temperature=0.15,
        num_predict=500 if (breakdown or not many) else 200,
    )
