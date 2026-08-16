"""Classify a recruiter's question into an executable query plan.

The failure this fixes is structural, not cosmetic. Ask a plain RAG pipeline
"how many candidates know Python?" and it retrieves five chunks and counts
those five - the honest answer requires scanning all 28 CVs, which top-k
retrieval cannot do by construction. So questions are routed:

``retrieve``   semantic question -> hybrid search -> grounded answer
``aggregate``  counting/listing  -> pandas over the full candidate table
``chart``      same, plus a plot

Ollama constrains the output to :class:`QueryPlan`'s JSON Schema, so the shape
is guaranteed. The *semantics* are not: an early test had gemma2:9b label
"genera un histograma de las edades..." as ``retrieve``. Hence the few-shot
block below - it is load-bearing, not decoration.

Routing is also where a conversation becomes more than a list of unrelated
questions. Ask "who has machine learning experience?" and then "now chart those
candidates by seniority", and the second question names no criteria of its own:
routed alone it plots all 50 CVs. The previous turn's plan is therefore passed
back in, and the filters it resolved are inherited by the follow-up. See
:func:`carry_over`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..ingest.enrich import normalise_skill
from ..llm import client
from ..textutils import WORD_RE, fold_accents

Intent = Literal["retrieve", "aggregate", "chart"]
ChartType = Literal["histogram", "bar", "pie", "none"]
Dimension = Literal[
    "age", "years_experience", "seniority", "city", "current_role",
    "cv_language", "skills", "university", "unsupported", "none",
]

# Words that must appear in the question for a dimension to be believable.
#
# These exist because of how constrained decoding fails. `Dimension` is an enum
# in the JSON Schema, so Ollama will only ever emit one of its members - which
# is exactly why "make me a pie chart of candidates by gender" is dangerous.
# The model cannot answer "gender": it is not a legal token. Instead the decoder
# renormalises over the values that ARE legal and returns the nearest one, with
# no exception and no low-confidence signal. Measured on this corpus, that
# question yielded `cv_language` in English and `seniority` in Spanish - a
# confident, well-formed, completely wrong plot beside a correct text answer.
#
# The schema guarantee that removes all JSON-repair code is the same mechanism
# that manufactures this, so the plan has to be checked against the question
# rather than trusted. Accents are folded before matching, hence "anos".
DIMENSION_CUES: dict[str, tuple[str, ...]] = {
    "age": ("age", "ages", "edad", "edades", "old", "nacim", "born"),
    "years_experience": (
        "experience", "experiencia", "years", "anos", "antiguedad", "trayectoria",
    ),
    "seniority": ("seniority", "senior", "junior", "mid-level", "lead", "nivel", "categoria"),
    "city": ("city", "cities", "ciudad", "ciudades", "location", "ubicacion", "donde", "where"),
    "current_role": (
        "role", "roles", "rol", "position", "puesto", "cargo", "perfil",
        "disciplina", "job title", "area",
    ),
    "cv_language": (
        "language", "languages", "idioma", "idiomas", "lengua",
        "english", "spanish", "ingles", "espanol",
    ),
    "skills": (
        "skill", "technolog", "tecnolog", "habilidad", "conocimiento",
        "stack", "herramienta", "tool",
    ),
    "university": (
        "universit", "universidad", "college", "school", "alma mater",
        "estudios", "titulacion", "degree", "carrera",
    ),
}

# What the UI can offer instead, in the order a person would expect to read it.
CHARTABLE_DIMENSIONS = tuple(DIMENSION_CUES)


def dimension_supported_by(question: str, dimension: str) -> bool:
    """Is ``dimension`` actually mentioned in ``question``?

    A deliberately dumb lexical check. It cannot be fooled the way the model
    can, because it only ever confirms what the user literally wrote - and the
    failure it guards against is precisely the model inventing a field the user
    never named.
    """
    cues = DIMENSION_CUES.get(dimension)
    if not cues:
        return False
    folded = fold_accents(question).casefold()
    return any(cue in folded for cue in cues)


# --------------------------------------------------------------------------
# Follow-up questions
# --------------------------------------------------------------------------
# Which fields a follow-up inherits from the turn before it.
#
# Filters say *who* the question is about; `intent`, `chart_type` and
# `dimension` say *what to do with them*. Only the first group is carried over,
# because a follow-up always supplies its own verb - "now chart them by
# seniority" - and inheriting `chart` would replot on every question after the
# first one.
#
# Note what is inherited: the previous *filter*, not the previous *answer*.
# That matters. The first question may have been semantic, so its answer came
# from the five chunks that happened to rank; re-running the filter instead
# means "those candidates" resolves against the whole table. The follow-up can
# therefore cover more people than the answer it refers to - which is the
# correct reading of the question, not a bug.
CARRIED_FIELDS = ("skill", "seniority", "city", "university", "candidate_name", "min_years")

# Words that only make sense pointing at something already said.
#
# Matched as whole words, never as substrings: "esas" sits inside "empresas",
# and a question about companies is not a follow-up.
#
# The list is deliberately short. Since a false positive answers a fresh
# question about the wrong people, a word earns its place only if it is hard to
# use non-anaphorically - which ruled out several obvious candidates:
# "previo/previa" ("experiencia previa en Kubernetes"), "above" ("candidates
# above 30"), and "they"/"their" ("candidates and where they studied"). Those
# cases are left to the model's own judgement, which has the sentence.
_BACKREFERENCE_WORDS = frozenset(
    """
    anterior anteriores
    esos esas estos estas aquellos aquellas dichos dichas
    mismos mismas ellos ellas
    those these them previous same latter aforementioned
    """.split()
)


def refers_back(question: str) -> bool:
    """Does ``question`` point at candidates named in an earlier turn?

    The same deliberately dumb lexical check as :func:`dimension_supported_by`,
    used the same way: to confirm what the user literally wrote. It can only
    ever force a follow-up *on*. Its absence proves nothing - "¿y por
    seniority?" is a follow-up with no such word in it - so the model's own
    judgement is what covers that case.
    """
    words = set(WORD_RE.findall(fold_accents(question).casefold()))
    return bool(words & _BACKREFERENCE_WORDS)


def carry_over(plan: "QueryPlan", previous: "QueryPlan") -> list[str]:
    """Inherit ``previous``'s filters onto ``plan``. Returns what was copied.

    Only fields the new question left empty are filled, so anything the user
    restates wins over anything they said before.

    The risk here is inheriting on a genuine change of subject - "who knows
    machine learning?" followed by "how many candidates are in Barcelona?"
    should not quietly answer "ML candidates in Barcelona". Two things keep
    that honest rather than silent: the aggregate answer states its own
    criteria ("N of 50 candidates match skill=machine learning, city=
    Barcelona"), and the UI shows one chip per active filter. An inherited
    filter is always on screen, and "New query" drops the lot.
    """
    inherited: list[str] = []
    for name in CARRIED_FIELDS:
        current, earlier = getattr(plan, name), getattr(previous, name)
        if not current and earlier:  # 0 and "" both mean "not set"
            setattr(plan, name, earlier)
            inherited.append(name)
    return inherited


class QueryPlan(BaseModel):
    intent: Intent = Field(description="retrieve, aggregate or chart")
    # Emitted early, before the filters, so the model commits to "this points
    # backwards" while it still has the filter fields left to leave empty.
    follow_up: bool = Field(
        default=False,
        description="True if the question is about the candidates from the previous question",
    )
    skill: str = Field(default="", description="Skill to filter on, '' if none")
    seniority: str = Field(default="", description="Junior/Mid-level/Senior/Lead, '' if none")
    city: str = Field(default="", description="City to filter on, '' if none")
    university: str = Field(default="", description="University to filter on, '' if none")
    candidate_name: str = Field(default="", description="Specific person asked about")
    min_years: int = Field(default=0, description="Minimum years of experience, 0 if none")
    chart_type: ChartType = Field(default="none")
    dimension: Dimension = Field(default="none", description="Field to aggregate or plot")


FEW_SHOT = """\
Examples:

Q: Who has experience with Python?
{"intent":"retrieve","follow_up":false,"skill":"Python","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Q: Summarize the profile of Jane Doe.
{"intent":"retrieve","follow_up":false,"skill":"","seniority":"","city":"","university":"","candidate_name":"Jane Doe","min_years":0,"chart_type":"none","dimension":"none"}

Q: Which candidate graduated from UPC?
{"intent":"aggregate","follow_up":false,"skill":"","seniority":"","city":"","university":"UPC","candidate_name":"","min_years":0,"chart_type":"none","dimension":"university"}

Q: ¿Qué candidatos estudiaron en la UPC?
{"intent":"aggregate","follow_up":false,"skill":"","seniority":"","city":"","university":"UPC","candidate_name":"","min_years":0,"chart_type":"none","dimension":"university"}

Q: How many candidates know Kubernetes?
{"intent":"aggregate","follow_up":false,"skill":"Kubernetes","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: ¿Cuántos candidatos hay en Barcelona?
{"intent":"aggregate","follow_up":false,"skill":"","seniority":"","city":"Barcelona","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"city"}

Q: List all senior engineers with more than 5 years of experience.
{"intent":"aggregate","follow_up":false,"skill":"","seniority":"Senior","city":"","university":"","candidate_name":"","min_years":5,"chart_type":"none","dimension":"seniority"}

Q: Genera un histograma de las edades de los candidatos que sepan Python.
{"intent":"chart","follow_up":false,"skill":"Python","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"histogram","dimension":"age"}

Q: Show me a bar chart of candidates by city.
{"intent":"chart","follow_up":false,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"bar","dimension":"city"}

Q: Reparte por seniority en un gráfico circular.
{"intent":"chart","follow_up":false,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"seniority"}

Q: Make me a pie chart of the candidates by gender.
{"intent":"chart","follow_up":false,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}

Q: Haz un diagrama sectorial de los candidatos por género.
{"intent":"chart","follow_up":false,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}
"""

# Appended only when there is a previous turn, so single-question routing sees
# exactly the prompt it has always seen and the existing behaviour cannot drift.
# Both outcomes are shown - a follow-up and a change of subject - because an
# example set containing only follow-ups would teach the model to answer "true"
# whenever the context block is present.
FOLLOW_UP_EXAMPLES = """\
Examples with a previous question:

Previous question: Who has experience with machine learning?
Previous filters: skill=machine learning
Q: Now show me a pie chart of those candidates by seniority.
{"intent":"chart","follow_up":true,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"seniority"}

Previous question: ¿Quién tiene experiencia en aprendizaje automático?
Previous filters: skill=machine learning
Q: ¿Y cuántos de ellos son senior?
{"intent":"aggregate","follow_up":true,"skill":"","seniority":"Senior","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"seniority"}

Previous question: Resume el perfil de Jane Doe.
Previous filters: candidate_name=Jane Doe
Q: ¿Dónde estudió?
{"intent":"retrieve","follow_up":true,"skill":"","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Previous question: Who has experience with machine learning?
Previous filters: skill=machine learning
Q: How many candidates are based in Barcelona?
{"intent":"aggregate","follow_up":false,"skill":"","seniority":"","city":"Barcelona","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"city"}
"""

FOLLOW_UP_RULES = """\

The user may be following up on their previous question. Set "follow_up" to
true when this question is about the same people - "those candidates", "de
ellos", "¿y por ciudad?", or any question that names no criteria of its own.
Set it to false when the user has moved to a new subject.

When "follow_up" is true, leave the previous question's filters empty: they are
inherited automatically. Only fill a filter this question adds or changes."""

SYSTEM = """\
You route recruiter questions about a CV database into a query plan.

- "retrieve": the answer needs to read the prose of specific CVs - what someone
  did, how they describe their experience, a summary of one person.
- "aggregate": the answer is a count, a total, or the complete set of people
  matching a recorded field - city, university, seniority, years of experience.
  "How many", "cuántos", "list all", "todos los que", and also "which
  candidates studied at X" or "who is based in Y": naming only the handful that
  happen to surface in a search would be an incomplete answer to those.
- "chart": the user explicitly asks to see a plot, chart, graph, histogram,
  "gráfico", "histograma", "distribución", "visualiza", "represéntame".

"dimension" is the field to plot, and it must be the one the user actually
named. The database only holds: age, years_experience, seniority, city,
current_role, cv_language, skills, university. If the user asks to break the
candidates down by anything else - gender, salary, nationality, availability -
answer "unsupported". Never substitute a different field for the one requested.

Only fill a filter field when the question actually names it. Leave the rest
empty. Reply with JSON only."""


def _filter_summary(plan: QueryPlan) -> str:
    """The plan's active filters, as the prompt shows them back to the model."""
    parts = [
        f"{name}={getattr(plan, name)}" for name in CARRIED_FIELDS if getattr(plan, name)
    ]
    return ", ".join(parts) or "none"


def route(
    question: str,
    *,
    model: str | None = None,
    previous: QueryPlan | None = None,
    previous_question: str = "",
) -> QueryPlan:
    """Classify ``question``; degrade to plain retrieval if anything is off.

    ``previous`` is the *resolved* plan of the turn before this one - already
    carrying anything it inherited itself, so a chain of follow-ups accumulates
    without this function needing to remember more than one step. The caller
    supplies it, which keeps the API stateless: there is no session to expire,
    and clearing the conversation in the UI is simply not sending it.
    """
    prompt = f"{FEW_SHOT}\nQ: {question}\n"
    system = SYSTEM
    if previous is not None:
        prompt = (
            f"{FEW_SHOT}\n{FOLLOW_UP_EXAMPLES}\n"
            f"Previous question: {previous_question}\n"
            f"Previous filters: {_filter_summary(previous)}\n"
            f"Q: {question}\n"
        )
        system = SYSTEM + FOLLOW_UP_RULES

    try:
        plan = client.structured(
            prompt,
            QueryPlan,
            model=model or settings.chat_model,
            system=system,
            temperature=0.0,
            num_predict=220,
        )
    except Exception:  # noqa: BLE001 - routing must never break the chat
        return QueryPlan(intent="retrieve")

    # --- Repair the plan before anyone acts on it -------------------------
    # `intent` and `chart_type` are two fields the model can set independently,
    # which means it can set them to contradict each other. Rather than trusting
    # it, make them consistent here.

    # Asked for a chart but did not say which kind: pick from the data type.
    # Continuous quantities want a histogram, categories want bars.
    if plan.intent == "chart" and plan.chart_type == "none":
        plan.chart_type = "histogram" if plan.dimension in ("age", "years_experience") else "bar"

    # Named a chart type without being asked for a chart: drop it. Showing a
    # plot nobody requested is worse than showing none.
    if plan.intent != "chart":
        plan.chart_type = "none"

    # Verify the chosen dimension against the question's own words. The prompt
    # and the few-shot examples above ask the model to answer "unsupported",
    # but a constrained decoder can always pick a legal-and-wrong value, so the
    # instruction is a hint and this is the guarantee. Plotting the wrong field
    # is worse than plotting nothing: the figure looks authoritative, sits next
    # to a correct text answer, and nothing about it says "this is not what you
    # asked for".
    if plan.intent == "chart" and plan.dimension not in ("none", "unsupported"):
        if not dimension_supported_by(question, plan.dimension):
            plan.dimension = "unsupported"

    # --- Resolve the question against the conversation --------------------
    # A back-reference in the user's own words forces the flag on; nothing
    # forces it off. Unlike `dimension`, this is a field the model *can* answer
    # correctly - `follow_up` is a boolean, so both truths are legal tokens and
    # the constrained decoder is never cornered into a confident wrong value the
    # way it was with "gender". So here the model is the recall, and the lexical
    # check is only a floor under it.
    if previous is not None:
        if plan.follow_up or refers_back(question):
            plan.follow_up = True
            carry_over(plan, previous)
        else:
            plan.follow_up = False
    else:
        plan.follow_up = False

    # Fold the skill onto its canonical name so "Aprendizaje Automático" and
    # "Machine Learning" hit the same rows in the candidate table.
    if plan.skill:
        plan.skill = normalise_skill(plan.skill)
    return plan
