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
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings
from ..ingest.enrich import normalise_skill
from ..llm import client
from ..textutils import fold_accents

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


class QueryPlan(BaseModel):
    intent: Intent = Field(description="retrieve, aggregate or chart")
    skill: str = Field(default="", description="Skill to filter on, '' if none")
    seniority: str = Field(default="", description="Junior/Mid-level/Senior/Lead, '' if none")
    city: str = Field(default="", description="City to filter on, '' if none")
    candidate_name: str = Field(default="", description="Specific person asked about")
    min_years: int = Field(default=0, description="Minimum years of experience, 0 if none")
    chart_type: ChartType = Field(default="none")
    dimension: Dimension = Field(default="none", description="Field to aggregate or plot")


FEW_SHOT = """\
Examples:

Q: Who has experience with Python?
{"intent":"retrieve","skill":"Python","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Q: Summarize the profile of Jane Doe.
{"intent":"retrieve","skill":"","seniority":"","city":"","candidate_name":"Jane Doe","min_years":0,"chart_type":"none","dimension":"none"}

Q: ¿Qué candidatos estudiaron en la UPC?
{"intent":"retrieve","skill":"","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Q: How many candidates know Kubernetes?
{"intent":"aggregate","skill":"Kubernetes","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: ¿Cuántos candidatos hay en Barcelona?
{"intent":"aggregate","skill":"","seniority":"","city":"Barcelona","candidate_name":"","min_years":0,"chart_type":"none","dimension":"city"}

Q: List all senior engineers with more than 5 years of experience.
{"intent":"aggregate","skill":"","seniority":"Senior","city":"","candidate_name":"","min_years":5,"chart_type":"none","dimension":"seniority"}

Q: Genera un histograma de las edades de los candidatos que sepan Python.
{"intent":"chart","skill":"Python","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"histogram","dimension":"age"}

Q: Show me a bar chart of candidates by city.
{"intent":"chart","skill":"","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"bar","dimension":"city"}

Q: Reparte por seniority en un gráfico circular.
{"intent":"chart","skill":"","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"seniority"}

Q: Make me a pie chart of the candidates by gender.
{"intent":"chart","skill":"","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}

Q: Haz un diagrama sectorial de los candidatos por género.
{"intent":"chart","skill":"","seniority":"","city":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}
"""

SYSTEM = """\
You route recruiter questions about a CV database into a query plan.

- "retrieve": the answer needs the text of specific CVs (who, what, describe,
  summarise, which candidate...).
- "aggregate": the answer is a count, a total or an exhaustive list across ALL
  candidates ("how many", "cuántos", "list all", "todos los que").
- "chart": the user explicitly asks to see a plot, chart, graph, histogram,
  "gráfico", "histograma", "distribución", "visualiza", "represéntame".

"dimension" is the field to plot, and it must be the one the user actually
named. The database only holds: age, years_experience, seniority, city,
current_role, cv_language, skills, university. If the user asks to break the
candidates down by anything else - gender, salary, nationality, availability -
answer "unsupported". Never substitute a different field for the one requested.

Only fill a filter field when the question actually names it. Leave the rest
empty. Reply with JSON only."""


def route(question: str) -> QueryPlan:
    """Classify ``question``; degrade to plain retrieval if anything is off."""
    try:
        plan = client.structured(
            f"{FEW_SHOT}\nQ: {question}\n",
            QueryPlan,
            model=settings.chat_model,
            system=SYSTEM,
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

    # Fold the skill onto its canonical name so "Aprendizaje Automático" and
    # "Machine Learning" hit the same rows in the candidate table.
    if plan.skill:
        plan.skill = normalise_skill(plan.skill)
    return plan
