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

Intent = Literal["retrieve", "aggregate", "chart"]
ChartType = Literal["histogram", "bar", "pie", "none"]
Dimension = Literal[
    "age", "years_experience", "seniority", "city", "current_role",
    "cv_language", "skills", "university", "none",
]


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
"""

SYSTEM = """\
You route recruiter questions about a CV database into a query plan.

- "retrieve": the answer needs the text of specific CVs (who, what, describe,
  summarise, which candidate...).
- "aggregate": the answer is a count, a total or an exhaustive list across ALL
  candidates ("how many", "cuántos", "list all", "todos los que").
- "chart": the user explicitly asks to see a plot, chart, graph, histogram,
  "gráfico", "histograma", "distribución", "visualiza", "represéntame".

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

    # Fold the skill onto its canonical name so "Aprendizaje Automático" and
    # "Machine Learning" hit the same rows in the candidate table.
    if plan.skill:
        plan.skill = normalise_skill(plan.skill)
    return plan
