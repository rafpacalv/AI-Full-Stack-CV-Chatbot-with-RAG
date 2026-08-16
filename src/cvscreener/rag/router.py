"""Classify a recruiter's question into an executable query plan.

The failure this fixes is structural, not cosmetic. Ask a plain RAG pipeline
"how many candidates know Python?" and it retrieves five chunks and counts
those five - the honest answer requires scanning all 50 CVs, which top-k
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

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..config import settings
from ..ingest.enrich import normalise_skill
from ..llm import client
from ..textutils import WORD_RE, fold_accents

Intent = Literal["retrieve", "aggregate", "chart"]
ChartType = Literal["histogram", "bar", "pie", "none"]
SkillMatch = Literal["all", "any"]
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
CARRIED_FIELDS = ("skills", "seniority", "city", "university", "candidate_name", "min_years")

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


# Words that turn a list of skills into "any of these" rather than "all of them".
#
# Held to the same bar as _BACKREFERENCE_WORDS above, and it excludes more than
# it admits. A false positive here silently converts an intersection into a
# union - a bigger, plausible, wrong number - so a word earns its place only if
# it is hard to use non-disjunctively. That ruled out:
#
#   "any"        - "do you have any candidates who know Python and Docker?"
#   "cualquiera" - "cualquiera que sepa Python y Docker" is an AND
#   "alguno/a"   - "¿hay alguno que sepa Python y Docker?" likewise
#
# What survives is the bare conjunctions and "either", which are disjunctive in
# almost any sentence they appear in. "u" is included because Spanish swaps it
# for "o" before an o- sound: "Java u Oracle".
_ANY_SKILL_WORDS = frozenset("o u or either".split())


def asks_for_any_skill(question: str) -> bool:
    """Did the user write "or" between the skills they listed?

    The same deliberately dumb lexical check as :func:`refers_back`, in the same
    direction: it can only ever force "any" *on*. Its absence proves nothing,
    which is why the model gets to answer this too - like `follow_up` and unlike
    `dimension`, both values are legal tokens, so the decoder is never cornered
    into a confident wrong one.
    """
    words = set(WORD_RE.findall(fold_accents(question).casefold()))
    return bool(words & _ANY_SKILL_WORDS)


def carry_over(plan: "QueryPlan", previous: "QueryPlan") -> list[str]:
    """Inherit ``previous``'s filters onto ``plan``. Returns what was copied.

    Only fields the new question left empty are filled, so anything the user
    restates wins over anything they said before.

    The risk here is inheriting on a genuine change of subject - "who knows
    machine learning?" followed by "how many candidates are in Barcelona?"
    should not quietly answer "ML candidates in Barcelona". Two things keep
    that honest rather than silent: the aggregate answer states its own
    criteria ("N of 50 candidates match skills=machine learning, city=
    Barcelona"), and the UI shows one chip per active filter. An inherited
    filter is always on screen, and "New query" drops the lot.
    """
    inherited: list[str] = []
    for name in CARRIED_FIELDS:
        current, earlier = getattr(plan, name), getattr(previous, name)
        if not current and earlier:  # 0 and "" both mean "not set"
            setattr(plan, name, earlier)
            inherited.append(name)

    # `skill_match` travels with `skills` rather than on its own terms. It cannot
    # use the loop above: its default is "all", which is truthy, so "not current"
    # is never true and it would silently stay "all" while inheriting a list the
    # user had joined with "or" - turning their union back into an intersection
    # on the follow-up.
    if "skills" in inherited:
        plan.skill_match = previous.skill_match
    return inherited


# Separators a packed skill string can use. Applied as a safety net *after* the
# schema, because a list field does not stop the model putting "python,node.js"
# in one element of it.
#
# Note what is absent: "/". Five canonical skills contain one - ci/cd, ui/ux,
# a/b testing - so splitting on it would shatter real names. Commas, semicolons
# and the conjunctions are safe: no skill in the corpus contains any of them,
# and "y"/"and" are matched as whole words so "Ruby" survives.
_SKILL_SEPARATORS = re.compile(r"\s*(?:[,;&]|\band\b|\by\b)\s*", re.IGNORECASE)


class QueryPlan(BaseModel):
    intent: Intent = Field(description="retrieve, aggregate or chart")
    # Emitted early, before the filters, so the model commits to "this points
    # backwards" while it still has the filter fields left to leave empty.
    follow_up: bool = Field(
        default=False,
        description="True if the question is about the candidates from the previous question",
    )
    # A list, because a single string cannot express "Python and Kubernetes" and
    # the decoder will not tell you it failed. Constrained to a string field, it
    # packed both into one value - "python,kubernetes" - which then matched no
    # canonical skill and returned zero rows, indistinguishable from "nobody has
    # these". Five candidates did. Same family of failure as `dimension` below:
    # the schema guarantees the shape, never the meaning.
    skills: list[str] = Field(
        default_factory=list,
        description='Skills to filter on, e.g. ["Python","Kubernetes"]. [] if none',
    )
    # Defaults to "all" because that is what a list of skills means when the
    # question does not say otherwise - "candidates who know Python, Docker and
    # AWS" wants the people who have the lot.
    skill_match: SkillMatch = Field(
        default="all",
        description='"all" if the candidate must have every skill, "any" if one is enough',
    )
    seniority: str = Field(default="", description="Junior/Mid-level/Senior/Lead, '' if none")
    city: str = Field(default="", description="City to filter on, '' if none")
    university: str = Field(default="", description="University to filter on, '' if none")
    candidate_name: str = Field(default="", description="Specific person asked about")
    min_years: int = Field(default=0, description="Minimum years of experience, 0 if none")
    chart_type: ChartType = Field(default="none")
    dimension: Dimension = Field(default="none", description="Field to aggregate or plot")

    @field_validator("skills", mode="before")
    @classmethod
    def _unpack_skills(cls, value: object) -> list[str]:
        """Split anything the model packed into one element, and drop blanks.

        Accepts a bare string too: the schema asks for an array, but this runs on
        whatever arrives, including a previous plan replayed by a client.
        """
        items = [value] if isinstance(value, str) else list(value or [])
        out: list[str] = []
        for item in items:
            for part in _SKILL_SEPARATORS.split(str(item)):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return out


FEW_SHOT = """\
Examples:

Q: Who has experience with Python?
{"intent":"retrieve","follow_up":false,"skills":["Python"],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Q: Summarize the profile of Jane Doe.
{"intent":"retrieve","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"Jane Doe","min_years":0,"chart_type":"none","dimension":"none"}

Q: Which candidate graduated from UPC?
{"intent":"aggregate","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"UPC","candidate_name":"","min_years":0,"chart_type":"none","dimension":"university"}

Q: ¿Qué candidatos estudiaron en la UPC?
{"intent":"aggregate","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"UPC","candidate_name":"","min_years":0,"chart_type":"none","dimension":"university"}

Q: How many candidates know Kubernetes?
{"intent":"aggregate","follow_up":false,"skills":["Kubernetes"],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: Dame un listado de candidatos que sepan Python y Kubernetes.
{"intent":"aggregate","follow_up":false,"skills":["Python","Kubernetes"],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: List candidates with knowledge of Python and Node.js.
{"intent":"aggregate","follow_up":false,"skills":["Python","Node.js"],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: ¿Quién sabe Python o Kubernetes?
{"intent":"aggregate","follow_up":false,"skills":["Python","Kubernetes"],"skill_match":"any","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: Candidates who know either Java or Scala.
{"intent":"aggregate","follow_up":false,"skills":["Java","Scala"],"skill_match":"any","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"skills"}

Q: ¿Cuántos candidatos hay en Barcelona?
{"intent":"aggregate","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"Barcelona","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"city"}

Q: List all senior engineers with more than 5 years of experience.
{"intent":"aggregate","follow_up":false,"skills":[],"skill_match":"all","seniority":"Senior","city":"","university":"","candidate_name":"","min_years":5,"chart_type":"none","dimension":"seniority"}

Q: Genera un histograma de las edades de los candidatos que sepan Python.
{"intent":"chart","follow_up":false,"skills":["Python"],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"histogram","dimension":"age"}

Q: Show me a bar chart of candidates by city.
{"intent":"chart","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"bar","dimension":"city"}

Q: Reparte por seniority en un gráfico circular.
{"intent":"chart","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"seniority"}

Q: Make me a pie chart of the candidates by gender.
{"intent":"chart","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}

Q: Haz un diagrama sectorial de los candidatos por género.
{"intent":"chart","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"unsupported"}
"""

# Appended only when there is a previous turn, so single-question routing sees
# exactly the prompt it has always seen and the existing behaviour cannot drift.
# Both outcomes are shown - a follow-up and a change of subject - because an
# example set containing only follow-ups would teach the model to answer "true"
# whenever the context block is present.
FOLLOW_UP_EXAMPLES = """\
Examples with a previous question:

Previous question: Who has experience with machine learning?
Previous filters: skills=machine learning
Q: Now show me a pie chart of those candidates by seniority.
{"intent":"chart","follow_up":true,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"pie","dimension":"seniority"}

Previous question: ¿Quién tiene experiencia en aprendizaje automático?
Previous filters: skills=machine learning
Q: ¿Y cuántos de ellos son senior?
{"intent":"aggregate","follow_up":true,"skills":[],"skill_match":"all","seniority":"Senior","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"seniority"}

Previous question: Resume el perfil de Jane Doe.
Previous filters: candidate_name=Jane Doe
Q: ¿Dónde estudió?
{"intent":"retrieve","follow_up":true,"skills":[],"skill_match":"all","seniority":"","city":"","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"none"}

Previous question: Who has experience with machine learning?
Previous filters: skills=machine learning
Q: How many candidates are based in Barcelona?
{"intent":"aggregate","follow_up":false,"skills":[],"skill_match":"all","seniority":"","city":"Barcelona","university":"","candidate_name":"","min_years":0,"chart_type":"none","dimension":"city"}
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

"skills" is a list. Put each technology the question names in its own element -
["Python","Kubernetes"], never ["Python and Kubernetes"] - because they are
matched one by one against what each CV lists.

"skill_match" says how to combine them. "all" - the default - is for "Python
and Kubernetes": the candidate must have every skill listed. Use "any" only
when the user separates the skills with "or" / "o" / "either", where having one
of them is enough.

Only fill a filter field when the question actually names it. Leave the rest
empty. Reply with JSON only."""


def _filter_summary(plan: QueryPlan) -> str:
    """The plan's active filters, as the prompt shows them back to the model."""
    parts = []
    for name in CARRIED_FIELDS:
        value = getattr(plan, name)
        if not value:
            continue
        # Lists are flattened rather than repr'd: showing the model
        # "skills=['python', 'node.js']" invites it to copy the brackets into a
        # field it is meant to leave empty. The skills are joined by the operator
        # that is actually in force, so a follow-up inheriting a union is not
        # shown something that reads like an intersection.
        if isinstance(value, list):
            value = (" or " if plan.skill_match == "any" else ", ").join(value)
        parts.append(f"{name}={value}")
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
    #
    # Checked on every intent, not just `chart`. The dimension now decides which
    # exact counts the narrator is handed on an aggregate question too, and a
    # field the user never named is as wrong in a sentence as it is in a figure -
    # it just arrives without a picture drawing attention to it.
    if plan.dimension not in ("none", "unsupported"):
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

    # Fold each skill onto its canonical name so "Aprendizaje Automático" and
    # "Machine Learning" hit the same rows in the candidate table. Normalising
    # can collapse two spellings of one skill into a single name, so duplicates
    # are dropped afterwards rather than before.
    if plan.skills:
        canonical: list[str] = []
        for skill in plan.skills:
            folded = normalise_skill(skill)
            if folded and folded not in canonical:
                canonical.append(folded)
        plan.skills = canonical

    # "or" in the user's own words forces `any` on; nothing forces it off. Same
    # shape as the `follow_up` floor above, and applied only once there are two
    # skills to combine - with fewer, `skill_match` changes no result, and a
    # stray "o" in "de 30 o mas anos" would set a flag that then gets inherited
    # by a follow-up that does have two.
    if len(plan.skills) > 1 and asks_for_any_skill(question):
        plan.skill_match = "any"

    # A question naming two or more skills is a set-membership question, so it
    # belongs on the aggregate branch even when it reads like prose.
    #
    # Semantic search has no boolean logic: asked for "Python and Node.js" it
    # returns the nearest chunks, which are the candidates matching *either*
    # term, and the answer then lists them as though they matched both. Observed
    # exactly that - four names offered for an intersection that is empty. The
    # aggregate branch computes the intersection and says so when it is empty,
    # which is the honest answer to the question that was asked.
    #
    # Deliberately not extended to a single skill. That question ("who has
    # experience with Python?") is asking what people did with it, and the
    # aggregate branch would answer a tally - see
    # test_one_skill_still_reaches_the_prose. Retrieval keeps it, and is instead
    # scoped to the candidates who qualify before it searches (see api/main.py).
    if plan.intent == "retrieve" and len(plan.skills) > 1:
        plan.intent = "aggregate"
        if plan.dimension == "none":
            plan.dimension = "skills"

    return plan
