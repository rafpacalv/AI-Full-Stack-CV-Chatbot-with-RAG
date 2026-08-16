"""Derive structured facts about each candidate from the extracted PDF text.

This is what lets the assistant answer questions a top-k vector search
structurally cannot: "how many candidates know Python?", "what is the age
distribution?". Those need a table over *all* 50 candidates, not the 5 chunks
that happened to rank highest.

Extraction is deliberately hybrid:

* **Regex** for fields with a rigid surface form - e-mail, phone, date of
  birth, LinkedIn. A 9B model adds latency and error to problems ``re`` solves
  exactly.
* **LLM (schema-constrained)** for everything needing judgement - seniority,
  which strings are really skills, how many years of experience a career adds
  up to.

Skills are then normalised through an alias table, without which the aggregate
layer would count "aprendizaje automático" and "machine learning" as two
different skills and quietly halve every cross-language total.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import client
from ..textutils import fold_accents

log = logging.getLogger(__name__)

TODAY = date(2026, 8, 14)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+\d{1,3}[\s\d]{7,15}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+")
DOB_RE = re.compile(
    r"(?:fecha de nacimiento|date of birth)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{4})",
    re.IGNORECASE,
)

# Canonical skill names. Keys are folded to lowercase ASCII before lookup, so
# "Aprendizaje Automático" and "aprendizaje automatico" both resolve.
SKILL_ALIASES: dict[str, str] = {
    "aprendizaje automatico": "machine learning",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "aprendizaje profundo": "deep learning",
    "deep learning": "deep learning",
    "redes neuronales": "neural networks",
    "neural networks": "neural networks",
    "procesamiento de lenguaje natural": "nlp",
    "procesamiento del lenguaje natural": "nlp",
    "natural language processing": "nlp",
    "pnl": "nlp",
    "nlp": "nlp",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "ci/cd": "ci/cd",
    "integracion continua": "ci/cd",
    "continuous integration": "ci/cd",
    "bases de datos": "databases",
    "databases": "databases",
    "diseno de bases de datos": "database design",
    "database design": "database design",
    "pruebas unitarias": "unit testing",
    "unit testing": "unit testing",
    "testing": "testing",
    "rest api": "rest apis",
    "rest apis": "rest apis",
    "restful apis": "rest apis",
    "apis rest": "rest apis",
    "microservicios": "microservices",
    "microservices": "microservices",
    "computacion en la nube": "cloud",
    "cloud computing": "cloud",
    "analisis de datos": "data analysis",
    "data analysis": "data analysis",
    "visualizacion de datos": "data visualization",
    "data visualization": "data visualization",
    "modelado de datos": "data modelling",
    "data modelling": "data modelling",
    "data modeling": "data modelling",
    "almacenamiento de datos": "data warehousing",
    "data warehousing": "data warehousing",
    "investigacion de usuarios": "user research",
    "user research": "user research",
    "sistemas de diseno": "design systems",
    "design systems": "design systems",
    "accesibilidad": "accessibility",
    "accessibility": "accessibility",
    "posicionamiento seo": "seo",
    "seo tecnico": "seo",
    "technical seo": "seo",
    "seo": "seo",
}


class ExtractedLanguage(BaseModel):
    language: str
    level: str


class LLMFacts(BaseModel):
    """The judgement-requiring subset, produced under a JSON Schema."""

    full_name: str = Field(description="Candidate's full name as printed on the CV")
    headline: str
    current_role: str = Field(description="Their most recent job title")
    seniority: str = Field(description="One of: Junior, Mid-level, Senior, Lead")
    years_experience: int
    city: str
    country: str
    university: str = Field(description="Main university or school attended")
    degree: str
    companies: list[str]
    skills: list[str]
    tools: list[str]
    languages: list[ExtractedLanguage]
    certifications: list[str]


class CandidateFacts(BaseModel):
    """One row of the candidate table."""

    cv_id: str
    source_file: str
    cv_language: str

    full_name: str
    headline: str
    current_role: str
    seniority: str
    years_experience: int
    city: str
    country: str
    university: str
    degree: str

    email: str = ""
    phone: str = ""
    linkedin: str = ""
    birth_date: str = ""
    age: int | None = None

    companies: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    skills_normalised: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    languages: list[ExtractedLanguage] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


def normalise_skill(skill: str) -> str:
    """Map a raw skill string onto its canonical, language-neutral name."""
    cleaned = " ".join(skill.split()).strip(" .·•-")
    # Drop parenthetical qualifiers: "Python (Expertise)" -> "Python".
    cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned).strip()
    key = fold_accents(cleaned).lower()
    return SKILL_ALIASES.get(key, key)


def detect_language(text: str) -> str:
    """Cheap ES/EN detector using function words - no extra dependency.

    Counts the commonest grammatical words in each language. Crude, but on a
    document of this length it is decisive, and it avoids pulling in a whole
    language-detection library for one boolean. Only the first 2500 characters
    are needed to be sure.
    """
    sample = fold_accents(text[:2500]).lower()
    es = sum(sample.count(w) for w in (" de ", " la ", " el ", " en ", " para ", " con "))
    en = sum(sample.count(w) for w in (" the ", " and ", " of ", " for ", " with ", " to "))
    return "es" if es >= en else "en"


def _regex_fields(text: str) -> dict:
    """The half of extraction that needs no intelligence, only precision.

    These four fields have rigid surface forms. Regex gets them exactly right
    in microseconds; an LLM would be slower and occasionally wrong. Whatever is
    absent is simply omitted, and the field keeps its default.
    """
    out: dict = {}
    if m := EMAIL_RE.search(text):
        out["email"] = m.group(0)
    if m := PHONE_RE.search(text):
        out["phone"] = " ".join(m.group(0).split())  # collapse odd PDF spacing
    if m := LINKEDIN_RE.search(text):
        out["linkedin"] = m.group(0)
    if m := DOB_RE.search(text):
        day, month, year = (int(g) for g in m.groups())
        out["birth_date"] = f"{day:02d}/{month:02d}/{year}"
        # Age computed here, not asked of the model: arithmetic is not what a
        # 9B model is for. The comparison subtracts a year when the birthday
        # has not yet occurred this year.
        age = TODAY.year - year - ((TODAY.month, TODAY.day) < (month, day))
        out["age"] = age
    return out


def _prompt(text: str) -> str:
    return f"""Extract structured facts from this CV. Use ONLY what the document states.

Rules:
- "full_name": exactly as printed on the CV.
- "seniority": one of Junior, Mid-level, Senior, Lead - infer from job titles
  and total experience.
- "years_experience": total professional years, as an integer.
- "skills": the technical skills listed. Keep each one short (1-3 words) and
  do not invent any that are absent.
- "university": the main higher-education institution, with its acronym if the
  CV shows one.
- Keep every value in the language the CV is written in.

--- CV ---
{text}
--- END CV ---"""


def enrich_document(
    *, cv_id: str, source_file: str, text: str, force: bool = False
) -> CandidateFacts:
    """Extract (or load from cache) the fact row for one CV."""
    settings.ensure_dirs()

    # One cache file per CV. This is what makes re-indexing cheap: adding CVs
    # only pays the LLM cost for the new ones, at roughly 30 s each.
    cache: Path = settings.index_dir / "facts" / f"{cv_id}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force:
        return CandidateFacts.model_validate_json(cache.read_text(encoding="utf-8"))

    # The judgement half. temperature=0 because this is extraction, not
    # writing: the same CV must always yield the same facts. The schema is
    # enforced by Ollama, so the result is guaranteed to parse.
    llm_facts = client.structured(
        _prompt(text),
        LLMFacts,
        model=settings.chat_model,
        system="You are a precise information-extraction system for recruitment documents.",
        temperature=0.0,
        num_predict=1400,
    )

    # Tidy the skill strings: collapse whitespace, strip stray bullet
    # characters the model sometimes copies out of the PDF, drop blanks.
    data = llm_facts.model_dump()
    skills = [" ".join(s.split()).strip(" .·•-") for s in data.pop("skills")]
    skills = [s for s in skills if s]

    facts = CandidateFacts(
        cv_id=cv_id,
        source_file=source_file,
        cv_language=detect_language(text),
        # Both forms are kept: the raw strings for display (a Spanish CV should
        # still read "Aprendizaje Automático" in the UI), and the canonical
        # ones for counting, so ES and EN CVs aggregate together.
        skills=skills,
        skills_normalised=sorted({normalise_skill(s) for s in skills if s}),
        **data,               # the LLM's fields
        **_regex_fields(text),  # the regex fields, which win on any overlap
    )

    cache.write_text(
        json.dumps(facts.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return facts
