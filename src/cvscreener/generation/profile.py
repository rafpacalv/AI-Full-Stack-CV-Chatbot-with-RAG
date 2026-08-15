"""Author CV content with a local LLM, constrained to :class:`GeneratedContent`.

The model is given the persona's hard facts and asked only to write prose around
them. Identity and contact details are synthesised deterministically here.
"""

from __future__ import annotations

import json
import logging
import random
import unicodedata
from pathlib import Path

from ..config import settings
from ..llm import client
from .personas import DEGREE_FIELDS_ES, Persona
from .schema import CVProfile, GeneratedContent

log = logging.getLogger(__name__)

# RFC 2606 reserves these domains precisely so synthetic data can never collide
# with, or route to, a real mailbox.
EMAIL_DOMAINS = ["example.com", "example.org", "example.net"]


def _ascii_slug(text: str) -> str:
    """'Núria Badia' -> 'nuria.badia' (accents folded, ASCII only)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    parts = [p.lower() for p in stripped.replace("'", "").split() if p]
    return ".".join(parts[:2])


def _contact(persona: Persona) -> tuple[str, str, str]:
    """Build contact details deterministically, not with the LLM.

    Seeded on the cv_id, so the same persona always gets the same details
    across runs - regenerating one CV never silently changes its e-mail.
    """
    rng = random.Random(persona.cv_id)
    handle = _ascii_slug(persona.full_name)
    email = f"{handle}@{rng.choice(EMAIL_DOMAINS)}"
    # Country-appropriate dialling code, so a Barcelona candidate does not end
    # up with a Dutch number.
    if persona.country in ("España", "Spain"):
        phone = f"+34 6{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
    else:
        phone = f"+{rng.randint(31, 49)} {rng.randint(100, 999)} {rng.randint(100000, 999999)}"
    linkedin = f"linkedin.com/in/{handle.replace('.', '-')}"
    return email, phone, linkedin


def _birth_date(persona: Persona) -> str:
    """A date of birth consistent with the persona's birth year.

    Capped at day 28 so no persona is ever born on 30 February. The year comes
    from the matrix, so the age the RAG later extracts matches by construction.
    """
    rng = random.Random(persona.cv_id + "dob")
    day, month = rng.randint(1, 28), rng.randint(1, 12)
    return f"{day:02d}/{month:02d}/{persona.birth_year}"


def _expected_jobs(years: int) -> int:
    """How many roles a career of this length should show.

    Left to itself the model gives a junior four jobs and a lead one, which
    reads wrong immediately. This is also reused after generation to trim any
    duplicates the model produced.
    """
    if years <= 2:
        return 1
    if years <= 5:
        return 2
    if years <= 10:
        return 3
    return 4


def _prompt(persona: Persona) -> tuple[str, str]:
    """Return (system, user) prompts in the persona's own language."""
    jobs = _expected_jobs(persona.years_experience)
    skills = ", ".join(persona.signature_skills)
    role = persona.role_display
    start_year = 2026 - persona.years_experience

    if persona.language == "es":
        field = DEGREE_FIELDS_ES.get(persona.degree_field, persona.degree_field)
        system = (
            "Eres un redactor profesional de currículums en España. Escribes CVs "
            "creíbles, concretos y con logros cuantificados. Escribes ÚNICAMENTE "
            "en español de España, sin mezclar palabras en inglés salvo nombres "
            "de tecnologías."
        )
        user = f"""Redacta el contenido del CV de esta persona ficticia:

- Nombre: {persona.full_name}
- Puesto actual: {role} ({persona.seniority})
- Años de experiencia: {persona.years_experience} (empezó a trabajar en {start_year})
- Ciudad: {persona.city}, {persona.country}
- Universidad: {persona.university}
- Titulación: Grado en {field}
- Tecnologías que domina: {skills}

Requisitos:
- "headline": un TITULAR PROFESIONAL, nunca el nombre de la persona.
  Formato: "<puesto> | <especialidad 1> y <especialidad 2>".
  Ejemplo: "Ingeniera Backend | Python, Django y arquitecturas distribuidas".
- "summary": 3-4 frases en primera persona implícita, sin repetir el titular
  literalmente y sin frases hechas tipo "buscando un nuevo desafío".
- Exactamente {jobs} puesto(s) en "experience", en orden cronológico inverso.
  El más reciente termina en "Actualidad". Los periodos deben cubrir desde
  {start_year} hasta 2026 y NO solaparse.
- Empresas ficticias pero verosímiles del sector tecnológico español o europeo.
- 2-4 viñetas por puesto, cada una con un logro concreto y una métrica
  (porcentajes, tiempos, volúmenes, usuarios).
- 1-2 titulaciones en "education", la principal el Grado en {field}
  por {persona.university}.
- 8-12 "technical_skills" ESPECÍFICAS del puesto de {role}, incluyendo
  {skills}. No incluyas habilidades genéricas irrelevantes para el puesto.
- 4-6 "tools" (herramientas y plataformas concretas).
- "languages": español nativo, inglés con nivel realista, y a veces un tercero.
- 0-3 certificaciones creíbles.
- TODO el texto en español.
"""
    else:
        system = (
            "You are a professional CV writer. You write believable, specific "
            "résumés with quantified achievements. You write ONLY in English."
        )
        user = f"""Write the CV content for this fictional person:

- Name: {persona.full_name}
- Current role: {role} ({persona.seniority})
- Years of experience: {persona.years_experience} (started working in {start_year})
- City: {persona.city}, {persona.country}
- University: {persona.university}
- Degree: BSc/MSc in {persona.degree_field}
- Core technologies: {skills}

Requirements:
- "headline": a PROFESSIONAL TITLE LINE, never the person's name.
  Format: "<role> | <specialism 1> and <specialism 2>".
  Example: "Backend Engineer | Java, Kafka and event-driven systems".
- "summary": 3-4 sentences, not a restatement of the headline, and no filler
  phrases such as "seeking a new challenge".
- Exactly {jobs} position(s) in "experience", reverse-chronological.
  The most recent ends in "Present". Periods must span {start_year} to 2026
  and must NOT overlap.
- Fictional but plausible European tech companies.
- 2-4 bullets per position, each a concrete achievement with a metric
  (percentages, latencies, volumes, users).
- 1-2 entries in "education", the main one being a Bachelor's degree in
  {persona.degree_field} at {persona.university}. Write the degree out in full
  (e.g. "BSc Computer Engineering"); never write "BSc/MSc".
- 8-12 "technical_skills" SPECIFIC to a {role}, including {skills}.
  Do not pad with generic skills irrelevant to the role.
- 4-6 "tools" (concrete platforms and tooling).
- "languages": English plus realistic others.
- 0-3 believable certifications.
- ALL text in English.
"""
    return system, user


def _dedupe_strings(values: list[str]) -> list[str]:
    """Strip, drop blanks, and remove case-insensitive duplicates in order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = " ".join(raw.split())
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _clean(content: GeneratedContent, persona: Persona) -> GeneratedContent:
    """Repair the failure modes local models reliably exhibit here.

    Constrained decoding guarantees the JSON *shape*, not its sanity: a 9B model
    will happily emit the same job twice, or pad a skills list with duplicates
    and stray whitespace. Cheap to fix deterministically, so we do.
    """
    # Drop repeated jobs, keyed on (company, position). This is a real observed
    # failure: one generated CV listed the same two roles twice, producing four
    # entries for a four-year career.
    seen: set[tuple[str, str]] = set()
    jobs: list = []
    for item in content.experience:
        key = (item.company.strip().casefold(), item.position.strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        item.bullets = _dedupe_strings(item.bullets)[:4]
        jobs.append(item)

    # Enforce the caps the prompt asked for. Asking is not the same as getting.
    content.experience = jobs[: _expected_jobs(persona.years_experience)]
    content.technical_skills = _dedupe_strings(content.technical_skills)[:12]
    content.tools = _dedupe_strings(content.tools)[:6]
    content.certifications = _dedupe_strings(content.certifications)[:3]
    content.headline = " ".join(content.headline.split())
    content.summary = " ".join(content.summary.split())

    # A headline that merely echoes the name is the one failure the layout
    # cannot absorb, so fall back to something sensible.
    if content.headline.casefold().startswith(persona.full_name.split()[0].casefold()):
        content.headline = f"{persona.role_display}"

    return content


def generate_profile(persona: Persona, *, force: bool = False) -> CVProfile:
    """Build (or load from cache) the full profile for a persona."""
    settings.ensure_dirs()
    cache: Path = settings.profiles_dir / f"{persona.cv_id}.json"

    if cache.exists() and not force:
        return CVProfile.model_validate_json(cache.read_text(encoding="utf-8"))

    system, user = _prompt(persona)
    content = client.structured(
        user,
        GeneratedContent,
        model=settings.gen_model,
        system=system,
        # Deliberately high. This is the one place in the project that wants
        # variety rather than determinism: at a low temperature every CV comes
        # out in the same voice, with the same phrasing and the same invented
        # company names. The persona matrix supplies the structure; temperature
        # supplies the prose variation.
        temperature=0.85,
        num_predict=2600,
    )
    content = _clean(content, persona)

    email, phone, linkedin = _contact(persona)
    profile = CVProfile(
        cv_id=persona.cv_id,
        language=persona.language,
        full_name=persona.full_name,
        email=email,
        phone=phone,
        city=persona.city,
        country=persona.country,
        birth_date=_birth_date(persona),
        linkedin=linkedin,
        template=persona.template,
        **content.model_dump(),
    )

    cache.write_text(
        json.dumps(profile.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return profile
