"""Pydantic schema for a generated CV.

This doubles as the JSON Schema handed to Ollama for constrained decoding, so
the model physically cannot emit a malformed profile.

Note the split of responsibilities: identity fields (name, contact, birth date,
city) are filled deterministically from the persona matrix, never by the model.
That saves tokens and removes an entire class of bug where the model quietly
renames the candidate halfway through the document.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperienceItem(BaseModel):
    company: str
    position: str
    period: str = Field(description="e.g. 'Mar 2021 - Present' or 'Ene 2019 - Feb 2021'")
    location: str
    bullets: list[str] = Field(description="2 to 4 achievement bullets")


class EducationItem(BaseModel):
    degree: str
    institution: str
    period: str
    detail: str


class LanguageItem(BaseModel):
    language: str
    level: str


class GeneratedContent(BaseModel):
    """The portion of a CV the LLM is responsible for authoring."""

    headline: str = Field(description="Short professional title line")
    summary: str = Field(description="3-4 sentence professional profile")
    experience: list[ExperienceItem]
    education: list[EducationItem]
    technical_skills: list[str]
    tools: list[str]
    languages: list[LanguageItem]
    certifications: list[str]


class CVProfile(BaseModel):
    """A complete CV: deterministic identity + LLM-authored content."""

    cv_id: str
    language: str
    full_name: str
    email: str
    phone: str
    city: str
    country: str
    birth_date: str
    linkedin: str
    template: int
    photo_file: str | None = None

    headline: str
    summary: str
    experience: list[ExperienceItem]
    education: list[EducationItem]
    technical_skills: list[str]
    tools: list[str]
    languages: list[LanguageItem]
    certifications: list[str]

    @property
    def initials(self) -> str:
        parts = [p for p in self.full_name.split() if p]
        return "".join(p[0].upper() for p in parts[:2])
