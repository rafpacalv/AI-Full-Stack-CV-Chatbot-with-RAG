"""Split extracted CV text into retrievable, self-describing chunks.

Two choices drive the design:

**Section-aware, not fixed-window.** A CV is already structured; slicing it
every N characters cuts job entries in half and glues the tail of one role onto
the head of the next. Splitting on section headings keeps each chunk about one
thing, which is what makes the retrieved context readable to the LLM.

**Every chunk names its owner.** A chunk retrieved in isolation is useless if it
says "Led the migration to Kubernetes" without saying whose CV that is - the
model would happily attribute it to the wrong candidate. Prefixing each chunk
with "<Name> — <Section>" also gives the embedding a strong subject signal and
lets BM25 match on the candidate's name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..textutils import fold_accents

# Canonical section names, keyed by the headings our two languages produce.
SECTION_ALIASES: dict[str, str] = {
    # Spanish
    "perfil profesional": "Perfil",
    "perfil": "Perfil",
    "experiencia profesional": "Experiencia",
    "experiencia": "Experiencia",
    "formacion academica": "Formación",
    "formacion": "Formación",
    "competencias tecnicas": "Competencias",
    "herramientas": "Herramientas",
    "idiomas": "Idiomas",
    "certificaciones": "Certificaciones",
    "contacto": "Contacto",
    # English
    "professional profile": "Profile",
    "profile": "Profile",
    "work experience": "Experience",
    "experience": "Experience",
    "education": "Education",
    "technical skills": "Skills",
    "skills": "Skills",
    "tools": "Tools",
    "languages": "Languages",
    "certifications": "Certifications",
    "contact": "Contact",
}

MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    chunk_id: str
    cv_id: str
    source_file: str
    candidate: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded and indexed."""
        return f"{self.candidate} — {self.section}\n{self.text}"


def _normalise_heading(line: str) -> str | None:
    """Return the canonical section name if ``line`` is a heading."""
    stripped = line.strip().rstrip(":")
    if not (2 < len(stripped) < 46):
        return None

    # Two independent signals must agree before a line is called a heading,
    # because a false positive silently truncates the section above it.
    # Signal 1: it is set in capitals (>=85%, tolerating "y" or accents).
    letters = [c for c in stripped if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.85:
        return None

    # Signal 2: it is a heading name we recognise. Accents and punctuation are
    # stripped first so "FORMACIÓN ACADÉMICA" matches the plain-ASCII key.
    key = re.sub(r"[^a-z ]", "", fold_accents(stripped).lower()).strip()
    return SECTION_ALIASES.get(key)


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break an over-long section on paragraph, then sentence, boundaries.

    Only reached when one section (usually a long work history) exceeds the
    limit on its own. It splits at the least damaging seam available: bullet
    boundaries first, and only if a single bullet is still too big does it fall
    back to sentence boundaries. It never cuts mid-sentence.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    buffer = ""  # the chunk being accumulated

    # Split at bullet starts and blank lines - the natural seams in a section.
    for piece in re.split(r"\n(?=[•\-•]|\n)", text):
        # Still room in the current chunk: keep filling it.
        if len(buffer) + len(piece) + 1 <= limit:
            buffer = f"{buffer}\n{piece}".strip()
            continue

        # No room. Bank what we have and start fresh.
        if buffer:
            parts.append(buffer)

        if len(piece) <= limit:
            buffer = piece.strip()
        else:
            # One bullet alone exceeds the limit: drop to sentence level.
            sentences = re.split(r"(?<=[.!?])\s+", piece)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) + 1 <= limit:
                    buffer = f"{buffer} {sentence}".strip()
                else:
                    if buffer:
                        parts.append(buffer)
                    buffer = sentence.strip()

    if buffer:  # whatever is left over
        parts.append(buffer)
    return [p for p in parts if p]


def chunk_document(
    *, cv_id: str, source_file: str, text: str, candidate: str, metadata: dict | None = None
) -> list[Chunk]:
    """Split one CV into section-scoped chunks."""
    metadata = metadata or {}

    # PASS 1 - walk the lines and cut a new section at every heading.
    # Everything before the first heading (name, job title, contact details) is
    # real content too, so it starts in a bucket of its own rather than being
    # discarded.
    sections: list[tuple[str, list[str]]] = [("Encabezado", [])]
    for line in text.split("\n"):
        section = _normalise_heading(line)
        if section:
            sections.append((section, []))  # heading: open a new section
        else:
            sections[-1][1].append(line)    # body: add to the current one

    # PASS 2 - turn each section into one or more chunks.
    chunks: list[Chunk] = []
    for section, lines in sections:
        body = "\n".join(lines).strip()
        # Skip near-empty sections: a lone heading embeds to noise.
        if len(body) < MIN_CHUNK_CHARS:
            continue
        for piece in _split_long(body):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{cv_id}#{len(chunks):02d}",
                    cv_id=cv_id,
                    source_file=source_file,
                    candidate=candidate,
                    section=section,
                    text=piece,
                    metadata=metadata,
                )
            )
    return chunks
