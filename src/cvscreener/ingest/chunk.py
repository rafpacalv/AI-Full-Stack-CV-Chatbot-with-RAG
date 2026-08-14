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

    # Headings in these layouts are set in caps; require that plus a known name.
    letters = [c for c in stripped if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.85:
        return None

    key = re.sub(r"[^a-z ]", "", _fold(stripped).lower()).strip()
    return SECTION_ALIASES.get(key)


def _fold(text: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break an over-long section on paragraph, then sentence, boundaries."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    buffer = ""
    # Bullet lines and blank lines are the natural seams inside a section.
    for piece in re.split(r"\n(?=[•\-•]|\n)", text):
        if len(buffer) + len(piece) + 1 <= limit:
            buffer = f"{buffer}\n{piece}".strip()
            continue
        if buffer:
            parts.append(buffer)
        if len(piece) <= limit:
            buffer = piece.strip()
        else:
            sentences = re.split(r"(?<=[.!?])\s+", piece)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) + 1 <= limit:
                    buffer = f"{buffer} {sentence}".strip()
                else:
                    if buffer:
                        parts.append(buffer)
                    buffer = sentence.strip()
    if buffer:
        parts.append(buffer)
    return [p for p in parts if p]


def chunk_document(
    *, cv_id: str, source_file: str, text: str, candidate: str, metadata: dict | None = None
) -> list[Chunk]:
    """Split one CV into section-scoped chunks."""
    metadata = metadata or {}
    sections: list[tuple[str, list[str]]] = [("Encabezado", [])]

    for line in text.split("\n"):
        section = _normalise_heading(line)
        if section:
            sections.append((section, []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for section, lines in sections:
        body = "\n".join(lines).strip()
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
