"""Small text helpers shared by the generation and ingestion pipelines."""

from __future__ import annotations

import re
import unicodedata


def fold_accents(text: str) -> str:
    """'Núria Badia Roldán' -> 'Nuria Badia Roldan'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def ascii_slug(text: str, sep: str = "-") -> str:
    """Filesystem- and URL-safe slug.

    Filenames become part of API paths and citation identifiers, so they are
    kept strictly ASCII even though the CV content itself is fully Unicode.
    """
    folded = fold_accents(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", sep, folded).strip(sep)
    return cleaned or "unnamed"
