"""Small text helpers shared by generation, ingestion and query time.

Deliberately the only module that imports ``unicodedata``. Accent folding is
applied on both sides of every comparison in this project - at index time and
again at query time - so two subtly different implementations would mean the
retriever stops matching what the ingester stored. A test enforces the rule.
"""

from __future__ import annotations

import re
import unicodedata


# Runs of letters, in any alphabet, excluding digits and underscores.
#
# `\w` would match those two as well, which is wrong for both callers: the
# router matches whole words so "esas" cannot be found inside "empresas", and
# the university matcher takes word initials, where a digit would corrupt the
# acronym.
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


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
