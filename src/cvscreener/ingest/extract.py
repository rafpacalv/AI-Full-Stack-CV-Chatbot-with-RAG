"""Extract raw text from CV PDFs.

The ingestion pipeline deliberately reads only the PDFs. It never opens
``data/profiles/*.json``, even though those files sit right next to them and
contain the same information already parsed.

That constraint is the whole point: if the retrieval layer were fed the
generator's own structured output, the demo would prove nothing about handling
real documents. Everything downstream is derived from text that came out of a
PDF, exactly as it would be for a CV a candidate actually sent in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    cv_id: str
    source_file: str
    text: str
    n_pages: int

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _tidy(text: str) -> str:
    """Normalise whitespace without destroying line structure.

    Line breaks matter here: the chunker uses them to spot section headings,
    so this collapses runs of spaces and blank lines but keeps single newlines.
    """
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


# A gutter narrower than this is just word spacing, not a column boundary.
MIN_GUTTER_PT = 8.0
# Each side must hold at least this share of the page's words for the split to
# be believable; otherwise a stray indent would masquerade as a column.
MIN_COLUMN_SHARE = 0.12
LINE_TOLERANCE_PT = 3.0


def _find_gutter(words: list[dict], page_width: float) -> float | None:
    """Locate a vertical whitespace corridor separating two text columns.

    pdfplumber reads strictly by vertical position across the full page width,
    so a CV with a sidebar comes out interleaved - a contact detail landing in
    the middle of a sentence from the main column. That wrecks both the
    chunking and the embeddings.

    Rather than hard-coding the geometry of our own templates (which would only
    work on CVs we generated ourselves), find the corridor empirically: scan
    candidate x positions and keep those no word crosses.
    """
    if not words:
        return None

    step = 2
    candidates = range(int(page_width * 0.18), int(page_width * 0.62), step)
    clear = [x for x in candidates if not any(w["x0"] < x < w["x1"] for w in words)]
    if not clear:
        return None

    # Collapse consecutive clear positions into bands, keep the widest.
    bands: list[tuple[int, int]] = []
    start = prev = clear[0]
    for x in clear[1:]:
        if x - prev <= step:
            prev = x
        else:
            bands.append((start, prev))
            start = prev = x
    bands.append((start, prev))

    low, high = max(bands, key=lambda b: b[1] - b[0])
    if high - low < MIN_GUTTER_PT:
        return None

    gutter = (low + high) / 2
    left = sum(1 for w in words if w["x1"] <= gutter)
    right = sum(1 for w in words if w["x0"] >= gutter)
    total = len(words)
    if min(left, right) / total < MIN_COLUMN_SHARE:
        return None
    return gutter


def _words_to_text(words: list[dict]) -> str:
    """Rebuild reading-order text from positioned words."""
    if not words:
        return ""
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(word["top"] - rows[-1][0]["top"]) <= LINE_TOLERANCE_PT:
            rows[-1].append(word)
        else:
            rows.append([word])
    return "\n".join(
        " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"])) for row in rows
    )


def _page_text(page) -> str:
    """Extract one page, splitting columns first when the layout has them."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    gutter = _find_gutter(words, page.width)
    if gutter is None:
        return page.extract_text() or ""

    left = [w for w in words if w["x1"] <= gutter]
    right = [w for w in words if w["x0"] >= gutter]
    # Sidebar first, then the main column: each stays internally coherent.
    return f"{_words_to_text(left)}\n{_words_to_text(right)}"


def extract_pdf(path: Path) -> ExtractedDocument:
    with pdfplumber.open(path) as pdf:
        pages = [_page_text(page) for page in pdf.pages]
        n_pages = len(pdf.pages)

    # cv_07_omar-benali.pdf -> cv_07
    cv_id = "_".join(path.stem.split("_")[:2])
    return ExtractedDocument(
        cv_id=cv_id,
        source_file=path.name,
        text=_tidy("\n".join(pages)),
        n_pages=n_pages,
    )


def extract_all(directory: Path | None = None) -> list[ExtractedDocument]:
    directory = directory or settings.cvs_dir
    docs: list[ExtractedDocument] = []
    for path in sorted(directory.glob("cv_*.pdf")):
        try:
            doc = extract_pdf(path)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop ingestion
            log.error("failed to extract %s: %s", path.name, exc)
            continue
        if doc.n_chars < 200:
            log.warning("%s produced only %d chars; skipping", path.name, doc.n_chars)
            continue
        docs.append(doc)
    return docs
