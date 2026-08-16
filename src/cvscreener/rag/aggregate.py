"""Answer counting, listing and charting questions over the whole candidate set.

Everything here runs on ``candidates.parquet`` with pandas, so the numbers are
exact and cover all 50 CVs rather than whichever handful a retriever surfaced.
The output is a small, serialisable result the API can hand to the UI: a
sentence, the matching rows, and - when a chart was requested - the data to
plot, never a rendered image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..config import settings
from ..ingest.enrich import normalise_skill
from ..textutils import WORD_RE, fold_accents
from .router import QueryPlan


class TableNotBuilt(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_candidates() -> pd.DataFrame:
    path = settings.index_dir / "candidates.parquet"
    if not path.exists():
        raise TableNotBuilt(
            f"{path} missing. Run: python -m cvscreener.ingest.index"
        )
    return pd.read_parquet(path)


@dataclass
class AggregateResult:
    text: str
    matched: pd.DataFrame
    filters: dict[str, str] = field(default_factory=dict)
    chart: dict | None = None

    @property
    def cv_ids(self) -> list[str]:
        return self.matched["cv_id"].tolist()


# Human-readable labels for the dimensions we can group by.
DIMENSION_LABELS = {
    "age": "Age",
    "years_experience": "Years of experience",
    "seniority": "Seniority",
    "city": "City",
    "current_role": "Role",
    "cv_language": "CV language",
    "university": "University",
    "skills": "Skill",
}


def _university_keys(value: str) -> set[str]:
    """Every string this university could reasonably be called.

    Needed because the extractor read each CV independently and normalised
    inconsistently: most rows came back as an acronym (``UPC``, ``TCD``) but
    some kept the full name (``Technische Universität Berlin``). A recruiter
    may type either form, so both sides of the comparison are expanded into a
    set of keys and matched on overlap.

    The keys are the folded full string, the initials of its significant words
    (``Universitat Politècnica de Catalunya`` -> ``upc``), and anything in
    parentheses (``Danmarks Tekniske Universitet (DTU)`` -> ``dtu``).
    """
    text = str(value)
    folded = fold_accents(text).casefold().strip()
    keys = {folded}

    # Initials of the words longer than two letters, which skips the
    # "de"/"of"/"the" that acronyms leave out. Case is ignored so a question
    # typed in lower case still resolves. This is what unifies the rows the
    # extractor spelled out with the rows it abbreviated: "Universitat de
    # Barcelona" and "UB" both produce the key "ub".
    initials = "".join(w[0] for w in WORD_RE.findall(text) if len(w) > 2)
    if len(initials) >= 2:
        keys.add(initials.casefold())

    keys.update(part.strip() for part in re.findall(r"\(([^)]+)\)", folded))
    return {k for k in keys if k}


def _is_spelled_out(folded: str) -> bool:
    """Long enough to be a name rather than an acronym.

    Six characters, because the longest acronym in this corpus is four (URLL)
    and the shortest spelled-out name is six (Deusto). Below that, substring
    matching does real damage: "UB" is inside "TUB", and "US" is inside half
    the corpus.
    """
    return len(folded) >= 6


def _university_matches(query: str, stored: str) -> bool:
    query_keys, stored_keys = _university_keys(query), _university_keys(stored)
    if query_keys & stored_keys:
        return True

    # Fall back to substring, but only between two spelled-out names. Letting
    # acronyms take this path makes "UB" match "TUB", and "US" (Universidad de
    # Sevilla) a substring of half the corpus - both measured, both wrong.
    left = fold_accents(str(query)).casefold().strip()
    right = fold_accents(str(stored)).casefold().strip()
    if not (_is_spelled_out(left) and _is_spelled_out(right)):
        return False
    return left in right or right in left


def name_matches(query: str, stored: str) -> bool:
    """Is ``query`` a way of referring to the candidate called ``stored``?

    Substring on folded, case-insensitive text, because people ask for
    "Wilczynska", not the full legal name, and usually without the diacritics.

    One definition, used twice: the table filter below narrows rows with it, and
    the retrieval branch of the API resolves "summarise X" to a cv_id with it.
    Those were two independent copies of the same three lines, which is exactly
    how "the aggregate found her but the search did not" starts.
    """
    return fold_accents(str(query)).casefold() in fold_accents(str(stored)).casefold()


def cv_ids_for_name(name: str) -> list[str]:
    """Every CV the name could refer to; empty when nobody matches.

    Empty is meaningful to the caller: the model may have misread a name that is
    not in the corpus at all, and searching everything beats searching nothing.
    """
    frame = load_candidates()
    return frame[frame["full_name"].apply(lambda n: name_matches(name, n))]["cv_id"].tolist()


def apply_filters(frame: pd.DataFrame, plan: QueryPlan) -> tuple[pd.DataFrame, dict[str, str]]:
    """Narrow the table to the rows the question is about."""
    # Filters are applied in sequence, each narrowing the previous result.
    # `used` records what actually fired, so the answer can state its criteria
    # and the UI can show them as chips - the user always sees what was applied.
    out = frame
    used: dict[str, str] = {}

    # Skills live in a list column, so this is a membership test per row, not an
    # equality test. Both sides go through normalise_skill so a Spanish query
    # term matches the canonical name stored at ingest time.
    if plan.skill:
        wanted = normalise_skill(plan.skill)
        out = out[out["skills_normalised"].apply(lambda s: wanted in list(s))]
        used["skill"] = wanted

    if plan.seniority:
        target = plan.seniority.casefold()
        out = out[out["seniority"].str.casefold() == target]
        used["seniority"] = plan.seniority

    # Accents are folded on both sides so "Malaga" finds "Málaga".
    if plan.city:
        target = fold_accents(plan.city).casefold()
        out = out[out["city"].apply(lambda c: fold_accents(str(c)).casefold() == target)]
        used["city"] = plan.city

    # The brief's own sample question is "Which candidate graduated from UPC?".
    # Six of the 50 did, so a top-k=5 search cannot answer it completely - it is
    # a set-membership question over a recorded field, which is what this path
    # is for.
    if plan.university:
        out = out[
            out["university"].apply(lambda u: _university_matches(plan.university, u))
        ]
        used["university"] = plan.university

    if plan.min_years:
        out = out[out["years_experience"] >= plan.min_years]
        used["min_years"] = f">= {plan.min_years}"

    if plan.candidate_name:
        out = out[out["full_name"].apply(lambda n: name_matches(plan.candidate_name, n))]
        used["candidate"] = plan.candidate_name

    return out, used


def _describe(count: int, total: int, used: dict[str, str]) -> str:
    if not used:
        return f"There are {total} candidates in the database."
    criteria = ", ".join(f"{k}={v}" for k, v in used.items())
    return f"{count} of {total} candidates match {criteria}."


def _chart_payload(frame: pd.DataFrame, plan: QueryPlan) -> dict | None:
    """Shape the plot data.

    Returns plain JSON-serialisable data, never a rendered image. The backend
    decides *what* the chart says; the UI decides how it looks. That keeps the
    brand palette and Plotly entirely on the front end, and means the same
    payload could feed a different renderer without touching this file.
    """
    dim = plan.dimension
    # "unsupported" means the question asked to break the candidates down by a
    # field this table does not have (gender, salary, nationality...). The text
    # answer is still computed and still correct; only the figure is withheld,
    # because the alternative - plotting a different field - is what produced a
    # confident, wrong pie chart next to a right answer. See router.py.
    if dim in ("none", "", "unsupported") or frame.empty:
        return None

    label = DIMENSION_LABELS.get(dim, dim)

    # Skills need exploding first: one row per candidate becomes one row per
    # (candidate, skill) pair, so they can be counted individually.
    if dim == "skills":
        exploded = frame.explode("skills_normalised")["skills_normalised"].dropna()
        counts = exploded.value_counts().head(15)
        return {
            "type": "bar",
            "dimension": dim,
            "label": label,
            "categories": counts.index.tolist(),
            "values": [int(v) for v in counts.tolist()],
        }

    if dim not in frame.columns:
        return None

    series = frame[dim].dropna()
    if series.empty:
        return None

    # A histogram needs the raw values, not counts: binning is a presentation
    # decision, so the UI does it. Everything else gets pre-counted below.
    if plan.chart_type == "histogram" and pd.api.types.is_numeric_dtype(series):
        return {
            "type": "histogram",
            "dimension": dim,
            "label": label,
            "values": [float(v) for v in series.tolist()],
            "names": frame["full_name"].tolist(),
        }

    counts = series.astype(str).value_counts()
    return {
        "type": "pie" if plan.chart_type == "pie" else "bar",
        "dimension": dim,
        "label": label,
        "categories": counts.index.tolist(),
        "values": [int(v) for v in counts.tolist()],
    }


def _chart_summary(chart: dict | None) -> str:
    """State the plotted numbers exactly, for the narrator to quote.

    Without this a chart question hands the model only "there are 50
    candidates" plus a truncated table, and asks it to describe a distribution -
    so it counts the rows itself and gets it wrong. Measured: asked to split 50
    candidates by seniority it answered 40 % Junior / 50 % Mid-level / 10 %
    Senior against a true 20 / 28 / 52, near enough to inverted.

    The chart payload already holds the exact counts, so they are handed over as
    a fact. Same rule as the count itself: the model phrases arithmetic, it
    never performs it.
    """
    if not chart:
        return ""
    label = chart.get("label", chart.get("dimension", ""))

    if chart["type"] == "histogram":
        values = chart.get("values") or []
        if not values:
            return ""
        average = sum(values) / len(values)
        return (
            f" {label} across those {len(values)} candidates: "
            f"minimum {min(values):g}, maximum {max(values):g}, mean {average:.1f}."
        )

    pairs = ", ".join(
        f"{category} {value}" for category, value in zip(chart["categories"], chart["values"])
    )
    total = sum(chart["values"])
    return f" Exact breakdown by {label.lower()} (total {total}): {pairs}."


def run_aggregate(plan: QueryPlan) -> AggregateResult:
    frame = load_candidates()
    matched, used = apply_filters(frame, plan)
    chart = _chart_payload(matched, plan) if plan.intent == "chart" else None
    return AggregateResult(
        text=_describe(len(matched), len(frame), used) + _chart_summary(chart),
        matched=matched,
        filters=used,
        chart=chart,
    )


def candidates_summary() -> list[dict]:
    """Compact candidate list for the UI's browser tab."""
    frame = load_candidates()
    columns = [
        "cv_id", "full_name", "current_role", "seniority", "years_experience",
        "age", "city", "country", "university", "cv_language", "source_file",
    ]
    present = [c for c in columns if c in frame.columns]
    records = frame[present].to_dict("records")
    for row, skills in zip(records, frame["skills_normalised"]):
        row["skills"] = list(skills)
    return records
