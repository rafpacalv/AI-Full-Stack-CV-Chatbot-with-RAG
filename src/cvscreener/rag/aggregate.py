"""Answer counting, listing and charting questions over the whole candidate set.

Everything here runs on ``candidates.parquet`` with pandas, so the numbers are
exact and cover all 28 CVs rather than whichever handful a retriever surfaced.
The output is a small, serialisable result the API can hand to the UI: a
sentence, the matching rows, and - when a chart was requested - the data to
plot, never a rendered image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..config import settings
from ..ingest.enrich import normalise_skill
from ..textutils import fold_accents
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


def apply_filters(frame: pd.DataFrame, plan: QueryPlan) -> tuple[pd.DataFrame, dict[str, str]]:
    """Narrow the table to the rows the question is about."""
    out = frame
    used: dict[str, str] = {}

    if plan.skill:
        wanted = normalise_skill(plan.skill)
        out = out[out["skills_normalised"].apply(lambda s: wanted in list(s))]
        used["skill"] = wanted

    if plan.seniority:
        target = plan.seniority.casefold()
        out = out[out["seniority"].str.casefold() == target]
        used["seniority"] = plan.seniority

    if plan.city:
        target = fold_accents(plan.city).casefold()
        out = out[out["city"].apply(lambda c: fold_accents(str(c)).casefold() == target)]
        used["city"] = plan.city

    if plan.min_years:
        out = out[out["years_experience"] >= plan.min_years]
        used["min_years"] = f">= {plan.min_years}"

    if plan.candidate_name:
        target = fold_accents(plan.candidate_name).casefold()
        out = out[
            out["full_name"].apply(lambda n: target in fold_accents(str(n)).casefold())
        ]
        used["candidate"] = plan.candidate_name

    return out, used


def _describe(count: int, total: int, used: dict[str, str]) -> str:
    if not used:
        return f"There are {total} candidates in the database."
    criteria = ", ".join(f"{k}={v}" for k, v in used.items())
    return f"{count} of {total} candidates match {criteria}."


def _chart_payload(frame: pd.DataFrame, plan: QueryPlan) -> dict | None:
    """Shape the plot data. Returns values for a histogram, counts otherwise."""
    dim = plan.dimension
    if dim in ("none", "") or frame.empty:
        return None

    label = DIMENSION_LABELS.get(dim, dim)

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

    # Continuous fields keep their raw values so the UI can bin them properly.
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


def run_aggregate(plan: QueryPlan) -> AggregateResult:
    frame = load_candidates()
    matched, used = apply_filters(frame, plan)
    chart = _chart_payload(matched, plan) if plan.intent == "chart" else None
    return AggregateResult(
        text=_describe(len(matched), len(frame), used),
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
