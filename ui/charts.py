"""Plotly figures for the Insights tab, themed with the validated palette.

The colours come from :mod:`cvscreener.branding`, whose chart sequence is a
re-stepped version of the LeadTech brand palette that passes the six-check
colour validator (lightness band, chroma floor, CVD separation, normal-vision
floor, contrast). See the comment block in that module for why the raw brand
hues could not be used directly.

Charts here are single-series by nature (a distribution, a set of counts), so
they take slot 0 - mint - and identity never rests on colour alone: every mark
is labelled on its axis.
"""

from __future__ import annotations

import plotly.graph_objects as go

from cvscreener.branding import (
    CHART_AXIS,
    CHART_GRID,
    CHART_SEQUENCE,
    CHART_SURFACE,
    CHART_TEXT,
    CHART_TEXT_MUTED,
    FONT_BODY,
)

PRIMARY = CHART_SEQUENCE[0]


def _base_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": CHART_TEXT}},
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font={"family": f"{FONT_BODY}, sans-serif", "color": CHART_TEXT, "size": 12},
        margin={"l": 56, "r": 24, "t": 52, "b": 56},
        showlegend=False,  # single series: the title names it
        hoverlabel={
            "bgcolor": "#211B33",
            "bordercolor": PRIMARY,
            "font": {"color": CHART_TEXT, "family": f"{FONT_BODY}, sans-serif"},
        },
        bargap=0.22,
        height=380,
    )
    axis = {
        "gridcolor": CHART_GRID,
        "linecolor": CHART_AXIS,
        "zeroline": False,
        "tickfont": {"color": CHART_TEXT_MUTED, "size": 11},
        "title_font": {"color": CHART_TEXT_MUTED, "size": 12},
    }
    fig.update_xaxes(title_text=x_title, showgrid=False, **axis)
    fig.update_yaxes(title_text=y_title, showgrid=True, **axis)
    return fig


def histogram(values: list[float], label: str, names: list[str] | None = None) -> go.Figure:
    """Distribution of a continuous field (ages, years of experience)."""
    span = max(values) - min(values) if values else 0
    nbins = max(4, min(12, int(span) or 4))
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=nbins,
            marker={
                "color": PRIMARY,
                # A 2px surface-coloured gap between bars, per the mark spec.
                "line": {"color": CHART_SURFACE, "width": 2},
            },
            hovertemplate=f"{label}: %{{x}}<br>Candidates: %{{y}}<extra></extra>",
        )
    )
    return _base_layout(fig, f"Distribution of {label.lower()}", label, "Candidates")


def bar(categories: list[str], values: list[int], label: str) -> go.Figure:
    """Counts per category, sorted so the ranking is readable."""
    pairs = sorted(zip(categories, values), key=lambda p: -p[1])
    cats = [str(c) for c, _ in pairs]
    vals = [v for _, v in pairs]

    fig = go.Figure(
        go.Bar(
            x=cats,
            y=vals,
            marker={"color": PRIMARY, "line": {"color": CHART_SURFACE, "width": 2}},
            text=vals,  # direct labels: the value is never colour-only
            textposition="outside",
            textfont={"color": CHART_TEXT, "size": 11},
            hovertemplate=f"{label}: %{{x}}<br>Candidates: %{{y}}<extra></extra>",
        )
    )
    fig = _base_layout(fig, f"Candidates by {label.lower()}", label, "Candidates")
    fig.update_yaxes(range=[0, max(vals) * 1.18 if vals else 1])
    if max((len(c) for c in cats), default=0) > 10:
        fig.update_xaxes(tickangle=-30)
    return fig


def pie(categories: list[str], values: list[int], label: str) -> go.Figure:
    """Composition across a small set of categories."""
    fig = go.Figure(
        go.Pie(
            labels=[str(c) for c in categories],
            values=values,
            hole=0.5,
            marker={
                "colors": CHART_SEQUENCE[: len(categories)],
                "line": {"color": CHART_SURFACE, "width": 2},
            },
            textinfo="label+value",
            textfont={"color": CHART_TEXT, "size": 12},
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            sort=True,
        )
    )
    fig.update_layout(
        title={"text": f"Candidates by {label.lower()}", "font": {"size": 15, "color": CHART_TEXT}},
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font={"family": f"{FONT_BODY}, sans-serif", "color": CHART_TEXT, "size": 12},
        margin={"l": 24, "r": 24, "t": 52, "b": 24},
        showlegend=True,
        legend={"font": {"color": CHART_TEXT_MUTED, "size": 11}},
        height=380,
    )
    return fig


def from_spec(spec: dict) -> go.Figure | None:
    """Build the figure described by an aggregate result's chart payload."""
    if not spec:
        return None
    kind = spec.get("type")
    label = spec.get("label", spec.get("dimension", ""))

    if kind == "histogram":
        return histogram(spec.get("values", []), label, spec.get("names"))
    if kind == "pie":
        return pie(spec.get("categories", []), spec.get("values", []), label)
    if kind == "bar":
        return bar(spec.get("categories", []), spec.get("values", []), label)
    return None
