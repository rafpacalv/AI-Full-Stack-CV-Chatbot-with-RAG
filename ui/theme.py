"""LeadTech look-and-feel for the Streamlit app.

All colour values come from :mod:`cvscreener.branding`, which was itself
extracted from leadtech.com's production CSS - so the UI stays on-brand from a
single source rather than from hex codes sprinkled through the markup.
"""

from __future__ import annotations

import streamlit as st

from cvscreener.branding import (
    AMBER,
    FONT_BODY,
    FONT_HEADING,
    GREY_LINE,
    GREY_MUTED,
    GREY_TEXT,
    INK,
    MINT,
    PURPLE,
    SLATE,
    google_fonts_url,
)

CARD = "#1E1636"
CARD_RAISED = "#251C40"


def inject_css() -> None:
    """Inject the LeadTech stylesheet.

    Written as one blank-line-free block on purpose: a blank line inside a raw
    HTML block terminates it under CommonMark, after which Streamlit renders the
    remaining CSS to the page as literal text.
    """
    rules = [
        f'html, body, [class*="css"] {{ font-family: "{FONT_BODY}", -apple-system, sans-serif; }}',
        f".stApp {{ background: {INK}; }}",
        f'h1, h2, h3, h4, .lt-brand {{ font-family: "{FONT_HEADING}", "{FONT_BODY}", sans-serif !important; letter-spacing: -0.01em; }}',
        # masthead
        f".lt-header {{ display: flex; align-items: baseline; gap: 14px; padding: 4px 0 14px 0; border-bottom: 1px solid {GREY_LINE}; margin-bottom: 18px; }}",
        f".lt-brand {{ font-size: 30px; font-weight: 700; color: {MINT}; line-height: 1; }}",
        f".lt-tagline {{ color: {GREY_TEXT}; font-size: 14px; }}",
        # tabs
        f'.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {GREY_LINE}; }}',
        f'.stTabs [data-baseweb="tab"] {{ background: transparent; color: {GREY_MUTED}; border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500; }}',
        f'.stTabs [aria-selected="true"] {{ background: {CARD}; color: {MINT} !important; border-bottom: 2px solid {MINT}; }}',
        # chat bubbles
        f".lt-msg-user {{ background: {CARD_RAISED}; border-left: 3px solid {MINT}; padding: 12px 16px; border-radius: 4px 12px 12px 4px; margin: 6px 0 14px 0; color: #F2F1F6; }}",
        f".lt-msg-bot {{ background: {CARD}; border-left: 3px solid {PURPLE}; padding: 14px 18px; border-radius: 4px 12px 12px 4px; margin: 6px 0 10px 0; color: #EDEBF2; line-height: 1.62; }}",
        # citation chips
        f".lt-chip {{ display: inline-block; background: rgba(0,255,198,0.10); border: 1px solid rgba(0,255,198,0.38); color: {MINT}; padding: 3px 11px; border-radius: 999px; font-size: 12px; margin: 3px 5px 3px 0; font-weight: 500; }}",
        f".lt-chip-muted {{ display: inline-block; background: rgba(255,255,255,0.05); border: 1px solid {GREY_LINE}; color: {GREY_TEXT}; padding: 3px 11px; border-radius: 999px; font-size: 12px; margin: 3px 5px 3px 0; }}",
        # routing badge
        ".lt-badge { display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }",
        f".lt-badge-retrieve {{ background: rgba(0,255,198,0.14); color: {MINT}; }}",
        f".lt-badge-aggregate {{ background: rgba(155,113,255,0.18); color: {PURPLE}; }}",
        f".lt-badge-chart {{ background: rgba(247,181,62,0.16); color: {AMBER}; }}",
        # metric cards
        f".lt-metric {{ background: {CARD}; border: 1px solid {GREY_LINE}; border-radius: 10px; padding: 14px 16px; height: 100%; }}",
        f'.lt-metric-value {{ font-family: "{FONT_HEADING}", sans-serif; font-size: 26px; font-weight: 700; color: {MINT}; line-height: 1.15; }}',
        f".lt-metric-label {{ font-size: 11px; color: {GREY_MUTED}; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px; }}",
        f".lt-source {{ background: {SLATE}; border-left: 2px solid {GREY_LINE}; padding: 9px 13px; margin: 5px 0; border-radius: 4px; font-size: 13px; color: {GREY_TEXT}; line-height: 1.5; }}",
        f".stButton>button {{ background: transparent; color: {MINT}; border: 1px solid rgba(0,255,198,0.42); border-radius: 8px; font-weight: 500; }}",
        f".stButton>button:hover {{ background: rgba(0,255,198,0.10); border-color: {MINT}; }}",
        "#MainMenu, footer { visibility: hidden; }",
    ]
    st.markdown(
        f'<link href="{google_fonts_url()}" rel="stylesheet">'
        f"<style>{''.join(rules)}</style>",
        unsafe_allow_html=True,
    )


def masthead(subtitle: str) -> None:
    st.markdown(
        f"""<div class="lt-header">
              <div class="lt-brand">leadtech<span>;</span></div>
              <div class="lt-tagline">{subtitle}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def metric(value: str | int, label: str) -> str:
    return (
        f'<div class="lt-metric"><div class="lt-metric-value">{value}</div>'
        f'<div class="lt-metric-label">{label}</div></div>'
    )
