"""LeadTech look-and-feel for the Streamlit app.

All colour values come from :mod:`cvscreener.branding`, which was itself
verified against leadtech.com's production CSS with a computed-style audit
(``docs/brand-capture.md``) - so the UI stays on-brand from a single source
rather than from hex codes sprinkled through the markup.

The audit changed how this file is written. leadtech.com turns out to be a
*light* site built from flat, hard-edged colour blocks, and four of its rules
are strong enough to be worth obeying even in a dark console:

1. **Mint is a fill, never a stroke.** Their stylesheet has no mint border, no
   mint underline, no mint link and no mint heading - and in fact zero
   non-transparent borders of any colour on any of the four pages audited.
   So nothing here outlines anything in mint; mint fills the wordmark's
   semicolon, the active tab, the user's own messages, chips and badges.
2. **Mint always carries black text.** Mint measures luminance ~205; white on
   it fails contrast and appears nowhere on their site.
3. **Square corners.** ``border-radius: 3px`` exists on exactly one thing on the
   whole site - buttons. Everything else is 0.
4. **Separate panels by stepping the background, not by drawing a line.** Black
   canvas next to ``#262627`` panel next to a colour block. No 1px grey rules,
   no elevation shadows, no gradients.

One deliberate deviation, noted so it does not read as an oversight: their
buttons do not react to hover at all (only ``transform: scale(.95)`` on press).
A console with a dozen clickable things needs more feedback than a landing page
does, so buttons here take a one-step background lift on hover. Everything else
follows the site.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

from cvscreener.branding import (
    AMBER,
    BLACK,
    FONT_BODY,
    FONT_HEADING,
    GREY_MUTED,
    MINT,
    MINT_HOVER,
    PURPLE,
    SKY,
    SLATE,
    WHITE,
    google_fonts_url,
    group_colour,
)

# One step up from the SLATE panel, for the few surfaces that sit on top of
# another panel (hovered buttons, the tab strip's inactive tabs).
SLATE_RAISED = "#333335"

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo-leadtech.svg"


@lru_cache(maxsize=1)
def _logo_svg() -> str:
    """Return the LeadTech wordmark, ready to inline.

    Inlined rather than served as an ``<img>`` so the two glyph groups stay
    styleable from CSS - the wordmark and its semicolon take different colours,
    and both come from :mod:`cvscreener.branding`. Sliced from the first
    ``<svg`` so the provenance comment in the asset does not reach the DOM.
    """
    try:
        raw = LOGO_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""  # a missing asset degrades to the text wordmark, never a crash
    return raw[raw.index("<svg") :]


def inject_css() -> None:
    """Inject the LeadTech stylesheet.

    Written as one blank-line-free block on purpose: a blank line inside a raw
    HTML block terminates it under CommonMark, after which Streamlit renders the
    remaining CSS to the page as literal text.
    """
    rules = [
        f'html, body, [class*="css"] {{ font-family: "{FONT_BODY}", -apple-system, sans-serif; }}',
        f".stApp {{ background: {BLACK}; }}",
        # Display type is huge and negatively tracked on their site (65-90px,
        # weight 900, -1.2 to -1.7px) against 13-14px UI type, with almost
        # nothing in between. That gap is the most characteristic thing about
        # their typography, so headings here are tightened rather than merely
        # enlarged.
        f'h1, h2, h3, h4 {{ font-family: "{FONT_HEADING}", "{FONT_BODY}", sans-serif !important; letter-spacing: -0.02em; }}',
        # masthead: the wordmark itself, at a size that reads as a logo rather
        # than as a label. No rule underneath it - the tab strip below is a
        # solid block, which is how their layout separates regions.
        ".lt-header { display: flex; align-items: center; gap: 22px; padding: 6px 0 20px 0; flex-wrap: wrap; }",
        ".lt-logo { height: 64px; width: auto; display: block; flex: 0 0 auto; }",
        f".lt-logo-word {{ fill: {WHITE}; }}",
        f".lt-logo-mark {{ fill: {MINT}; }}",
        # Fallback wordmark, used only if the SVG asset cannot be read.
        f'.lt-brand {{ font-family: "{FONT_HEADING}", sans-serif; font-size: 40px; font-weight: 700; color: {WHITE}; letter-spacing: -0.03em; line-height: 1; }}',
        f".lt-brand span {{ color: {MINT}; }}",
        # Sky, not grey: their pale blue is a larger field than mint on every
        # page they ship, and it is what keeps mint from having to mean every
        # kind of emphasis at once.
        f".lt-tagline {{ color: {SKY}; font-size: 13px; letter-spacing: 0.02em; line-height: 1.5; }}",
        # tabs: solid blocks, the active one filled mint with black text
        '.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: none; background: transparent; }',
        f'.stTabs [data-baseweb="tab"] {{ background: {SLATE}; color: {WHITE}; border-radius: 0; padding: 11px 24px; font-weight: 500; font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; }}',
        f'.stTabs [aria-selected="true"] {{ background: {MINT}; color: {BLACK} !important; }}',
        # Streamlit paints its own sliding underline in primaryColor; mint as a
        # 2px rule is exactly the stroke role their brand never uses.
        '.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { background: transparent !important; height: 0 !important; }',
        # chat: the question is a mint block, the answer a slate one. Mint
        # shrinks to its content rather than filling the column, which is what
        # keeps it inside the 1-3% area budget the brand implies.
        f".lt-msg-user {{ display: inline-block; max-width: 88%; background: {MINT}; color: {BLACK}; padding: 13px 19px; border-radius: 0; margin: 12px 0 2px 0; font-weight: 500; line-height: 1.5; }}",
        f".lt-msg-bot {{ background: {SLATE}; color: {WHITE}; padding: 18px 22px; border-radius: 0; margin: 4px 0 16px 0; line-height: 1.65; }}",
        # citation chips: mint fill, black text - their "Barcelona HQ" chip
        f".lt-chip {{ display: inline-block; background: {MINT}; color: {BLACK}; padding: 4px 13px; border-radius: 0; font-size: 12px; margin: 3px 6px 3px 0; font-weight: 500; letter-spacing: 0.02em; }}",
        f".lt-chip-muted {{ display: inline-block; background: {SLATE}; color: {SKY}; padding: 4px 13px; border-radius: 0; font-size: 12px; margin: 3px 6px 3px 0; }}",
        # Mustard, not mint: this chip reports an absence and must not read as a
        # normal filter chip. It is also the brand's own idiom for an
        # overridden section - their CSR band swaps every button to this colour
        # and squares the corners.
        f".lt-chip-warn {{ display: inline-block; background: {AMBER}; color: {BLACK}; padding: 4px 13px; border-radius: 0; font-size: 12px; margin: 3px 6px 3px 0; font-weight: 500; }}",
        # routing badge: a fill per intent, always under black text
        f".lt-badge {{ display: inline-block; padding: 4px 12px; border-radius: 0; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: {BLACK}; }}",
        f".lt-badge-retrieve {{ background: {MINT}; }}",
        f".lt-badge-aggregate {{ background: {PURPLE}; }}",
        f".lt-badge-chart {{ background: {AMBER}; }}",
        # metric cards: no border, no radius, no shadow - just a stepped block
        f".lt-metric {{ background: {SLATE}; border: none; border-radius: 0; padding: 16px 18px; height: 100%; }}",
        f'.lt-metric-value {{ font-family: "{FONT_HEADING}", sans-serif; font-size: 34px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; }}',
        f".lt-metric-label {{ font-size: 11px; color: {GREY_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }}",
        # discipline tiles: LeadTech's own 14-colour job code, reused to code
        # candidate roles. This is the most colour-dense thing they publish.
        ".lt-tile { display: inline-block; padding: 11px 15px; border-radius: 0; margin: 0 6px 6px 0; min-width: 86px; }",
        f'.lt-tile-count {{ display: block; font-family: "{FONT_HEADING}", sans-serif; font-size: 23px; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }}',
        ".lt-tile-label { display: block; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 5px; opacity: 0.82; }",
        f".lt-source {{ background: {SLATE}; border: none; border-radius: 0; padding: 12px 16px; margin: 6px 0; font-size: 13px; color: #DDDDE0; line-height: 1.58; }}",
        f".lt-source b {{ color: {SKY}; font-weight: 500; }}",
        # buttons: square-ish (3px is the one radius their site uses), no
        # border, no shadow, and a press scale rather than a glow
        f'.stButton>button, .stDownloadButton>button {{ background: {SLATE}; color: {WHITE}; border: none; border-radius: 3px; font-family: "{FONT_BODY}", sans-serif; font-weight: 500; transition: all 0.2s; }}',
        f".stButton>button:hover, .stDownloadButton>button:hover {{ background: {SLATE_RAISED}; color: {WHITE}; border: none; }}",
        ".stButton>button:active, .stDownloadButton>button:active { transform: scale(0.95); }",
        # Streamlit draws a focus ring in primaryColor, which would put a mint
        # stroke on everything - the one role the brand never gives it.
        ".stButton>button:focus, .stDownloadButton>button:focus { box-shadow: none !important; outline: none !important; }",
        f'.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{ background: {MINT}; color: {BLACK}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 500; }}',
        f'.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover {{ background: {MINT_HOVER}; color: {BLACK}; }}',
        # inputs and containers: square, flat, no mint outline
        f'[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{ border-radius: 0 !important; background: {SLATE} !important; border: none !important; }}',
        f'[data-testid="stChatInput"] textarea {{ color: {WHITE}; }}',
        f'[data-testid="stExpander"] {{ border: none !important; border-radius: 0 !important; background: {SLATE}; }}',
        f'[data-testid="stExpander"] summary {{ color: {SKY}; font-size: 13px; }}',
        f'[data-testid="stSidebar"] {{ background: {SLATE}; }}',
        f'[data-testid="stSidebar"] hr {{ border-color: {SLATE_RAISED}; }}',
        # The sidebar is itself a SLATE block, so a SLATE button on it is
        # invisible. Everything in there steps up one level instead.
        f'[data-testid="stSidebar"] .stButton>button {{ background: {SLATE_RAISED}; }}',
        f'[data-testid="stSidebar"] .stButton>button:hover {{ background: #3F3F41; }}',
        # Sample questions are sentences, not labels: centred text makes a
        # two-line question read as two unrelated fragments.
        '[data-testid="stSidebar"] .stButton>button { justify-content: flex-start; text-align: left; }',
        '[data-testid="stSidebar"] .stButton>button p { text-align: left; }',
        # Streamlit's own top strip is a pink-to-yellow gradient - the one thing
        # the brand pages have none of.
        '[data-testid="stDecoration"] { display: none; }',
        "#MainMenu, footer { visibility: hidden; }",
    ]
    st.markdown(
        f'<link href="{google_fonts_url()}" rel="stylesheet">'
        f"<style>{''.join(rules)}</style>",
        unsafe_allow_html=True,
    )


def masthead(subtitle: str) -> None:
    """Render the LeadTech wordmark and a one-line description of the app."""
    logo = _logo_svg()
    mark = (
        logo.replace("<svg", '<svg class="lt-logo"', 1)
        if logo
        else '<div class="lt-brand">leadtech<span>;</span></div>'
    )
    st.markdown(
        f'<div class="lt-header">{mark}'
        f'<div class="lt-tagline">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_logo(subtitle: str) -> str:
    """The same wordmark at sidebar scale, as an HTML string."""
    logo = _logo_svg()
    mark = (
        logo.replace("<svg", '<svg class="lt-logo" style="height:30px"', 1)
        if logo
        else '<div class="lt-brand" style="font-size:22px">leadtech<span>;</span></div>'
    )
    return (
        f'<div style="padding:2px 0 16px 0">{mark}'
        f'<div style="color:{GREY_MUTED};font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-top:9px">{subtitle}</div></div>'
    )


def metric(value: str | int, label: str, accent: str = MINT) -> str:
    """A single stat block.

    The accent is a parameter because leadtech.com never shows two accents in
    one viewport - it recolours whole sections instead (their CSR band is
    mustard, /about-us is pink, /work-with-us is mint). Each tab here follows
    that rule and picks one.
    """
    return (
        f'<div class="lt-metric"><div class="lt-metric-value" style="color:{accent}">{value}</div>'
        f'<div class="lt-metric-label">{label}</div></div>'
    )


def role_tiles(counts: dict[str, int]) -> str:
    """Candidate disciplines as LeadTech tiles, largest group first.

    Reproduces the colour-coded tile grid from their careers page, which is the
    densest use of colour they publish, and doubles as a readable summary of
    what the corpus actually contains. The count is the tile's own label, so the
    colour never has to be decoded to read the chart.
    """
    tiles = []
    for group, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        background, text = group_colour(group)
        tiles.append(
            f'<div class="lt-tile" style="background:{background};color:{text}">'
            f'<span class="lt-tile-count">{count}</span>'
            f'<span class="lt-tile-label">{group}</span></div>'
        )
    return f'<div style="margin:4px 0 12px 0">{"".join(tiles)}</div>'
