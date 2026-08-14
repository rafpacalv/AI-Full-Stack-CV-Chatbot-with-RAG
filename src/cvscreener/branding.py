"""LeadTech corporate identity - single source of truth.

Palette extracted from the live site's production CSS bundle
(leadtech.com/assets/c819ff5b4e3014b573fdb3b11672ee57.css) and its logo SVGs.
Everything visual in this project - the generated PDFs, the Streamlit theme and
the Plotly charts - reads its colours from here, so the whole deliverable stays
on-brand from one file.
"""

from __future__ import annotations

# --- Core palette ---------------------------------------------------------
MINT = "#00FFC6"  # primary accent, the logo colour
GREEN = "#44DE97"  # secondary accent
INK = "#140C29"  # deep indigo background
SLATE = "#262627"  # cards / raised surfaces
WHITE = "#FFFFFF"

# --- Supporting accents (also from their CSS) -----------------------------
AMBER = "#F7B53E"
BLUE = "#2490FF"
CYAN = "#04AEFF"
PURPLE = "#9B71FF"
PINK = "#F54F81"

# --- Neutrals -------------------------------------------------------------
GREY_LINE = "#3A3A42"
GREY_TEXT = "#B9B9C4"
GREY_MUTED = "#7E7E8C"

# --- Chart palette --------------------------------------------------------
# The raw brand colours are a UI palette, not a data palette, and feeding them
# straight to a chart fails on two counts:
#
#   1. Brand MINT (#00FFC6) sits at OKLCH L 0.889 - far above the L 0.48-0.67
#      band a dark chart surface needs. It glares and vibrates against #140C29.
#   2. MINT (hue 169) and GREEN (159), and BLUE (254) and CYAN (240), are near
#      hue-twins. Under deuteranopia they collapse into each other, so a reader
#      with the most common form of colour blindness cannot tell two series apart.
#
# So the brand hues are kept and only their lightness/chroma are re-stepped in
# OKLCH, and the two colliding hues are dropped rather than recoloured - five
# well-separated hues beat seven that lie about being distinct.
#
# Both sequences below pass all six checks of the dataviz validator
# (lightness band · chroma floor · CVD separation · normal-vision floor ·
# contrast vs surface), verified with:
#   node scripts/validate_palette.js "<hexes>" --mode dark --surface "#140C29"
#
# Worst adjacent pair, dark: #DB5C7D <-> #C17E00, deuteranopia dE 10.3 (target >= 8).
CHART_SEQUENCE_DARK = ["#00AC7A", "#9375E3", "#C17E00", "#DB5C7D", "#3A8FEB"]
CHART_SEQUENCE_LIGHT = ["#009868", "#8261D3", "#A96C00", "#C8436A", "#1A7ADB"]

# The UI is dark, so this is the default a chart should reach for.
CHART_SEQUENCE = CHART_SEQUENCE_DARK

# Single-hue mint ramp for magnitude (sequential), light -> dark.
CHART_SEQUENTIAL = ["#CFF7E8", "#7FE3C4", "#2FC79C", "#00A77B", "#00785A", "#004C39"]

# Chart furniture: recessive, so the data carries the emphasis.
CHART_SURFACE = INK
CHART_GRID = "#2E2A3D"
CHART_AXIS = "#6F6B80"
CHART_TEXT = "#D8D6E0"
CHART_TEXT_MUTED = "#8E8AA0"

# Their own stylesheet declares these as the fallbacks for the proprietary
# "leadtech-font", so they are the closest defensible web-safe stand-ins.
FONT_HEADING = "Comfortaa"
FONT_BODY = "Roboto"


def google_fonts_url() -> str:
    return (
        "https://fonts.googleapis.com/css2"
        "?family=Comfortaa:wght@400;600;700"
        "&family=Roboto:wght@300;400;500;700"
        "&display=swap"
    )
