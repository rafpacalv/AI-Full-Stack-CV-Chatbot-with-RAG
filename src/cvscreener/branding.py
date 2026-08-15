"""LeadTech corporate identity - single source of truth.

Every value here was verified against the live site with a Playwright
computed-style audit of ``/``, ``/about-us``, ``/work-with-us`` and ``/contact``
(full write-up and screenshots in ``docs/brand-capture.md``). Everything visual
in this project - the generated PDFs, the Streamlit theme and the Plotly charts
- reads its colours from this file, so the whole deliverable stays on-brand from
one place.

Three findings from that audit shape the values below:

* **leadtech.com is a light site**, not a dark one: ``body { background: #FFFFFF }``
  on all four pages, white alone covering 51-61 % of every full-page capture.
  This app is dark anyway - it is a screening console, not a marketing page -
  so what is copied is their *discipline* (flat blocks, square corners, no
  borders, one accent per view), not their canvas. See ``docs/brand-capture.md``
  section 8.
* **The indigo ``#140C29`` this project first used is not a LeadTech colour.**
  It is in their CSS but 17 of its 26 rules are scoped to ``.adventure-2022``, a
  one-off campaign microsite, and it appears zero times in the computed styles
  of the four brand pages. Their real dark tones are ``#000000`` and ``#262627``.
* **Mint is a surface, not a stroke.** The audit found zero non-transparent
  borders anywhere on the site, and mint is never a border, underline, link or
  heading colour - only a fill, always under black text.
"""

from __future__ import annotations

# --- Core palette ---------------------------------------------------------
MINT = "#00FFC6"
"""Primary accent, and the logo colour.

A *fill*: on their site it covers buttons, chips and roughly one card per
screen, always with black text on top, and never appears as a border or as
text. It also has a budget - between 0.3 % and 5.5 % of page area depending on
the page. That budget matters more here than there: mint on white measures a
contrast ratio of only 1.30 (it reads as a pastel), while mint on black
measures 16.12. The same coverage that looks quiet on their canvas would shout
on this one, so mint is kept to the wordmark, the active tab, chips and badges.
"""

GREEN = "#44DE97"  # secondary accent; their CSR and about-us section headings
BLACK = "#000000"  # app canvas - their hero cards, nav overlay and footer block
SLATE = "#262627"  # raised panel - their body ink, legal bar and contact form
WHITE = "#FFFFFF"

SKY = "#D1F2FF"
"""The quiet secondary.

Easy to miss and worth not missing: this pale blue is a *larger* field than mint
on every page they ship (3.3-12.8 % against mint's 0.3-5.5 %), doing the work of
"this region is calm supporting material". Too bright to be a surface in a dark
UI, but exactly right for informational text and labels on black - which gives
the interface somewhere to go that is not mint, and stops mint from having to
mean every kind of emphasis at once.
"""

# --- Supporting accents (all measured on their pages) ---------------------
AMBER = "#F7B53E"  # their social-responsibility band, and its square buttons
PINK = "#F54F81"  # the about-us accent, where mint has no fill presence at all
BLUE = "#2490FF"
CYAN = "#04AEFF"
PURPLE = "#9B71FF"

# Their interactive states step *away* from mint rather than into it: this is
# the nav-link hover colour, and #15856C the social-icon hover fill. Mint itself
# never changes on hover - the only press feedback on the whole site is a
# transform: scale(.95).
MINT_HOVER = "#50E3C2"

# --- Neutrals -------------------------------------------------------------
# Neutral, not indigo-tinted: the brand's greys sit on a black/#262627 axis.
GREY_LINE = "#3A3A3C"
GREY_TEXT = "#C3C3C6"
GREY_MUTED = "#8A8A8E"

# --- The 14-discipline colour code ----------------------------------------
# A closed set LeadTech uses only to colour-code job disciplines, rendered as
# 49x49 tiles and 19px bold labels on their careers site. It is the most
# colour-dense thing they publish, and it is reused here to code candidate roles.
DISCIPLINE_COLOURS = {
    "UX": "#FF4F81",
    "Frontend": "#8E44E7",
    "SEM": "#E74444",
    "Backend": "#2CDE97",
    "Project management": "#FFC168",
    "Finance": "#B84692",
    "BI": "#1BC7D0",
    "SEO": "#FF6C5E",
    "Customer service": "#3369E7",
    "Social media": "#04AEFF",
    "Content": "#0060B4",
    "QA": "#8687E9",
    "Sysadmin": "#F99F61",
    "HR": "#85D0E9",
}

# They set #262627 on every tile except Finance and BI, which take white. The
# last two are this project's extension: Content and Customer service are dark
# enough that ink on them would be unreadable, and they are only in play because
# the corpus contains roles LeadTech does not list.
_LIGHT_TEXT_DISCIPLINES = {"Finance", "BI", "Content", "Customer service"}

# Corpus role -> the group it is displayed under. Matched on substrings because
# `current_role` is re-derived by the LLM from the PDF text and is therefore
# never a fixed enum: 50 CVs yield ~47 distinct strings ("Ingeniera Backend",
# "Senior Backend Engineer", "Full-Stack Engineer (Junior)"). Counting those raw
# would produce a legend of one tile each and no signal at all. First match
# wins, so the more specific patterns come first - "product manager" has to be
# tested before "designer" and "data" before "backend" would ever be reached.
_ROLE_GROUP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("qa", "quality", "test"), "QA"),
    (("seo",), "SEO"),
    (("sem", "paid media", "growth", "ads"), "SEM & Growth"),
    (("product manager", "product owner"), "Product"),
    (("ux", "ui", "design", "diseñador", "diseñadora"), "UX & Design"),
    (("devops", "reliability", "sre", "sysadmin", "platform", "cloud"), "DevOps"),
    (("machine learning", "ml engineer", "inteligencia artificial"), "Machine Learning"),
    (("data", "datos", "analyst", "analista", "business intelligence"), "Data & BI"),
    (("frontend", "front-end", "mobile", "android", "ios"), "Frontend & Mobile"),
    (("backend", "back-end", "full-stack", "full stack", "fullstack"), "Backend"),
)

# Group -> which of LeadTech's 14 discipline colours it wears.
#
# The label and the colour are kept separate on purpose. Eight of these groups
# are a LeadTech discipline outright. Two are not on their careers site at all -
# machine learning and the mobile half of Frontend - and one ("Other") is a
# catch-all. Those borrow unused codes from the same closed set, which keeps the
# strip inside their palette without pretending LeadTech has an "ML" discipline:
# the tile says Machine Learning and merely wears the Content colour.
_GROUP_DISCIPLINE = {
    "Backend": "Backend",
    "Frontend & Mobile": "Frontend",
    "Data & BI": "BI",
    "Machine Learning": "Content",
    "DevOps": "Sysadmin",
    "SEO": "SEO",
    "SEM & Growth": "SEM",
    "UX & Design": "UX",
    "QA": "QA",
    "Product": "Project management",
    "Other": "Customer service",
}


def role_group(role: str | None) -> str:
    """Bucket a free-text job title into one of the display groups above."""
    haystack = (role or "").casefold()
    for needles, group in _ROLE_GROUP:
        if any(needle in haystack for needle in needles):
            return group
    return "Other"


def group_colour(group: str) -> tuple[str, str]:
    """Return ``(background, text)`` for a display group."""
    discipline = _GROUP_DISCIPLINE.get(group, "Customer service")
    text = WHITE if discipline in _LIGHT_TEXT_DISCIPLINES else SLATE
    return DISCIPLINE_COLOURS[discipline], text


def role_colour(role: str | None) -> tuple[str, str]:
    """Return ``(background, text)`` for a candidate's free-text job title.

    Both halves come from LeadTech's own discipline code, so a strip of these
    tiles reproduces the palette of their careers page rather than inventing a
    categorical scheme.
    """
    return group_colour(role_group(role))


# --- Chart palette --------------------------------------------------------
# The raw brand colours are a UI palette, not a data palette, and feeding them
# straight to a chart fails on two counts:
#
#   1. Brand MINT (#00FFC6) sits at OKLCH L 0.889 - far above the L 0.48-0.67
#      band a dark chart surface needs. It glares and vibrates against black.
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
#
# The surface later moved from that indigo to pure black. Only one of the six
# checks depends on the surface, and moving to a darker one can only raise
# contrast: measured, the five ratios went 6.45/5.29/5.61/5.24/5.66 to
# 7.18/5.89/6.25/5.84/6.31. The worst case improved from 5.24 to 5.84, so the
# palette did not need re-stepping.
CHART_SEQUENCE_DARK = ["#00AC7A", "#9375E3", "#C17E00", "#DB5C7D", "#3A8FEB"]
CHART_SEQUENCE_LIGHT = ["#009868", "#8261D3", "#A96C00", "#C8436A", "#1A7ADB"]

# The UI is dark, so this is the default a chart should reach for.
CHART_SEQUENCE = CHART_SEQUENCE_DARK

# Single-hue mint ramp for magnitude (sequential), light -> dark.
CHART_SEQUENTIAL = ["#CFF7E8", "#7FE3C4", "#2FC79C", "#00A77B", "#00785A", "#004C39"]

# Chart furniture: recessive, so the data carries the emphasis.
CHART_SURFACE = BLACK
CHART_GRID = "#2B2B2C"
CHART_AXIS = "#6F6F72"
CHART_TEXT = "#EFEFF1"
CHART_TEXT_MUTED = "#9A9A9E"

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
