"""Render a :class:`CVProfile` to a realistic PDF with ReportLab.

Three distinct layouts exist so that 28 CVs do not read as one template filled
in 28 times - which is exactly what a screening tool would never see in real
life, and which would make the retrieval demo suspiciously easy.

ReportLab (rather than an HTML-to-PDF route) because it is pure Python with
wheels on every platform: no Chrome, no GTK, no system libraries for a reviewer
to install before the project runs.

Fonts are registered from real TrueType files rather than ReportLab's built-ins
because the built-ins are Latin-1 only, and this dataset contains names like
"Katarzyna Wilczyńska" whose characters live outside Latin-1.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
)

from ..config import settings
from ..textutils import ascii_slug
from .schema import CVProfile

PAGE_W, PAGE_H = A4

# Each CV gets its own restrained accent, the way real CV templates do.
ACCENTS = ["#1F3A5F", "#0F6E6E", "#6E2338", "#23503B", "#37474F", "#4A2C5A"]

FONT_DIR = Path("C:/Windows/Fonts")
_FONTS_READY = False


def _register_fonts() -> None:
    """Register Unicode-capable TrueType families, once per process."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    families = {
        "CV-Sans": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf"),
        "CV-Serif": ("georgia.ttf", "georgiab.ttf", "georgiai.ttf"),
        "CV-Grotesk": ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"),
    }
    for family, (regular, bold, italic) in families.items():
        pdfmetrics.registerFont(TTFont(family, FONT_DIR / regular))
        pdfmetrics.registerFont(TTFont(f"{family}-Bold", FONT_DIR / bold))
        pdfmetrics.registerFont(TTFont(f"{family}-Italic", FONT_DIR / italic))
        pdfmetrics.registerFontFamily(
            family, normal=family, bold=f"{family}-Bold", italic=f"{family}-Italic"
        )
    _FONTS_READY = True


# --- Bilingual section labels --------------------------------------------
LABELS = {
    "es": {
        "profile": "PERFIL PROFESIONAL",
        "experience": "EXPERIENCIA PROFESIONAL",
        "education": "FORMACIÓN ACADÉMICA",
        "skills": "COMPETENCIAS TÉCNICAS",
        "tools": "HERRAMIENTAS",
        "languages": "IDIOMAS",
        "certs": "CERTIFICACIONES",
        "contact": "CONTACTO",
        "born": "Fecha de nacimiento",
    },
    "en": {
        "profile": "PROFESSIONAL PROFILE",
        "experience": "WORK EXPERIENCE",
        "education": "EDUCATION",
        "skills": "TECHNICAL SKILLS",
        "tools": "TOOLS",
        "languages": "LANGUAGES",
        "certs": "CERTIFICATIONS",
        "contact": "CONTACT",
        "born": "Date of birth",
    },
}


@lru_cache(maxsize=64)
def _circular_photo(path: Path, px: int = 420) -> ImageReader | None:
    """Return the headshot as a circular image with a transparent surround.

    Cached because ReportLab invokes the page decorator on every page, and
    re-masking a 512px JPEG each time is pure waste.
    """
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")

    # Centre-crop to a square first. Cropping rather than squashing keeps the
    # face in proportion; taking it from the centre keeps the face in frame.
    side = min(img.size)
    img = img.crop(
        (
            (img.width - side) // 2,
            (img.height - side) // 2,
            (img.width + side) // 2,
            (img.height + side) // 2,
        )
    ).resize((px, px), Image.LANCZOS)

    # The circle is an alpha mask, not a drawn shape: a greyscale image that is
    # white inside the ellipse and black outside, applied as the transparency
    # channel. The corners become transparent, so whatever the template paints
    # behind the photo shows through.
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, px, px), fill=255)
    img.putalpha(mask)

    # Handed to ReportLab in memory as PNG - JPEG has no alpha channel, so
    # saving as JPEG here would fill the corners with black.
    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return ImageReader(buf)


class HRule(Flowable):
    """A thin horizontal rule used to separate sections."""

    def __init__(self, width: float, colour: HexColor, thickness: float = 0.6):
        super().__init__()
        self.width, self.colour, self.thickness = width, colour, thickness
        self.height = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def _styles(font: str, accent: HexColor) -> dict[str, ParagraphStyle]:
    return {
        "name": ParagraphStyle(
            "name", fontName=f"{font}-Bold", fontSize=24, leading=27, textColor=accent
        ),
        "name_light": ParagraphStyle(
            "name_light", fontName=f"{font}-Bold", fontSize=24, leading=27, textColor=white
        ),
        "headline": ParagraphStyle(
            "headline", fontName=font, fontSize=11, leading=14,
            textColor=HexColor("#4A4A4A"),
        ),
        "headline_light": ParagraphStyle(
            "headline_light", fontName=font, fontSize=10.5, leading=14,
            textColor=HexColor("#E8E8E8"),
        ),
        "section": ParagraphStyle(
            "section", fontName=f"{font}-Bold", fontSize=10.5, leading=13,
            textColor=accent, spaceBefore=9, spaceAfter=3,
        ),
        "section_side": ParagraphStyle(
            "section_side", fontName=f"{font}-Bold", fontSize=9.5, leading=12,
            textColor=white, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName=font, fontSize=9.4, leading=13.2,
            textColor=HexColor("#232323"), alignment=TA_JUSTIFY,
        ),
        "side": ParagraphStyle(
            "side", fontName=font, fontSize=8.6, leading=12,
            textColor=HexColor("#EDEDED"),
        ),
        "job": ParagraphStyle(
            "job", fontName=f"{font}-Bold", fontSize=10, leading=13,
            textColor=HexColor("#1A1A1A"), spaceBefore=6,
        ),
        "meta": ParagraphStyle(
            "meta", fontName=f"{font}-Italic", fontSize=8.6, leading=11,
            textColor=HexColor("#6A6A6A"), spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=font, fontSize=9.2, leading=12.6,
            textColor=HexColor("#2E2E2E"), leftIndent=9, bulletIndent=1,
            alignment=TA_JUSTIFY, spaceAfter=1.5,
            # Without this the bullet glyph is drawn in ReportLab's default
            # Helvetica, where "•" has no mapping and extracts as "(cid:127)" -
            # silently poisoning every downstream chunk and embedding.
            bulletFontName=font,
            bulletFontSize=9.2,
        ),
    }


def _experience_flowables(p: CVProfile, s: dict, label: str, accent: HexColor, width: float):
    out = [Paragraph(label, s["section"]), HRule(width, accent), Spacer(1, 3)]
    for job in p.experience:
        out.append(Paragraph(escape(f"{job.position} · {job.company}"), s["job"]))
        out.append(Paragraph(escape(f"{job.period}  |  {job.location}"), s["meta"]))
        for b in job.bullets:
            out.append(Paragraph(escape(b), s["bullet"], bulletText="•"))
        out.append(Spacer(1, 3))
    return out


def _education_flowables(p: CVProfile, s: dict, label: str, accent: HexColor, width: float):
    out = [Paragraph(label, s["section"]), HRule(width, accent), Spacer(1, 3)]
    for ed in p.education:
        out.append(Paragraph(escape(ed.degree), s["job"]))
        out.append(Paragraph(escape(f"{ed.institution}  |  {ed.period}"), s["meta"]))
        if ed.detail:
            out.append(Paragraph(escape(ed.detail), s["body"]))
        out.append(Spacer(1, 3))
    return out


def _build_doc(path: Path, p: CVProfile) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
        title=f"CV - {p.full_name}",
        author=p.full_name,
        subject="Synthetic CV generated for a technical assessment - not a real person",
        creator="LeadTech CV Screener - generation pipeline",
    )
    return doc


# --- Template 0: coloured left sidebar ------------------------------------
# ReportLab splits the work in two. Anything that *flows* (paragraphs, which
# wrap and can spill onto another page) goes into Frames as a "story".
# Anything at a fixed position (background blocks, the photo) is painted
# directly onto the canvas by an onPage callback. Two different mechanisms,
# used together on every template here.
def _render_sidebar(path: Path, p: CVProfile, accent: HexColor, lab: dict) -> None:
    font = "CV-Sans"
    s = _styles(font, accent)
    side_w = 62 * mm                              # the coloured band
    main_w = PAGE_W - side_w - 26 * mm            # the rest, minus margins

    def decorate(canvas, _doc):
        """Painted under the flowing text, on every page."""
        canvas.setFillColor(accent)
        canvas.rect(0, 0, side_w, PAGE_H, fill=1, stroke=0)  # full-height band
        photo = _circular_photo(settings.photos_dir / f"{p.cv_id}.jpg")
        if photo:
            d = 34 * mm
            canvas.drawImage(
                photo,
                (side_w - d) / 2,        # horizontally centred in the band
                PAGE_H - d - 14 * mm,    # near the top (y counts up from bottom)
                d, d,
                mask="auto",             # honour the PNG's transparent corners
                preserveAspectRatio=True,
            )

    # Frames are (x, y, width, height) with the origin at the bottom-left.
    # Padding is zeroed so the measurements above are the real ones.
    sidebar = Frame(
        8 * mm, 10 * mm, side_w - 16 * mm, PAGE_H - 62 * mm, id="side",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    main = Frame(
        side_w + 10 * mm, 12 * mm, main_w, PAGE_H - 24 * mm, id="main",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    # Continuation pages drop the sidebar and use the full text column.
    later = Frame(
        side_w + 10 * mm, 12 * mm, main_w, PAGE_H - 24 * mm, id="later",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    doc = _build_doc(path, p)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[sidebar, main], onPage=decorate),
        PageTemplate(id="rest", frames=[later], onPage=decorate),
    ])

    # The story is one flat list of flowables. It fills the first frame
    # (sidebar) until FrameBreak() sends the rest to the second (main column).
    # NextPageTemplate says which layout any page after the first should use.
    #
    # `escape()` on every value is not optional: ReportLab parses Paragraph
    # text as XML, so an "&" or "<" in a company name would raise at build time.
    story: list = [NextPageTemplate("rest")]

    # -- sidebar column: contact, skills, tools, languages, certifications
    story += [
        Paragraph(lab["contact"], s["section_side"]),
        Paragraph(escape(p.city + ", " + p.country), s["side"]),
        Paragraph(escape(p.phone), s["side"]),
        Paragraph(escape(p.email), s["side"]),
        Paragraph(escape(p.linkedin), s["side"]),
        Paragraph(f"{lab['born']}: {escape(p.birth_date)}", s["side"]),
        Paragraph(lab["skills"], s["section_side"]),
    ]
    story += [Paragraph("• " + escape(sk), s["side"]) for sk in p.technical_skills]
    if p.tools:
        story.append(Paragraph(lab["tools"], s["section_side"]))
        story += [Paragraph("• " + escape(t), s["side"]) for t in p.tools]
    story.append(Paragraph(lab["languages"], s["section_side"]))
    story += [
        Paragraph(f"{escape(l.language)} — {escape(l.level)}", s["side"]) for l in p.languages
    ]
    if p.certifications:
        story.append(Paragraph(lab["certs"], s["section_side"]))
        story += [Paragraph("• " + escape(c), s["side"]) for c in p.certifications]

    # -- main column: everything from here lands in the second frame
    story.append(FrameBreak())
    story += [
        Spacer(1, 6 * mm),
        Paragraph(escape(p.full_name), s["name"]),
        Paragraph(escape(p.headline), s["headline"]),
        Spacer(1, 5),
        Paragraph(lab["profile"], s["section"]),
        HRule(main_w, accent),
        Spacer(1, 3),
        Paragraph(escape(p.summary), s["body"]),
    ]
    story += _experience_flowables(p, s, lab["experience"], accent, main_w)
    story += _education_flowables(p, s, lab["education"], accent, main_w)
    doc.build(story)


# --- Template 1: full-width header band -----------------------------------
# Same Frame + onPage mechanics as template 0 above. Two differences: the name
# and contact block are drawn straight onto the canvas with drawString instead
# of flowing as Paragraphs (they are fixed-position, inside the coloured band),
# and the body is a single full-width column.
def _render_banner(path: Path, p: CVProfile, accent: HexColor, lab: dict) -> None:
    font = "CV-Grotesk"
    s = _styles(font, accent)
    band_h = 46 * mm
    body_w = PAGE_W - 40 * mm

    def decorate(canvas, doc):
        if doc.page > 1:
            return
        canvas.setFillColor(accent)
        canvas.rect(0, PAGE_H - band_h, PAGE_W, band_h, fill=1, stroke=0)
        photo = _circular_photo(settings.photos_dir / f"{p.cv_id}.jpg")
        if photo:
            d = 30 * mm
            canvas.drawImage(
                photo, PAGE_W - d - 20 * mm, PAGE_H - band_h + (band_h - d) / 2, d, d,
                mask="auto", preserveAspectRatio=True,
            )
        canvas.setFont(f"{font}-Bold", 22)
        canvas.setFillColor(white)
        canvas.drawString(20 * mm, PAGE_H - 20 * mm, p.full_name)
        canvas.setFont(font, 10.5)
        canvas.setFillColor(HexColor("#E4E4E4"))
        canvas.drawString(20 * mm, PAGE_H - 27 * mm, p.headline[:88])
        canvas.setFont(font, 8.6)
        canvas.drawString(
            20 * mm, PAGE_H - 35 * mm,
            f"{p.email}  ·  {p.phone}  ·  {p.city}, {p.country}",
        )
        canvas.drawString(
            20 * mm, PAGE_H - 40 * mm,
            f"{p.linkedin}  ·  {lab['born']}: {p.birth_date}",
        )

    first = Frame(
        20 * mm, 14 * mm, body_w, PAGE_H - band_h - 24 * mm, id="first",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    rest = Frame(
        20 * mm, 14 * mm, body_w, PAGE_H - 28 * mm, id="rest",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    doc = _build_doc(path, p)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[first], onPage=decorate),
        PageTemplate(id="rest", frames=[rest], onPage=decorate),
    ])

    story: list = [
        NextPageTemplate("rest"),
        Paragraph(lab["profile"], s["section"]),
        HRule(body_w, accent),
        Spacer(1, 3),
        Paragraph(escape(p.summary), s["body"]),
    ]
    story += _experience_flowables(p, s, lab["experience"], accent, body_w)
    story += _education_flowables(p, s, lab["education"], accent, body_w)
    story += [
        Paragraph(lab["skills"], s["section"]),
        HRule(body_w, accent),
        Spacer(1, 3),
        Paragraph(escape(" · ".join(p.technical_skills)), s["body"]),
    ]
    if p.tools:
        story += [
            Paragraph(lab["tools"], s["section"]),
            HRule(body_w, accent),
            Spacer(1, 3),
            Paragraph(escape(" · ".join(p.tools)), s["body"]),
        ]
    story += [
        Paragraph(lab["languages"], s["section"]),
        HRule(body_w, accent),
        Spacer(1, 3),
        Paragraph(
            escape(" · ".join(f"{l.language} ({l.level})" for l in p.languages)), s["body"]
        ),
    ]
    if p.certifications:
        story += [
            Paragraph(lab["certs"], s["section"]),
            HRule(body_w, accent),
            Spacer(1, 3),
            Paragraph(escape(" · ".join(p.certifications)), s["body"]),
        ]
    doc.build(story)


# --- Template 2: classic single column ------------------------------------
# The conservative one, and the only serif layout. No colour block at all: just
# a rule under the header and a small photo top-right, so the three templates
# differ in weight as well as arrangement.
def _render_classic(path: Path, p: CVProfile, accent: HexColor, lab: dict) -> None:
    font = "CV-Serif"
    s = _styles(font, accent)
    body_w = PAGE_W - 44 * mm

    def decorate(canvas, doc):
        if doc.page > 1:
            return
        photo = _circular_photo(settings.photos_dir / f"{p.cv_id}.jpg")
        if photo:
            d = 27 * mm
            canvas.drawImage(
                photo, PAGE_W - d - 22 * mm, PAGE_H - d - 16 * mm, d, d,
                mask="auto", preserveAspectRatio=True,
            )
        canvas.setStrokeColor(accent)
        canvas.setLineWidth(2)
        canvas.line(22 * mm, PAGE_H - 47 * mm, PAGE_W - 22 * mm, PAGE_H - 47 * mm)

    first = Frame(
        22 * mm, 14 * mm, body_w, PAGE_H - 62 * mm, id="first",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    rest = Frame(
        22 * mm, 14 * mm, body_w, PAGE_H - 28 * mm, id="rest",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    doc = _build_doc(path, p)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[first], onPage=decorate),
        PageTemplate(id="rest", frames=[rest], onPage=decorate),
    ])

    contact = " · ".join([p.email, p.phone, f"{p.city}, {p.country}", p.linkedin])
    story: list = [
        NextPageTemplate("rest"),
        Paragraph(escape(p.full_name), s["name"]),
        Paragraph(escape(p.headline), s["headline"]),
        Spacer(1, 2),
        Paragraph(escape(contact), s["meta"]),
        Paragraph(f"{lab['born']}: {escape(p.birth_date)}", s["meta"]),
        Spacer(1, 4),
        Paragraph(escape(p.summary), s["body"]),
    ]
    story += _experience_flowables(p, s, lab["experience"], accent, body_w)
    story += _education_flowables(p, s, lab["education"], accent, body_w)
    story += [
        Paragraph(lab["skills"], s["section"]),
        HRule(body_w, accent),
        Spacer(1, 3),
        Paragraph(escape(" · ".join(p.technical_skills)), s["body"]),
    ]
    if p.tools:
        story.append(
            Paragraph(
                escape(f"{lab['tools'].title()}: " + " · ".join(p.tools)), s["body"]
            )
        )
    story += [
        Paragraph(lab["languages"], s["section"]),
        HRule(body_w, accent),
        Spacer(1, 3),
        Paragraph(
            escape(" · ".join(f"{l.language} ({l.level})" for l in p.languages)), s["body"]
        ),
    ]
    if p.certifications:
        story += [
            Paragraph(lab["certs"], s["section"]),
            HRule(body_w, accent),
            Spacer(1, 3),
            Paragraph(escape(" · ".join(p.certifications)), s["body"]),
        ]
    doc.build(story)


RENDERERS = (_render_sidebar, _render_banner, _render_classic)


def render_cv(profile: CVProfile, *, force: bool = False) -> Path:
    """Render ``profile`` to ``data/cvs/<cv_id>_<slug>.pdf`` and return the path."""
    _register_fonts()  # no-op after the first call
    settings.ensure_dirs()

    # Filename is ASCII-only: it becomes part of an API path and a citation id.
    path = settings.cvs_dir / f"{profile.cv_id}_{ascii_slug(profile.full_name)}.pdf"
    if path.exists() and not force:
        return path

    # Stable per-candidate accent. Hashing the cv_id rather than parsing an
    # index out of it keeps this working for any id format.
    accent = HexColor(ACCENTS[sum(profile.cv_id.encode()) % len(ACCENTS)])
    labels = LABELS[profile.language]  # ES or EN section headings

    # Dispatch to one of the three layout functions. The modulo means a
    # template number beyond the list simply wraps rather than crashing.
    RENDERERS[profile.template % len(RENDERERS)](path, profile, accent, labels)
    return path
