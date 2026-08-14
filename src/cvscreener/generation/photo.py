"""AI-generated headshots via Pollinations (free, no API key).

Ollama cannot generate images, so the brief's "call any LLM for the texts and
images" is satisfied by a hosted diffusion model. Pollinations needs no
credentials, which keeps the project runnable by a reviewer with nothing but a
clone and an Ollama install.

If the network is unavailable the pipeline degrades to a locally drawn monogram
avatar rather than failing the whole run.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from ..branding import INK, MINT
from ..config import settings
from .personas import Persona

log = logging.getLogger(__name__)

SIZE = 512


def _prompt(persona: Persona) -> str:
    gender = "woman" if persona.gender == "female" else "man"
    return (
        f"professional corporate headshot portrait photo of a {persona.age} year old "
        f"{persona.appearance} {gender}, {persona.role_display.lower()}, "
        "neutral office background, soft natural lighting, business casual attire, "
        "friendly confident expression, looking at camera, shallow depth of field, "
        "sharp focus, high detail, photorealistic, 50mm lens"
    )


def _monogram(persona: Persona, path: Path) -> Path:
    """Offline fallback: initials on a LeadTech-mint tile."""
    img = Image.new("RGB", (SIZE, SIZE), INK)
    draw = ImageDraw.Draw(img)
    draw.ellipse((36, 36, SIZE - 36, SIZE - 36), fill=MINT)
    initials = "".join(p[0].upper() for p in persona.full_name.split()[:2])
    try:
        font = ImageFont.truetype("arialbd.ttf", 190)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), initials, font=font)
    draw.text(
        ((SIZE - box[2] + box[0]) / 2, (SIZE - box[3] + box[1]) / 2 - 12),
        initials,
        fill=INK,
        font=font,
    )
    img.save(path, "JPEG", quality=92)
    return path


def generate_photo(persona: Persona, *, force: bool = False) -> Path:
    """Return the path to this persona's headshot, generating it if needed."""
    settings.ensure_dirs()
    path = settings.photos_dir / f"{persona.cv_id}.jpg"
    if path.exists() and not force:
        return path

    if not settings.photo_enabled:
        return _monogram(persona, path)

    # A per-persona seed keeps runs reproducible and stops the model returning
    # the same face for two similar prompts.
    seed = int(persona.cv_id.split("_")[1]) * 7919
    query = urllib.parse.quote(_prompt(persona))

    # The free tier returns sporadic 500s under load, so retry with backoff and
    # drop the explicit model pin on later attempts (the default pool is more
    # reliable than requesting flux specifically).
    attempts = [
        f"?width={SIZE}&height={SIZE}&seed={seed}&nologo=true&model=flux",
        f"?width={SIZE}&height={SIZE}&seed={seed}&nologo=true",
        f"?width={SIZE}&height={SIZE}&seed={seed + 1}&nologo=true",
    ]
    for i, params in enumerate(attempts):
        try:
            r = httpx.get(
                f"{settings.pollinations_url}/{query}{params}",
                timeout=180.0,
                follow_redirects=True,
            )
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((SIZE, SIZE))
            img.save(path, "JPEG", quality=92)
            return path
        except Exception as exc:  # noqa: BLE001 - retried, then degraded
            log.warning(
                "photo attempt %d/%d failed for %s: %s",
                i + 1, len(attempts), persona.cv_id, exc,
            )
            if i < len(attempts) - 1:
                time.sleep(3 * (i + 1))

    log.warning("photo generation exhausted for %s; using monogram", persona.cv_id)
    return _monogram(persona, path)
