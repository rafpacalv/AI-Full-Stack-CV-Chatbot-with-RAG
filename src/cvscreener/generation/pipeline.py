"""CLI orchestrator for the CV generation pipeline.

Resumability is the point. A full 28-CV batch is ~30 minutes of local
inference, so every stage caches its artefact on disk and is skipped when that
artefact already exists. Fixing a bug in the renderer therefore costs a few
seconds, not another half hour of re-generating text that was already fine.

    python -m cvscreener.generation.pipeline               # all 28, resume
    python -m cvscreener.generation.pipeline --count 2     # smoke test
    python -m cvscreener.generation.pipeline --only cv_07  # one candidate
    python -m cvscreener.generation.pipeline --stage render --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from ..config import settings
from .personas import build_personas
from .photo import generate_photo
from .profile import generate_profile
from .render import render_cv

log = logging.getLogger("pipeline")


def run(
    *,
    count: int | None = None,
    only: str | None = None,
    force: bool = False,
    stage: str = "all",
) -> None:
    settings.ensure_dirs()
    personas = build_personas()
    if only:
        personas = [p for p in personas if p.cv_id == only]
        if not personas:
            sys.exit(f"No persona with id {only!r}")
    elif count:
        personas = personas[:count]

    do_text = stage in ("all", "text")
    do_photo = stage in ("all", "photo")
    do_render = stage in ("all", "render")

    started = time.time()
    print(
        f"Generating {len(personas)} CVs  "
        f"[text={settings.gen_model} photos={'on' if settings.photo_enabled else 'off'}]\n"
    )

    for i, persona in enumerate(personas, 1):
        t0 = time.time()
        # `marks` records what each stage actually did. Printing cache-vs-new
        # per CV is what makes it obvious at a glance whether a run is doing
        # 30 seconds of work or 30 minutes.
        marks: list[str] = []

        # -- photo: network call, cached on disk
        if do_photo:
            cached = (settings.photos_dir / f"{persona.cv_id}.jpg").exists()
            generate_photo(persona, force=force)
            marks.append("photo:cache" if cached and not force else "photo:new")

        # -- text: the expensive stage, ~60 s of local inference per CV
        profile = None
        if do_text:
            cached = (settings.profiles_dir / f"{persona.cv_id}.json").exists()
            profile = generate_profile(persona, force=force)
            marks.append("text:cache" if cached and not force else "text:new")

        # -- render: milliseconds, so always redone. That asymmetry is the
        # point of the whole design: a bug in the PDF layout costs one cheap
        # re-render, never a regeneration of text that was already correct.
        if do_render:
            if profile is None:
                profile = generate_profile(persona)  # --stage render: load cache
            path = render_cv(profile, force=True)
            marks.append(f"pdf:{path.name}")

        print(
            f"[{i:>2}/{len(personas)}] {persona.cv_id} {persona.language} "
            f"{persona.full_name:<24.24} {time.time() - t0:>5.1f}s  {'  '.join(marks)}",
            flush=True,
        )

    mins, secs = divmod(int(time.time() - started), 60)
    pdfs = len(list(settings.cvs_dir.glob("cv_*.pdf")))
    print(f"\nDone in {mins}m{secs:02d}s - {pdfs} PDFs in {settings.cvs_dir}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Generate the synthetic CV dataset")
    ap.add_argument("--count", type=int, help="only the first N personas")
    ap.add_argument("--only", help="a single cv_id, e.g. cv_07")
    ap.add_argument("--force", action="store_true", help="ignore caches and regenerate")
    ap.add_argument(
        "--stage",
        choices=("all", "text", "photo", "render"),
        default="all",
        help="run a single stage (default: all)",
    )
    args = ap.parse_args()
    run(count=args.count, only=args.only, force=args.force, stage=args.stage)


if __name__ == "__main__":
    main()
