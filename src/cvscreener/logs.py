"""Logging for the API: a human-readable log, and a machine-readable transcript.

Two destinations, because they answer different questions.

``data/logs/cvscreener.log`` is the ordinary rotating log every module already
writes to via ``logging.getLogger(__name__)``. It is what you read when
something breaks: warmup failures, Ollama timeouts, tracebacks.

``data/logs/chat.jsonl`` is one JSON object per question - the question, the
answer, how it was routed and how long it took. It is separate because it is
*data*, not diagnostics: grep-ing a distribution of intents or finding every
question that produced an error out of an interleaved text log is miserable,
and one JSON object per line is a pandas one-liner.

Nothing here is on the request's critical path. Every write is wrapped, because
a full disk or a locked file should degrade a screening tool to "not logging",
never to "not answering".
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

# Appends happen from the SSE generator, which FastAPI runs in a worker thread,
# so two questions in flight could interleave a line. One lock around the write
# keeps each record whole.
_write_lock = threading.Lock()
_configured = False


def configure(level: int = logging.INFO) -> None:
    """Attach a rotating file handler and a console handler to the root logger.

    Idempotent: uvicorn's reloader imports the app more than once, and stacking
    handlers would duplicate every line.
    """
    global _configured
    if _configured:
        return

    settings.ensure_dirs()

    # Windows consoles default to cp1252, which cannot encode candidate names
    # like "Wilczyńska". Everything downstream is UTF-8, so make the console
    # match rather than lose a log line to a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # not a real console (pytest, pipes)
                pass

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1 MB x 3 is plenty for a demo and cannot quietly fill the disk.
    file_handler = RotatingFileHandler(
        settings.logs_dir / "cvscreener.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console)

    # httpx logs one INFO line per request, and every embedding call is a
    # request - it would bury everything else.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True
    log.info("logging to %s", settings.logs_dir)


def log_chat(
    question: str,
    answer: str,
    *,
    model: str,
    intent: str = "",
    elapsed_s: float | None = None,
    citations: list[str] | None = None,
    missing_terms: list[str] | None = None,
    chart: str | None = None,
    error: str | None = None,
) -> None:
    """Append one question/answer record to the transcript.

    The CVs are synthetic and the app is local, so questions and answers are
    recorded in full - there is no real personal data in this corpus. A
    deployment against real CVs would need a retention policy and a decision
    about whether recruiter queries and candidate names belong on disk at all;
    that is a policy question, not a code one, and it is flagged in the README
    rather than pretended away here.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "model": model,
        "intent": intent,
        "elapsed_s": elapsed_s,
        "answer": answer,
        "citations": citations or [],
        "missing_terms": missing_terms or [],
        "chart": chart,
        "error": error,
    }

    if error:
        log.error("chat failed after %ss: %s | q=%r", elapsed_s, error, question)
    else:
        log.info(
            "chat %s in %ss (%d citations)%s | q=%r",
            intent or "?",
            elapsed_s,
            len(record["citations"]),
            " [missing: " + ", ".join(record["missing_terms"]) + "]"
            if record["missing_terms"]
            else "",
            question,
        )

    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _write_lock:
            with open(settings.logs_dir / "chat.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:  # noqa: BLE001 - never fail a request over a log line
        log.warning("could not write the chat transcript: %s", exc)
