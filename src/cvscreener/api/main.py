"""FastAPI backend for the CV screener.

The interesting endpoint is ``POST /chat``. It streams Server-Sent Events so the
UI can render tokens as the local model produces them - at ~18 tok/s that is the
difference between a live-feeling assistant and a fifteen-second blank screen.

Event sequence:

    plan   -> the routing decision (also drives the observability panel)
    token  -> one chunk of the answer, many of these
    meta   -> citations, retrieved chunks with scores, optional chart payload
    done   -> terminator
    error  -> something failed; the UI surfaces it instead of hanging
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..llm import client
from ..logs import (
    clear_transcript,
    configure as configure_logging,
    log_chat,
    read_transcript,
)
from ..rag.aggregate import candidates_summary, load_candidates, run_aggregate
from ..rag.answer import stream_aggregate_answer, stream_retrieval_answer, tidy_answer
from ..rag.keywords import missing_from_corpus
from ..rag.retrieve import IndexNotBuilt, embed_query, load_index, search
from ..rag.router import QueryPlan, route

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the models and the index before the first user question.

    Cold, the first query costs ~31 s because Ollama loads gemma2 and bge-m3 on
    demand and the index is read lazily. Paying that at startup instead makes
    the first real question as fast as the rest - which matters when the first
    question is the one being demonstrated.
    """
    # Before the warmup thread starts, so its failures are captured too.
    configure_logging()

    def warm() -> None:
        try:
            load_index()
            client.embed(["warmup"])
            list(client.chat_stream([{"role": "user", "content": "hi"}], num_predict=1))
            log.info("warmup complete")
        except Exception as exc:  # noqa: BLE001 - never block startup
            log.warning("warmup skipped: %s", exc)

    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(
    title="LeadTech CV Screener API",
    version="1.0.0",
    description="Bilingual RAG over a synthetic CV corpus, running on local Ollama.",
    lifespan=lifespan,
)

# The Streamlit UI is a separate process on another port - but the request to
# this API is made by Streamlit's *server*, in Python, not by the page in the
# browser. No browser ever calls this API cross-origin, so CORS buys the app
# nothing and only widens what a web page can reach.
#
# That was harmless while every endpoint was read-only. `DELETE /logs` is not:
# with a wildcard policy, any site the user happened to be visiting could wipe
# their transcript with one fetch() to localhost. Narrowed to the UI's own
# origins, which keeps `fetch` from a browser console working during
# development without leaving the door open to the whole web.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{settings.ui_port}",
        f"http://127.0.0.1:{settings.ui_port}",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Candidate identifiers are generated as cv_01..cv_50, so anything outside this
# alphabet is either a typo or an attack. Excluding "/", "\\", "." and the glob
# metacharacters is the point - see cv_pdf below.
CV_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# The corpus is 375 chunks and the chat model has an 8k context, so nothing
# legitimate comes near these. They exist so a stray or hostile request cannot
# make the server embed a megabyte of text or assemble the entire index into
# one prompt.
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 50


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    model: str | None = Field(default=None, description="Override the chat model")
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=8, ge=1, le=MAX_TOP_K)


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/health")
def health() -> dict:
    ollama_up = client.is_up()
    try:
        index = load_index()
        indexed = {"chunks": len(index.chunks), "dim": index.dim}
    except IndexNotBuilt:
        indexed = None
    return {
        "status": "ok" if (ollama_up and indexed) else "degraded",
        "ollama": {"up": ollama_up, "host": settings.ollama_host},
        "models": {
            "chat": settings.chat_model,
            "embed": settings.embed_model,
            "available": client.available_models(),
        },
        "index": indexed,
    }


@app.get("/stats")
def stats() -> dict:
    path = settings.index_dir / "stats.json"
    if not path.exists():
        raise HTTPException(404, "Index not built yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/candidates")
def candidates() -> list[dict]:
    try:
        return candidates_summary()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc


@app.get("/cv/{cv_id}")
def cv_pdf(cv_id: str) -> FileResponse:
    """Serve the original PDF, so a citation can open the real document.

    ``cv_id`` is validated before it is interpolated into a glob. Without that
    this was a working path traversal, and the vector was not the obvious one:
    ``/cv/../secret`` and ``/cv/..%2Fsecret`` both 404 in Starlette's router,
    because the path is decoded and normalised before the route is matched. A
    **backslash** is not a URL path separator, so ``%5C`` survives routing
    intact - and on Windows ``Path.glob("..\\\\secret_*.pdf")`` happily walks up
    a directory. ``GET /cv/..%5Cdecoy`` returned a file from outside the corpus
    with HTTP 200. Verified, then fixed.

    Two independent checks, because one regex is a single point of failure:
    the identifier must be alphanumeric (which also excludes the glob
    metacharacters ``*?[``), and the resolved file must sit directly in the
    corpus directory.
    """
    if not CV_ID_RE.fullmatch(cv_id):
        raise HTTPException(400, "cv_id must be alphanumeric")

    root = settings.cvs_dir.resolve()
    for path in sorted(settings.cvs_dir.glob(f"{cv_id}_*.pdf")):
        if path.resolve().parent == root:
            return FileResponse(path, media_type="application/pdf", filename=path.name)
    raise HTTPException(404, f"No PDF for {cv_id}")


@app.get("/logs")
def logs(limit: int = Query(default=200, ge=1, le=1000)) -> dict:
    """The question/answer transcript, newest first."""
    records = read_transcript(limit)
    return {"returned": len(records), "limit": limit, "records": records}


@app.delete("/logs")
def clear_logs() -> dict:
    """Empty the transcript.

    The only endpoint here that destroys anything, which is why the CORS policy
    above stopped being a wildcard when this was added.
    """
    return {"removed": clear_transcript()}


@app.post("/search")
def search_endpoint(req: SearchRequest) -> dict:
    """Raw retrieval with scores - powers the observability panel."""
    t0 = time.time()
    hits = search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "results": [h.as_dict() for h in hits],
    }


@app.post("/route")
def route_endpoint(req: ChatRequest) -> QueryPlan:
    return route(req.question, model=req.model)


def _chat_events(req: ChatRequest) -> Iterator[str]:
    started = time.time()

    # The model is a per-request value, passed down explicitly.
    #
    # This used to be `settings.chat_model = req.model`, which mutated process
    # global state: the sidebar's model picker leaked into every other request,
    # and one request naming a model Ollama does not have left the server
    # answering 404 to everybody afterwards - including requests that sent no
    # model at all. Reproduced, then removed.
    model = req.model or settings.chat_model

    # Collected as the answer streams, so the transcript records what the user
    # actually saw rather than a second, differently-sampled generation.
    spoken: list[str] = []
    plan: QueryPlan | None = None

    try:
        # Routing (chat model) and query embedding (embedding model) are
        # independent, and both models are resident in VRAM at once. Running
        # them concurrently rather than back-to-back roughly halves the time
        # before the first token appears - ~13s down to ~7s on this machine.
        # Start the embedding in a background thread, then do the routing on
        # this one. Speculative: the aggregate branch will not need it, but the
        # wasted work is one embedding call and the saving on the common path
        # is several seconds.
        pool = ThreadPoolExecutor(max_workers=1)
        embedding_future = pool.submit(embed_query, req.question)
        try:
            plan = route(req.question, model=model)
        except Exception:
            embedding_future.cancel()  # do not leak the thread on failure
            pool.shutdown(wait=False)
            raise

        # Which literal terms from the question exist in no CV at all. Checked
        # against the whole corpus, not the retrieved chunks, so the statement
        # is definitive rather than an artefact of what happened to rank.
        missing = missing_from_corpus(req.question, skill=plan.skill)

        # The user asked to plot a field the candidate table does not hold.
        # Reported alongside the plan so the UI can say so before the answer
        # arrives, rather than the user waiting for a chart that never comes.
        chart_unavailable = plan.intent == "chart" and plan.dimension == "unsupported"

        # First event out: the routing decision. The UI paints its badge from
        # this immediately, so the user sees the system reacting long before
        # any answer text exists.
        yield _sse(
            "plan",
            {
                **plan.model_dump(),
                "missing_terms": missing,
                "chart_unavailable": chart_unavailable,
            },
        )

        # --- BRANCH A: aggregate / chart ---------------------------------
        # Counting and plotting questions are answered from the candidate
        # table, not from retrieved text, so the figure covers every CV.
        if plan.intent in ("aggregate", "chart"):
            embedding_future.cancel()  # this branch never searches
            pool.shutdown(wait=False)
            result = run_aggregate(plan)
            for token in stream_aggregate_answer(
                req.question, result, chart_unavailable=chart_unavailable, model=model
            ):
                spoken.append(token)
                yield _sse("token", {"t": token})

            frame = result.matched
            # Here a "citation" is every candidate the answer is drawn from -
            # these came from the table, not from retrieved chunks, so there is
            # no section to point at.
            #
            # Every match is sent, not a slice. This list is what the user
            # clicks through to check the answer against the real PDFs, so a
            # cap turns "the six graduates from UPC" into an arbitrary subset
            # of them, and a chart covering all 50 candidates should offer all
            # 50. If nothing matched, nothing is offered.
            #
            # The exception is a chart of a field the table does not hold:
            # there is no result to source, and listing the whole corpus
            # beneath "that field does not exist" claims a relevance it has not
            # got.
            citations = (
                []
                if chart_unavailable
                else [
                    {
                        "cv_id": row["cv_id"],
                        "candidate": row["full_name"],
                        "source_file": row["source_file"],
                        "sections": [],
                    }
                    for _, row in frame.iterrows()
                ]
            )
            yield _sse(
                "meta",
                {
                    "mode": plan.intent,
                    "missing_terms": missing,
                    "chart_unavailable": chart_unavailable,
                    "filters": result.filters,
                    "chart": result.chart,
                    "matched_count": int(len(frame)),
                    "citations": citations,
                    "chunks": [],
                    "elapsed_s": round(time.time() - started, 2),
                },
            )
            log_chat(
                req.question,
                tidy_answer("".join(spoken)),
                model=model,
                intent=plan.intent,
                elapsed_s=round(time.time() - started, 2),
                citations=[c["candidate"] for c in citations],
                missing_terms=missing,
                chart=f"{result.chart['type']}:{result.chart['dimension']}"
                if result.chart
                else None,
            )
            yield _sse("done", {})
            return

        # --- BRANCH B: semantic retrieval --------------------------------
        # If the question names a person ("summarise the profile of X"),
        # resolve them to a cv_id and search only their CV. Otherwise the other
        # candidates compete for the five slots and the summary comes out
        # patchy - a question about one person should read one person's CV.
        cv_ids = None
        if plan.candidate_name:
            frame = load_candidates()
            from ..textutils import fold_accents

            target = fold_accents(plan.candidate_name).casefold()
            hit = frame[
                frame["full_name"].apply(lambda n: target in fold_accents(str(n)).casefold())
            ]
            if not hit.empty:
                cv_ids = hit["cv_id"].tolist()
            # No match: fall through and search everything, because the model
            # may simply have misread a name that is not in the corpus.

        # `.result()` collects the embedding started before routing. By now it
        # is almost always finished, so this rarely blocks.
        hits = search(
            req.question,
            top_k=req.top_k or settings.top_k,
            cv_ids=cv_ids,
            query_vector=embedding_future.result(),
        )
        pool.shutdown(wait=False)

        # Stream the answer as the model produces it. At ~18 tok/s this is the
        # difference between a live assistant and a blank screen for 12 s.
        for token in stream_retrieval_answer(req.question, hits, missing_terms=missing, model=model):
            spoken.append(token)
            yield _sse("token", {"t": token})

        # Collapse chunks into one citation per candidate. Five chunks often
        # come from two people, and the user wants to see two sources, not
        # five - but every section that contributed is kept so they can tell
        # which parts of the CV were actually read.
        citations: dict[str, dict] = {}
        for hit in hits:
            entry = citations.setdefault(
                hit.chunk.cv_id,
                {
                    "cv_id": hit.chunk.cv_id,
                    "candidate": hit.chunk.candidate,
                    "source_file": hit.chunk.source_file,
                    "sections": [],
                },
            )
            if hit.chunk.section not in entry["sections"]:
                entry["sections"].append(hit.chunk.section)

        yield _sse(
            "meta",
            {
                "mode": "retrieve",
                "missing_terms": missing,
                "filters": {},
                "chart": None,
                "matched_count": len(citations),
                "citations": list(citations.values()),
                "chunks": [h.as_dict() for h in hits],
                "elapsed_s": round(time.time() - started, 2),
            },
        )
        log_chat(
            req.question,
            tidy_answer("".join(spoken)),
            model=model,
            intent="retrieve",
            elapsed_s=round(time.time() - started, 2),
            citations=[c["candidate"] for c in citations.values()],
            missing_terms=missing,
        )
        yield _sse("done", {})

    # Both handlers record the failure to the same transcript as a success, so
    # "what did the user ask and what came back" is answerable from one file
    # whichever way the request ended.
    except IndexNotBuilt as exc:
        log_chat(
            req.question,
            tidy_answer("".join(spoken)),
            model=model,
            intent=plan.intent if plan else "",
            elapsed_s=round(time.time() - started, 2),
            error=str(exc),
        )
        yield _sse("error", {"message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - never leave the UI hanging
        log.exception("chat failed")
        log_chat(
            req.question,
            tidy_answer("".join(spoken)),
            model=model,
            intent=plan.intent if plan else "",
            elapsed_s=round(time.time() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
        )
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
