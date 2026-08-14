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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..llm import client
from ..rag.aggregate import candidates_summary, load_candidates, run_aggregate
from ..rag.answer import stream_aggregate_answer, stream_retrieval_answer
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

# The Streamlit UI is a separate process on another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str | None = Field(default=None, description="Override the chat model")
    top_k: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 8


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
    """Serve the original PDF, so a citation can open the real document."""
    matches = sorted(settings.cvs_dir.glob(f"{cv_id}_*.pdf"))
    if not matches:
        raise HTTPException(404, f"No PDF for {cv_id}")
    return FileResponse(matches[0], media_type="application/pdf", filename=matches[0].name)


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
    return route(req.question)


def _chat_events(req: ChatRequest) -> Iterator[str]:
    started = time.time()
    if req.model:
        settings.chat_model = req.model

    try:
        # Routing (chat model) and query embedding (embedding model) are
        # independent, and both models are resident in VRAM at once. Running
        # them concurrently rather than back-to-back roughly halves the time
        # before the first token appears - ~13s down to ~7s on this machine.
        pool = ThreadPoolExecutor(max_workers=1)
        embedding_future = pool.submit(embed_query, req.question)
        try:
            plan = route(req.question)
        except Exception:
            embedding_future.cancel()
            pool.shutdown(wait=False)
            raise
        yield _sse("plan", plan.model_dump())

        # --- aggregate / chart: exact answers over the whole table -------
        if plan.intent in ("aggregate", "chart"):
            embedding_future.cancel()  # this branch never searches
            pool.shutdown(wait=False)
            result = run_aggregate(plan)
            for token in stream_aggregate_answer(req.question, result):
                yield _sse("token", {"t": token})

            frame = result.matched
            yield _sse(
                "meta",
                {
                    "mode": plan.intent,
                    "filters": result.filters,
                    "chart": result.chart,
                    "matched_count": int(len(frame)),
                    "citations": [
                        {
                            "cv_id": row["cv_id"],
                            "candidate": row["full_name"],
                            "source_file": row["source_file"],
                            "sections": [],
                        }
                        for _, row in frame.head(12).iterrows()
                    ],
                    "chunks": [],
                    "elapsed_s": round(time.time() - started, 2),
                },
            )
            yield _sse("done", {})
            return

        # --- retrieval ---------------------------------------------------
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

        hits = search(
            req.question,
            top_k=req.top_k or settings.top_k,
            cv_ids=cv_ids,
            query_vector=embedding_future.result(),
        )
        pool.shutdown(wait=False)
        for token in stream_retrieval_answer(req.question, hits):
            yield _sse("token", {"t": token})

        # One citation per candidate, keeping every section that contributed.
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
                "filters": {},
                "chart": None,
                "matched_count": len(citations),
                "citations": list(citations.values()),
                "chunks": [h.as_dict() for h in hits],
                "elapsed_s": round(time.time() - started, 2),
            },
        )
        yield _sse("done", {})

    except IndexNotBuilt as exc:
        yield _sse("error", {"message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - never leave the UI hanging
        log.exception("chat failed")
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
