"""LeadTech CV Screener - Streamlit front end.

Talks to the FastAPI backend over Server-Sent Events so answers stream in as the
local model produces them. Three tabs mirror the three things the brief asks to
be shown: the chat itself, the insights it can compute, and the pipeline behind
both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvscreener.branding import GREY_MUTED, GREY_TEXT, MINT  # noqa: E402
from cvscreener.config import settings  # noqa: E402

import charts  # noqa: E402
from theme import inject_css, masthead, metric  # noqa: E402

API = settings.api_url

st.set_page_config(
    page_title="LeadTech · CV Screener",
    page_icon="🟩",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

SAMPLE_QUESTIONS = [
    "¿Quién tiene experiencia con aprendizaje automático?",
    "Which candidate graduated from UPC?",
    "How many candidates know Kubernetes?",
    "Genera un histograma de las edades de los candidatos que sepan Python",
    "Resume el perfil de Katarzyna Wilczyńska",
    "Muestra un gráfico de candidatos por ciudad",
]

BADGE_TEXT = {
    "retrieve": "Semantic retrieval",
    "aggregate": "Table aggregate",
    "chart": "Chart + aggregate",
}


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
@st.cache_data(ttl=20, show_spinner=False)
def api_health() -> dict | None:
    try:
        return httpx.get(f"{API}/health", timeout=8).json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_stats() -> dict | None:
    try:
        r = httpx.get(f"{API}/stats", timeout=10)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_candidates() -> pd.DataFrame:
    try:
        r = httpx.get(f"{API}/candidates", timeout=15)
        return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    except httpx.HTTPError:
        return pd.DataFrame()


def stream_chat(question: str, model: str):
    """Yield ('plan'|'token'|'meta'|'error', payload) tuples from the SSE stream."""
    payload = {"question": question, "model": model}
    with httpx.stream(
        "POST", f"{API}/chat", json=payload, timeout=httpx.Timeout(600.0, connect=10.0)
    ) as response:
        response.raise_for_status()

        # A minimal SSE parser. The wire format is pairs of lines:
        #     event: token
        #     data:  {"t": "Los"}
        # The event name always arrives first and is remembered until its data
        # line follows, so the two are yielded together as one tuple.
        event = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if event == "done":
                    return
                try:
                    yield event, json.loads(data)
                except json.JSONDecodeError:
                    continue  # never let one malformed frame kill the stream


def clear_conversation() -> None:
    """Reset the chat.

    Clears the retrieval trace and last chart too, not just the messages: the
    Pipeline tab reads `last_meta`, so leaving it behind would show a trace for
    a question no longer on screen.
    """
    st.session_state.messages = []
    st.session_state.last_chart = None
    st.session_state.pop("last_meta", None)
    st.session_state.pop("pending", None)


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown(
        f"<div style='color:{GREY_MUTED};font-size:12px;margin-top:10px;"
        "text-transform:uppercase;letter-spacing:.07em'>Sources</div>",
        unsafe_allow_html=True,
    )
    chips = "".join(
        f"<span class='lt-chip'>{c['candidate']}</span>" for c in citations
    )
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander(f"Open the {len(citations)} source CV(s)"):
        for c in citations:
            cols = st.columns([3, 2, 1])
            cols[0].markdown(f"**{c['candidate']}**")
            sections = ", ".join(c.get("sections") or []) or "—"
            cols[1].markdown(
                f"<span style='color:{GREY_TEXT};font-size:13px'>{sections}</span>",
                unsafe_allow_html=True,
            )
            try:
                pdf = httpx.get(f"{API}/cv/{c['cv_id']}", timeout=15).content
                cols[2].download_button(
                    "PDF", pdf, file_name=c["source_file"],
                    mime="application/pdf", key=f"dl-{c['cv_id']}-{id(c)}",
                )
            except httpx.HTTPError:
                cols[2].caption("n/a")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
health = api_health()

with st.sidebar:
    st.markdown(
        f"<div class='lt-brand' style='font-size:22px'>leadtech<span>;</span></div>"
        f"<div style='color:{GREY_MUTED};font-size:12px;margin-bottom:14px'>CV Screener</div>",
        unsafe_allow_html=True,
    )

    if health is None:
        st.error("Backend unreachable.\n\nStart it with `.\\run.ps1`")
        available = [settings.chat_model]
    else:
        ok = health["status"] == "ok"
        st.markdown(
            f"<span class='lt-badge {"lt-badge-retrieve" if ok else "lt-badge-chart"}'>"
            f"{'connected' if ok else health['status']}</span>",
            unsafe_allow_html=True,
        )
        available = [m for m in health["models"]["available"] if "bge" not in m] or [
            settings.chat_model
        ]

    st.markdown("###### Chat model")
    default = settings.chat_model
    model = st.selectbox(
        "Chat model",
        available,
        index=available.index(default) if default in available else 0,
        label_visibility="collapsed",
        help=(
            "gemma2:9b fits entirely in this GPU's 8 GB and runs at ~18 tok/s. "
            "gemma4:12b is stronger but spills to CPU here, at ~7 tok/s."
        ),
    )

    if health and health.get("index"):
        st.markdown("###### Index")
        st.markdown(
            f"<div style='color:{GREY_TEXT};font-size:13px;line-height:1.7'>"
            f"{health['index']['chunks']} chunks<br>"
            f"{health['index']['dim']}-d · {health['models']['embed']}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("###### Try asking")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"s-{q}", use_container_width=True):
            st.session_state.pending = q

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
masthead("AI-powered CV screening · bilingual RAG on local Ollama")

tab_chat, tab_insights, tab_pipeline = st.tabs(["Chat", "Insights", "Pipeline"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chart" not in st.session_state:
    st.session_state.last_chart = None

with tab_chat:
    # Clear sits with the conversation, not only in the sidebar, and appears
    # only when there is something to clear - an always-visible destructive
    # action on an empty chat is just noise.
    if st.session_state.messages:
        spacer, clear_col = st.columns([6, 1])
        with clear_col:
            if st.button("Clear", key="clear-chat", use_container_width=True,
                         help="Start a new conversation"):
                clear_conversation()
                st.rerun()

    for msg in st.session_state.messages:
        css = "lt-msg-user" if msg["role"] == "user" else "lt-msg-bot"
        st.markdown(f"<div class='{css}'>{msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("citations"):
            render_citations(msg["citations"])
        if msg.get("chart"):
            fig = charts.from_spec(msg["chart"])
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"c{id(msg)}")

    question = st.chat_input("Ask about the candidates, in English or Spanish...")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"<div class='lt-msg-user'>{question}</div>", unsafe_allow_html=True)

        # Two placeholders reserved up front. `st.empty()` returns a handle
        # that can be rewritten in place, which is how the answer grows
        # smoothly instead of Streamlit appending a new block per token.
        badge_slot = st.empty()
        answer_slot = st.empty()
        answer, meta = "", {}

        try:
            for event, payload in stream_chat(question, model):
                if event == "plan":
                    # Routing decision - arrives first, so the user sees which
                    # strategy was chosen before any text appears.
                    intent = payload.get("intent", "retrieve")
                    bits = [
                        f"<span class='lt-badge lt-badge-{intent}'>{BADGE_TEXT.get(intent, intent)}</span>"
                    ]
                    # Show any filter the router extracted, so its reasoning is
                    # visible rather than hidden behind the answer.
                    for key in ("skill", "seniority", "city", "candidate_name"):
                        if payload.get(key):
                            bits.append(f"<span class='lt-chip-muted'>{key}: {payload[key]}</span>")
                    # Terms that appear in no CV. Shown as a warning chip so the
                    # absence is visible at a glance, not buried in the prose -
                    # a recruiter searching for a specific technology needs to
                    # know immediately that nobody lists it.
                    for term in payload.get("missing_terms") or []:
                        bits.append(
                            f"<span class='lt-chip-warn'>not in any CV: {term}</span>"
                        )
                    badge_slot.markdown(" ".join(bits), unsafe_allow_html=True)
                elif event == "token":
                    # Rewrite the whole bubble each token, with a block cursor
                    # on the end to signal that more is coming.
                    answer += payload.get("t", "")
                    answer_slot.markdown(
                        f"<div class='lt-msg-bot'>{answer}▌</div>", unsafe_allow_html=True
                    )
                elif event == "meta":
                    # Citations, retrieval trace and any chart. Sent once, after
                    # the text, because sources are only known after retrieval.
                    meta = payload
                elif event == "error":
                    st.error(payload.get("message", "Unknown error"))
        except httpx.HTTPError as exc:
            st.error(f"Backend error: {exc}")

        # Final repaint without the cursor.
        answer_slot.markdown(f"<div class='lt-msg-bot'>{answer}</div>", unsafe_allow_html=True)

        if elapsed := meta.get("elapsed_s"):
            st.caption(f"{elapsed}s · {model}")

        render_citations(meta.get("citations", []))

        if chart_spec := meta.get("chart"):
            fig = charts.from_spec(chart_spec)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.last_chart = chart_spec

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": meta.get("citations", []),
                "chart": meta.get("chart"),
                "chunks": meta.get("chunks", []),
            }
        )
        st.session_state.last_meta = meta


with tab_insights:
    frame = api_candidates()
    if frame.empty:
        st.info("No candidate table yet. Run `python -m cvscreener.ingest.index`.")
    else:
        cols = st.columns(4)
        cols[0].markdown(metric(len(frame), "Candidates"), unsafe_allow_html=True)
        cols[1].markdown(
            metric(f"{frame['years_experience'].mean():.1f}", "Avg. years exp."),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            metric(f"{frame['age'].mean():.0f}", "Avg. age"), unsafe_allow_html=True
        )
        cols[3].markdown(
            metric(frame["cv_language"].nunique(), "Languages"), unsafe_allow_html=True
        )

        st.markdown("")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.histogram(frame["age"].dropna().tolist(), "Age"),
                use_container_width=True,
            )
            counts = frame["seniority"].value_counts()
            st.plotly_chart(
                charts.pie(counts.index.tolist(), counts.tolist(), "Seniority"),
                use_container_width=True,
            )
        with right:
            counts = frame["city"].value_counts().head(10)
            st.plotly_chart(
                charts.bar(counts.index.tolist(), counts.tolist(), "City"),
                use_container_width=True,
            )
            skills = frame.explode("skills")["skills"].dropna().value_counts().head(12)
            st.plotly_chart(
                charts.bar(skills.index.tolist(), skills.tolist(), "Skill"),
                use_container_width=True,
            )

        st.markdown("###### Candidate table")
        st.dataframe(
            frame[
                [
                    "full_name", "current_role", "seniority", "years_experience",
                    "age", "city", "university", "cv_language",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


with tab_pipeline:
    stats = api_stats()
    if not stats:
        st.info("No index stats yet. Run `python -m cvscreener.ingest.index`.")
    else:
        st.markdown("###### Ingestion")
        cols = st.columns(5)
        for col, (value, label) in zip(
            cols,
            [
                (stats["n_cvs"], "PDFs"),
                (stats["n_chunks"], "Chunks"),
                (stats["embedding_dim"], "Vector dim"),
                (f"{stats['n_chars']:,}", "Characters"),
                (f"{stats['total_seconds']:.0f}s", "Build time"),
            ],
        ):
            col.markdown(metric(value, label), unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            f"""<div class='lt-source'>
            <b>Pipeline</b> &nbsp; PDF → pdfplumber (column-aware) → section chunking →
            <b>{stats['extraction_model']}</b> facts + <b>{stats['embedding_model']}</b> embeddings
            → numpy dense index + BM25 → Reciprocal Rank Fusion<br>
            <b>Languages</b> &nbsp; {stats['languages']} &nbsp;·&nbsp;
            <b>Fact extraction</b> {stats['enrich_seconds']}s &nbsp;·&nbsp;
            <b>Embedding</b> {stats['embed_seconds']}s
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("###### Retrieval trace for the last question")
        chunks = (st.session_state.get("last_meta") or {}).get("chunks", [])
        if not chunks:
            st.caption("Ask something in the Chat tab to see how it was retrieved.")
        else:
            trace = pd.DataFrame(
                [
                    {
                        "Candidate": c["candidate"],
                        "Section": c["section"],
                        "RRF": c["rrf_score"],
                        "Dense rank": c["dense_rank"],
                        "Dense cos": c["dense_score"],
                        "BM25 rank": c["bm25_rank"],
                        "BM25": c["bm25_score"],
                    }
                    for c in chunks
                ]
            )
            st.dataframe(trace, use_container_width=True, hide_index=True)
            st.caption(
                "Dense and BM25 rank independently; RRF fuses them on rank alone, "
                "so the two score scales never need reconciling."
            )
            for c in chunks[:3]:
                st.markdown(
                    f"<div class='lt-source'><b style='color:{MINT}'>{c['candidate']}"
                    f"</b> · {c['section']}<br>{c['text'][:320]}…</div>",
                    unsafe_allow_html=True,
                )
