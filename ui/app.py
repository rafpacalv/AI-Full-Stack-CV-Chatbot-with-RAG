"""LeadTech CV Screener - Streamlit front end.

Talks to the FastAPI backend over Server-Sent Events so answers stream in as the
local model produces them. Four tabs: the chat itself, the insights it can
compute, the pipeline behind both, and the log of everything asked of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvscreener.branding import (  # noqa: E402
    AMBER,
    GREY_MUTED,
    GREY_TEXT,
    PINK,
    SKY,
    role_group,
)
from cvscreener.config import settings  # noqa: E402
from cvscreener.rag.answer import tidy_answer  # noqa: E402

import charts  # noqa: E402
from theme import (  # noqa: E402
    inject_css,
    masthead,
    metric,
    render_markdown,
    role_tiles,
    sidebar_logo,
)

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

# How many source names to show as chips before collapsing the rest into a
# count. The full set is always listed in the expander underneath.
CHIP_LIMIT = 8

# The filters worth showing the user, in reading order. Not derived from
# `router.CARRIED_FIELDS`: that one governs what a follow-up inherits, and the
# two answering different questions is why `min_years` is inherited but not
# displayed as a chip - ">= 5" beside a name reads as noise.
FILTER_KEYS = ("skills", "seniority", "city", "university", "candidate_name")

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


def api_logs(limit: int = 200) -> list[dict]:
    """The question/answer transcript. Deliberately not cached.

    Everything else here is behind `st.cache_data`, but a log you have to wait
    60 seconds to see updating is not a log. It is a local file read, so the
    cost of skipping the cache is nothing.
    """
    try:
        r = httpx.get(f"{API}/logs", params={"limit": limit}, timeout=15)
        return r.json()["records"] if r.status_code == 200 else []
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def api_clear_logs() -> int | None:
    """Empty the transcript. Returns how many records went, or None on failure."""
    try:
        r = httpx.delete(f"{API}/logs", timeout=15)
        return r.json()["removed"] if r.status_code == 200 else None
    except (httpx.HTTPError, KeyError, ValueError):
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_candidates() -> pd.DataFrame:
    try:
        r = httpx.get(f"{API}/candidates", timeout=15)
        return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    except httpx.HTTPError:
        return pd.DataFrame()


def stream_chat(question: str, model: str, context: dict | None = None):
    """Yield ('plan'|'token'|'meta'|'error', payload) tuples from the SSE stream.

    ``context`` is the previous turn - its question and the plan the router
    resolved for it - so a follow-up like "now chart those by seniority" knows
    who "those" are. The API keeps no session state, so the conversation only
    exists for as long as this client keeps sending it: "New query" simply
    stops.
    """
    payload: dict = {"question": question, "model": model}
    if context:
        payload["previous_question"] = context["question"]
        payload["previous_plan"] = context["plan"]
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


def new_query() -> None:
    """Start again with no conversational context.

    `context` is the important one: it is what makes the next question a
    follow-up, so dropping it is the whole point of the button. The messages go
    with it deliberately - keeping the transcript on screen while silently
    forgetting it would leave the UI showing a conversation the model can no
    longer see.

    The retrieval trace and last chart go too, because the Pipeline tab reads
    `last_meta` and would otherwise show the trace of a question that is no
    longer anywhere. The Logs tab is untouched: that is the permanent record,
    and it has its own, separate, two-step clear.
    """
    st.session_state.messages = []
    st.session_state.last_chart = None
    st.session_state.context = None
    st.session_state.pop("last_meta", None)
    st.session_state.pop("pending", None)


def filter_text(value: object, joiner: str = ", ") -> str:
    """One filter value as a chip reads it.

    `skills` is a list, and the rest are scalars, so this is the one place that
    knows the difference - a chip saying "skills: ['python', 'node.js']" leaks
    Python syntax at the user.

    The joiner carries the operator. A chip reading "skills: java, python" beside
    an answer about 20 people is unreadable until it says "java or python".
    """
    if isinstance(value, list):
        return joiner.join(str(v) for v in value)
    return str(value) if value else ""


def routing_badge(plan: dict) -> str:
    """The routing decision, as the row of chips above an answer.

    Built from the `plan` SSE payload and kept on the message, so it survives
    the rerun that commits each turn. It used to be written straight into a
    live `st.empty()`, which meant the moment the transcript was re-rendered
    the explanation of *how* the answer was reached vanished - leaving the
    answer with nothing to justify it.
    """
    intent = plan.get("intent", "retrieve")
    bits = [f"<span class='lt-badge lt-badge-{intent}'>{BADGE_TEXT.get(intent, intent)}</span>"]

    # Read as a follow-up: the filter chips beside this one may include
    # something the user did not say in this question, so say where they
    # came from.
    if plan.get("follow_up"):
        bits.append("<span class='lt-chip-muted'>follows the previous question</span>")

    # Any filter the router extracted, so its reasoning is visible rather than
    # hidden behind the answer.
    joiner = " or " if plan.get("skill_match") == "any" else ", "
    for key in FILTER_KEYS:
        if value := filter_text(plan.get(key), joiner):
            bits.append(f"<span class='lt-chip-muted'>{key}: {value}</span>")

    # Terms that appear in no CV. Shown as a warning chip so the absence is
    # visible at a glance, not buried in the prose - a recruiter searching for a
    # specific technology needs to know immediately that nobody lists it.
    for term in plan.get("missing_terms") or []:
        bits.append(f"<span class='lt-chip-warn'>not in any CV: {term}</span>")

    # Asked to plot a field the candidate table does not hold. Said before the
    # answer streams, so the user is not left waiting for a figure that is
    # never coming.
    if plan.get("chart_unavailable"):
        bits.append(
            "<span class='lt-chip-warn'>field not in the database &mdash; no chart</span>"
        )
    return " ".join(bits)


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown(
        f"<div style='color:{GREY_MUTED};font-size:12px;margin-top:10px;"
        "text-transform:uppercase;letter-spacing:.07em'>Sources</div>",
        unsafe_allow_html=True,
    )
    # An aggregate can match all 50 candidates, and 50 mint chips is both
    # unreadable and several times over the accent budget. The chips are a
    # glance; the expander below is the complete list.
    shown = citations[:CHIP_LIMIT]
    chips = "".join(f"<span class='lt-chip'>{c['candidate']}</span>" for c in shown)
    if len(citations) > CHIP_LIMIT:
        chips += (
            f"<span class='lt-chip-muted'>+{len(citations) - CHIP_LIMIT} more</span>"
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
            # A link, not a download button. `st.download_button` needs the
            # bytes up front, so building this list used to fetch every PDF
            # eagerly on each rerun - fine for five retrieved chunks, ~10 MB of
            # blocking requests once an aggregate can match all 50 candidates.
            # The API already serves the file with a Content-Disposition
            # header, so the browser downloads it on click.
            cols[2].link_button(
                "PDF", f"{API}/cv/{c['cv_id']}", type="primary", help=c["source_file"]
            )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
health = api_health()

with st.sidebar:
    st.markdown(sidebar_logo("CV Screener"), unsafe_allow_html=True)

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

    # A real control, not a label. `settings.chat_model` is only the default;
    # the choice made here travels with each request as `ChatRequest.model` and
    # is threaded down to the router and both answer functions. It is also the
    # cheapest demonstration that nothing in the pipeline is tied to one model.
    #
    # It must stay a per-request argument. This used to assign
    # `settings.chat_model`, which made one user's pick the whole process's
    # default - see test_a_model_override_does_not_leak_between_requests.
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
    if st.button("New query", use_container_width=True, key="new-query-sidebar"):
        new_query()
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
masthead("AI-powered CV screening · bilingual RAG on local Ollama")

tab_chat, tab_insights, tab_pipeline, tab_logs = st.tabs(
    ["Chat", "Insights", "Pipeline", "Logs"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chart" not in st.session_state:
    st.session_state.last_chart = None
# The previous turn's question and resolved plan, or None for a fresh start.
if "context" not in st.session_state:
    st.session_state.context = None

with tab_chat:
    # "New query" sits with the conversation, not only in the sidebar, and
    # appears only when there is something to reset.
    #
    # Next to it, the context the next question will be resolved against. The
    # conversation is only useful if the user can see what it currently
    # remembers - an inherited filter that nobody can see is indistinguishable
    # from a bug.
    if st.session_state.messages:
        context_col, button_col = st.columns([5, 1])
        with context_col:
            if carried := st.session_state.context:
                joiner = " or " if carried["plan"].get("skill_match") == "any" else ", "
                filters = ", ".join(
                    f"{k}: {v}"
                    for k in FILTER_KEYS
                    if (v := filter_text(carried["plan"].get(k), joiner))
                )
                st.caption(
                    f"Following on from “{carried['question']}”"
                    + (f" · {filters}" if filters else "")
                )
        with button_col:
            if st.button("New query", key="new-query-chat", use_container_width=True,
                         help="Forget this conversation and start a fresh context"):
                new_query()
                st.rerun()

    # The whole conversation lives in one container declared *before* the input
    # box. Without it the new exchange streams in below `st.chat_input`, which
    # leaves the input stranded in the middle of the transcript until the next
    # rerun shuffles it back to the bottom.
    conversation = st.container()

    with conversation:
        # Streamlit derives a chart's element ID from its type and parameters,
        # so two figures with identical data collide with a
        # StreamlitDuplicateElementId - which is easy to hit here, because a
        # chat answer can legitimately plot the same breakdown the Insights tab
        # already shows. Every chart in this app therefore carries an explicit
        # key, and the position in the transcript is what makes this one unique.
        for turn, msg in enumerate(st.session_state.messages):
            # The badge belongs to the answer and is drawn above it, matching
            # the live order where the routing decision arrives before any text.
            if msg.get("plan"):
                st.markdown(routing_badge(msg["plan"]), unsafe_allow_html=True)
            css = "lt-msg-user" if msg["role"] == "user" else "lt-msg-bot"
            st.markdown(
                f"<div class='{css}'>{render_markdown(msg['content'])}</div>",
                unsafe_allow_html=True,
            )
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg.get("chart"):
                fig = charts.from_spec(msg["chart"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chat-turn-{turn}")

    question = st.chat_input("Ask about the candidates, in English or Spanish...")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        # `with conversation:` makes it the active container, so everything
        # below - including the placeholders and st.caption - lands above the
        # input box rather than after it.
        with conversation:
            st.markdown(
                f"<div class='lt-msg-user'>{render_markdown(question)}</div>",
                unsafe_allow_html=True,
            )

            # Two placeholders reserved up front. `st.empty()` returns a handle
            # that can be rewritten in place, which is how the answer grows
            # smoothly instead of Streamlit appending a new block per token.
            badge_slot = st.empty()
            answer_slot = st.empty()
        answer, meta = "", {}
        resolved_plan = None

        try:
            for event, payload in stream_chat(question, model, st.session_state.context):
                if event == "plan":
                    # Kept for the next turn. This is the *resolved* plan, so it
                    # already carries anything this question inherited - which
                    # is what lets a third question follow on from a second one
                    # without the client having to remember the whole chain.
                    resolved_plan = payload
                    # Arrives first, so the user sees which strategy was chosen
                    # before any text appears.
                    badge_slot.markdown(routing_badge(payload), unsafe_allow_html=True)
                elif event == "token":
                    # Rewrite the whole bubble each token, with a block cursor
                    # on the end to signal that more is coming.
                    answer += payload.get("t", "")
                    answer_slot.markdown(
                        f"<div class='lt-msg-bot'>{render_markdown(answer)}"
                        "<span class='lt-cursor'>▌</span></div>",
                        unsafe_allow_html=True,
                    )
                elif event == "meta":
                    # Citations, retrieval trace and any chart. Sent once, after
                    # the text, because sources are only known after retrieval.
                    meta = payload
                elif event == "error":
                    st.error(payload.get("message", "Unknown error"))
        except httpx.HTTPError as exc:
            st.error(f"Backend error: {exc}")

        # Final repaint: cursor gone, and the answer tidied. The clean-up waits
        # until the stream is complete because it works on whole lines - during
        # streaming a bullet legitimately looks empty for the moment before its
        # text arrives.
        answer = tidy_answer(answer)
        answer_slot.markdown(
            f"<div class='lt-msg-bot'>{render_markdown(answer)}</div>",
            unsafe_allow_html=True,
        )

        with conversation:
            if elapsed := meta.get("elapsed_s"):
                st.caption(f"{elapsed}s · {model}")

            render_citations(meta.get("citations", []))

            if chart_spec := meta.get("chart"):
                fig = charts.from_spec(chart_spec)
                if fig:
                    # Keyed on the turn this chart belongs to, matching the
                    # history loop above, so the live render and its replay on
                    # the next rerun claim the same slot rather than colliding.
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"chat-live-{len(st.session_state.messages)}",
                    )
                    st.session_state.last_chart = chart_spec

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "plan": resolved_plan,
                "citations": meta.get("citations", []),
                "chart": meta.get("chart"),
                "chunks": meta.get("chunks", []),
            }
        )
        st.session_state.last_meta = meta

        # Hand this turn forward. Only when the router actually produced a plan:
        # if the request failed there is nothing meaningful to follow on from,
        # and the previous context is more useful kept than replaced.
        if resolved_plan:
            st.session_state.context = {"question": question, "plan": resolved_plan}
        st.rerun()


with tab_insights:
    frame = api_candidates()
    if frame.empty:
        st.info("No candidate table yet. Run `python -m cvscreener.ingest.index`.")
    else:
        # Sky is this tab's accent. leadtech.com never shows two accents in one
        # viewport - it recolours whole sections instead - so Chat keeps mint,
        # Insights takes sky and Pipeline takes mustard.
        cols = st.columns(4)
        cols[0].markdown(metric(len(frame), "Candidates", SKY), unsafe_allow_html=True)
        cols[1].markdown(
            metric(f"{frame['years_experience'].mean():.1f}", "Avg. years exp.", SKY),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            metric(f"{frame['age'].mean():.0f}", "Avg. age", SKY), unsafe_allow_html=True
        )
        cols[3].markdown(
            metric(frame["cv_language"].nunique(), "Languages", SKY), unsafe_allow_html=True
        )

        # The corpus by discipline, in LeadTech's own 14-colour job code. Roles
        # are bucketed first because `current_role` is re-derived per CV, so the
        # raw values are ~50 near-unique strings.
        st.markdown("")
        st.markdown("###### Corpus by discipline")
        groups = frame["current_role"].apply(role_group).value_counts().to_dict()
        st.markdown(role_tiles(groups), unsafe_allow_html=True)

        st.markdown("")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.histogram(frame["age"].dropna().tolist(), "Age"),
                use_container_width=True,
                key="insights-age",
            )
            counts = frame["seniority"].value_counts()
            st.plotly_chart(
                charts.pie(counts.index.tolist(), counts.tolist(), "Seniority"),
                use_container_width=True,
                key="insights-seniority",
            )
        with right:
            counts = frame["city"].value_counts().head(10)
            st.plotly_chart(
                charts.bar(counts.index.tolist(), counts.tolist(), "City"),
                use_container_width=True,
                key="insights-city",
            )
            skills = frame.explode("skills")["skills"].dropna().value_counts().head(12)
            st.plotly_chart(
                charts.bar(skills.index.tolist(), skills.tolist(), "Skill"),
                use_container_width=True,
                key="insights-skills",
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
            col.markdown(metric(value, label, AMBER), unsafe_allow_html=True)

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
                    f"<div class='lt-source'><b>{c['candidate']}</b> · "
                    f"{c['section']}<br>{c['text'][:320]}…</div>",
                    unsafe_allow_html=True,
                )


with tab_logs:
    # Pink is this tab's accent, following the one-accent-per-view rule: Chat
    # is mint, Insights sky, Pipeline mustard. Pink is what leadtech.com uses
    # on /about-us, where mint has no fill presence at all.
    records = api_logs(limit=500)

    if not records:
        st.info(
            "Nothing logged yet. Every question, its answer and any error are "
            "written to `data/logs/chat.jsonl` as one JSON object per line."
        )
    else:
        frame = pd.DataFrame(records)
        failed = frame["error"].notna().sum() if "error" in frame else 0
        latencies = pd.to_numeric(frame.get("elapsed_s"), errors="coerce").dropna()

        cols = st.columns(4)
        cols[0].markdown(metric(len(frame), "Questions", PINK), unsafe_allow_html=True)
        cols[1].markdown(metric(int(failed), "Errors", PINK), unsafe_allow_html=True)
        cols[2].markdown(
            metric(f"{latencies.median():.1f}s" if len(latencies) else "—", "Median", PINK),
            unsafe_allow_html=True,
        )
        # Median, not mean: one cold-start question at 30s drags an average far
        # enough to misrepresent every other question in the file.
        cols[3].markdown(
            metric(frame["intent"].replace("", "?").nunique(), "Intents", PINK),
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("###### Exchanges")
        table = pd.DataFrame(
            {
                # Labelled UTC because that is what is stored, and on a machine
                # two hours ahead an unlabelled 10:15 next to a 12:15 wall clock
                # reads as a bug in the log rather than a timezone.
                "Time (UTC)": frame["ts"].astype(str).str.slice(11, 19),
                "Intent": frame["intent"].replace("", "—"),
                # Absent from records written before follow-ups existed, hence
                # the reindex rather than a bare column read.
                "Follow-up": frame.get(
                    "follow_up", pd.Series(False, index=frame.index)
                ).fillna(False).map({True: "yes", False: ""}),
                "Seconds": pd.to_numeric(frame.get("elapsed_s"), errors="coerce"),
                "Question": frame["question"],
                "Sources": frame["citations"].apply(len),
                "Missing": frame["missing_terms"].apply(lambda t: ", ".join(t) if t else ""),
                "Error": frame["error"].fillna(""),
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True, height=320)

        with st.expander("Read a full exchange"):
            labels = [
                f"{r['ts'][11:19]} · {r['question'][:70]}" for r in records
            ]
            chosen = st.selectbox("Exchange", labels, label_visibility="collapsed")
            record = records[labels.index(chosen)]
            st.markdown(
                f"<div class='lt-msg-user'>{render_markdown(record['question'])}</div>",
                unsafe_allow_html=True,
            )
            if record.get("error"):
                st.error(record["error"])
            if record.get("answer"):
                st.markdown(
                    f"<div class='lt-msg-bot'>{render_markdown(record['answer'])}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        left, right = st.columns([1, 1])
        with left:
            st.download_button(
                "Download JSONL",
                "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(records)),
                file_name="chat.jsonl",
                mime="application/x-ndjson",
                use_container_width=True,
            )
        with right:
            # Two steps, because this cannot be undone and the button sits next
            # to a harmless download. The confirmation replaces the button
            # rather than sitting beside it, so there is nothing to mis-click.
            if st.session_state.get("confirm_clear_logs"):
                if st.button(
                    f"Delete {len(records)} records — confirm",
                    type="primary",
                    use_container_width=True,
                ):
                    removed = api_clear_logs()
                    st.session_state.confirm_clear_logs = False
                    if removed is None:
                        st.error("Could not clear the log.")
                    else:
                        st.toast(f"Cleared {removed} records.")
                        st.rerun()
            elif st.button("Clear log", use_container_width=True):
                st.session_state.confirm_clear_logs = True
                st.rerun()

        if st.session_state.get("confirm_clear_logs"):
            if st.button("Cancel", key="cancel-clear-logs"):
                st.session_state.confirm_clear_logs = False
                st.rerun()

    st.caption(
        "The rotating application log (warmup, Ollama failures, tracebacks) is "
        "kept separately in `data/logs/cvscreener.log` and is not cleared here."
    )
