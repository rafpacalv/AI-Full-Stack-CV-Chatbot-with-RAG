"""Tests for the parts that are deterministic and cheap to check.

Anything that needs a live model is skipped when Ollama is not reachable, so
`pytest` stays fast and green on a machine without it. The retrieval tests do
need the built index - they are the ones that actually pin down the claims this
project makes about cross-lingual and exact-token search.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cvscreener.config import settings
from cvscreener.generation.personas import build_personas
from cvscreener.ingest.chunk import _normalise_heading, chunk_document
from cvscreener.ingest.enrich import detect_language, normalise_skill
from cvscreener.ingest.extract import _find_gutter
from cvscreener.ingest.index import tokenize
from cvscreener.rag.answer import detect_question_language
from cvscreener.textutils import ascii_slug, fold_accents

index_built = pytest.mark.skipif(
    not (settings.index_dir / "embeddings.npy").exists(),
    reason="index not built; run python -m cvscreener.ingest.index",
)


# --- persona matrix -------------------------------------------------------
def test_persona_matrix_is_balanced_and_unique():
    people = build_personas()
    assert len(people) == 50
    assert len({p.full_name for p in people}) == 50
    assert len({p.cv_id for p in people}) == 50
    assert sum(p.language == "es" for p in people) == 25
    assert sum(p.language == "en" for p in people) == 25
    # All three layouts must actually be exercised.
    assert {p.template for p in people} == {0, 1, 2}
    # The brief's sample question ("graduated from UPC?") needs an answer.
    assert sum("UPC" in p.university for p in people) >= 2


def test_persona_matrix_is_reproducible():
    assert [p.full_name for p in build_personas()] == [
        p.full_name for p in build_personas()
    ]


@pytest.mark.parametrize(
    ("role", "gender", "expected"),
    [
        ("Ingeniero/a Backend", "female", "Ingeniera Backend"),
        ("Ingeniero/a Backend", "male", "Ingeniero Backend"),
        ("Desarrollador/a Frontend", "female", "Desarrolladora Frontend"),
        ("Diseñador/a UX/UI", "female", "Diseñadora UX/UI"),
        ("Product Manager", "male", "Product Manager"),
    ],
)
def test_gendered_role_titles(role, gender, expected):
    """The slashed form must never reach the page."""
    person = build_personas()[0]
    assert type(person)(**{**person.__dict__, "role": role, "gender": gender}).role_display == expected


# --- text utilities -------------------------------------------------------
def test_ascii_slug_folds_accents_and_stays_url_safe():
    assert ascii_slug("Núria Badia Roldán") == "nuria-badia-roldan"
    assert ascii_slug("Katarzyna Wilczyńska") == "katarzyna-wilczynska"
    assert ascii_slug("Sinéad O'Halloran") == "sinead-o-halloran"


def test_fold_accents():
    assert fold_accents("Kraków") == "Krakow"


# --- skill normalisation --------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aprendizaje Automático", "machine learning"),
        ("Machine Learning", "machine learning"),
        ("PNL", "nlp"),
        ("Natural Language Processing", "nlp"),
        ("Python (Expertise)", "python"),
        ("  pruebas unitarias ", "unit testing"),
        ("K8s", "kubernetes"),
    ],
)
def test_skill_aliases_collapse_across_languages(raw, expected):
    """Without this, ES and EN CVs would never aggregate together."""
    assert normalise_skill(raw) == expected


# --- language detection ---------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¿Quién tiene experiencia con Python?", "es"),
        ("How many candidates know Kubernetes?", "en"),
        ("Genera un histograma de las edades", "es"),
        ("Summarize the profile of Jane Doe", "en"),
    ],
)
def test_question_language_detection(text, expected):
    assert detect_question_language(text) == expected


def test_document_language_detection():
    assert detect_language("El candidato tiene experiencia en el desarrollo de la web") == "es"
    assert detect_language("The candidate has experience with the development of the web") == "en"


# --- chunking -------------------------------------------------------------
def test_heading_recognition_is_bilingual():
    assert _normalise_heading("EXPERIENCIA PROFESIONAL") == "Experiencia"
    assert _normalise_heading("WORK EXPERIENCE") == "Experience"
    assert _normalise_heading("FORMACIÓN ACADÉMICA") == "Formación"
    # Body text must never be mistaken for a heading.
    assert _normalise_heading("Desarrollé la API principal de la plataforma") is None
    assert _normalise_heading("Python") is None


def test_chunks_carry_their_candidate():
    """A chunk retrieved alone must still say who it is about."""
    chunks = chunk_document(
        cv_id="cv_01",
        source_file="cv_01_test.pdf",
        candidate="Núria Badia Roldán",
        text=(
            "PERFIL PROFESIONAL\n"
            + "Ingeniera backend con 4 años de experiencia en Python y Django. " * 2
            + "\nEXPERIENCIA PROFESIONAL\n"
            + "Desarrollé la API principal, reduciendo la latencia un 30%. " * 2
        ),
    )
    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"Perfil", "Experiencia"}
    for chunk in chunks:
        assert chunk.embedding_text.startswith("Núria Badia Roldán — ")


# --- column detection -----------------------------------------------------
def _words(spans):
    return [
        {"x0": x0, "x1": x1, "top": top, "text": "w"} for x0, x1, top in spans
    ]


def test_gutter_found_in_two_column_layout():
    """Sidebar on the left, main column on the right, clear corridor between."""
    words = _words([(20, 120, t) for t in range(0, 400, 20)])
    words += _words([(220, 500, t) for t in range(0, 400, 20)])
    gutter = _find_gutter(words, page_width=595)
    assert gutter is not None
    assert 120 < gutter < 220


def test_no_gutter_in_single_column_layout():
    """Full-width text crosses every candidate position, so there is no column."""
    words = _words([(20, 560, t) for t in range(0, 400, 20)])
    assert _find_gutter(words, page_width=595) is None


# --- BM25 tokenizer -------------------------------------------------------
def test_tokenizer_folds_accents_and_keeps_tech_tokens():
    assert "formacion" in tokenize("FORMACIÓN ACADÉMICA")
    assert "node.js" in tokenize("Node.js and React")
    assert "ci/cd" in tokenize("CI/CD pipelines")


# --- literal-term verification --------------------------------------------
@pytest.mark.parametrize(
    ("question", "skill", "expected"),
    [
        ("¿Qué candidatos tienen conocimientos en CNN?", "cnn", ["CNN"]),
        ("Who has experience with CNN?", "cnn", ["CNN"]),
        ("Busco alguien que sepa COBOL", "cobol", ["COBOL"]),
        # Present in the corpus: must NOT be flagged.
        ("¿Quién sabe SQL y Docker?", "sql", []),
        ("¿Qué candidatos estudiaron en la UPC?", "", []),
        ("Who knows Kubernetes?", "kubernetes", []),
        ("Who has worked with Node.js?", "node.js", []),
        ("¿Quién sabe aprendizaje automático?", "machine learning", []),
        # No hard terms at all - a name is not a technology.
        ("Resume el perfil de Katarzyna Wilczyńska", "", []),
    ],
)
@index_built
def test_terms_absent_from_the_corpus_are_detected(question, skill, expected):
    """Regression: a question about "CNN" was answered with a Computer Vision CV.

    Dense retrieval returns nearest neighbours whether or not the exact term
    exists, so the model saw plausible context and inferred the skill from an
    adjacent one - claiming a candidate knew CNNs when no CV says so. Worse, it
    was inconsistent: the same question in English was refused correctly.

    Both halves matter here. Missing terms must be caught, and present terms
    must not be flagged, or every answer would carry a spurious warning.
    """
    from cvscreener.rag.keywords import missing_from_corpus

    assert missing_from_corpus(question, skill=skill) == expected


@index_built
def test_warning_is_injected_only_when_something_is_missing():
    from cvscreener.rag.keywords import warning_sentence

    for language in ("es", "en"):
        assert "CNN" in warning_sentence(["CNN"], language)
        assert warning_sentence([], language) == ""


@index_built
def test_absent_term_answer_states_the_absence():
    """The end-to-end behaviour, in both languages.

    Weaker than the unit tests above - it matches wording, and wording varies -
    so it checks only for a negation appearing near the term, in either
    language, and leaves the precise phrasing to the model.
    """
    from cvscreener.llm import client
    from cvscreener.rag.answer import stream_retrieval_answer
    from cvscreener.rag.keywords import missing_from_corpus
    from cvscreener.rag.retrieve import search

    if not client.is_up():
        pytest.skip("Ollama not reachable")

    negations = ("no aparece", "ningun", "not mentioned", "no cv", "does not", "not appear")
    for question in ("¿Qué candidatos tienen conocimientos en CNN?", "Who has experience with CNN?"):
        missing = missing_from_corpus(question, skill="cnn")
        assert missing == ["CNN"]
        answer = "".join(
            stream_retrieval_answer(question, search(question, top_k=5), missing_terms=missing)
        )
        folded = fold_accents(answer).casefold()
        assert any(n in folded for n in negations), f"no absence stated: {answer[:200]}"


# --- retrieval (needs the built index) ------------------------------------
@index_built
def test_exact_token_query_finds_upc_graduates():
    from cvscreener.rag.retrieve import search

    hits = search("¿Qué candidatos estudiaron en la UPC?", top_k=5)
    assert hits, "expected UPC graduates to be retrieved"
    assert any(h.bm25_rank is not None for h in hits), "BM25 should fire on an acronym"


@index_built
def test_cross_lingual_retrieval_crosses_the_language_barrier():
    """A Spanish question must be able to reach an English CV.

    This is the single claim the whole bilingual design rests on, so it is
    asserted rather than assumed.
    """
    from cvscreener.rag.retrieve import search

    hits = search("¿Quién sabe posicionamiento en buscadores?", top_k=8)
    languages = {h.chunk.metadata.get("language") for h in hits}
    assert "en" in languages, f"expected an English CV among the hits, got {languages}"


@index_built
@pytest.mark.parametrize(
    "question",
    [
        "¿Quién sabe posicionamiento en buscadores?",
        "¿Quién ha trabajado con almacenes de datos en la nube?",
        "¿Qué candidato se dedica al diseño de producto y accesibilidad?",
    ],
)
def test_competitive_english_cvs_are_not_suppressed(question):
    """Regression: BM25 cannot cross languages, and RRF used to punish that.

    A lexical retriever scores zero against every chunk in the other language,
    so an English chunk could only ever collect half the fused mass of a Spanish
    one. That suppressed correct answers - for the product-design question the
    best dense match in the entire index was an English CV that did not make the
    results at all.
    """
    from cvscreener.rag.retrieve import search

    hits = search(question, top_k=6)
    languages = {h.chunk.metadata.get("language") for h in hits}
    assert "en" in languages, f"English CVs suppressed for {question!r}"


@index_built
@pytest.mark.parametrize(
    "question",
    [
        "¿Quién ha trabajado en la NASA?",
        "recetas de paella valenciana con conejo",
        "Who has experience piloting a rescue helicopter?",
        "best recipes for valencian paella",
    ],
)
def test_off_topic_questions_name_nobody(question):
    """Refusing is the generator's job, not the retriever's.

    Retrieval always returns its top-k, and the cross-lingual ratio gate does
    not filter relevance - an off-topic query scores ratios just as high as a
    good one. So the guarantee that matters lives in the answer prompt.

    The assertion is deliberately *not* a search for refusal phrases. A first
    version of this test matched on wordings like "no hay" / "no consta" and so
    only understood Spanish: the model refuses an English question perfectly
    well with "None of the provided CV excerpts mention...", and the test called
    that a failure.

    What actually matters is not how the refusal is phrased but that no
    candidate is attributed something they never claimed. So that is what is
    checked, and it is language-independent.
    """
    from cvscreener.llm import client
    from cvscreener.rag.aggregate import load_candidates
    from cvscreener.rag.answer import stream_retrieval_answer
    from cvscreener.rag.retrieve import search

    if not client.is_up():
        pytest.skip("Ollama not reachable")

    hits = search(question, top_k=5)
    assert hits, "the retriever should still return its top-k"

    answer = "".join(stream_retrieval_answer(question, hits))
    folded = fold_accents(answer).casefold()

    named = [
        name
        for name in load_candidates()["full_name"]
        if fold_accents(str(name)).casefold() in folded
    ]
    assert not named, f"off-topic answer attributed content to {named}: {answer[:200]}"


@index_built
def test_rrf_scores_are_ordered_and_bounded():
    from cvscreener.config import settings as cfg
    from cvscreener.rag.retrieve import search

    hits = search("Python backend", top_k=5)
    scores = [h.rrf_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Two retrievers, best possible rank 0 each.
    assert all(0 < s <= 2 / (cfg.rrf_k + 1) for s in scores)


@index_built
def test_aggregate_counts_match_the_source_table():
    """The aggregate path must count the whole corpus, not a top-k slice."""
    from cvscreener.rag.aggregate import load_candidates, run_aggregate
    from cvscreener.rag.router import QueryPlan

    frame = load_candidates()
    expected = frame["skills_normalised"].apply(lambda s: "python" in list(s)).sum()

    result = run_aggregate(QueryPlan(intent="aggregate", skill="Python", dimension="skills"))
    assert len(result.matched) == expected
    assert len(frame) == 50


# --------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "role,expected",
    [
        # Both languages, all seniority suffixes the extractor actually emits.
        ("Ingeniera Backend (Lead)", "Backend"),
        ("Full-Stack Engineer (Junior)", "Backend"),
        ("Senior Frontend Engineer", "Frontend & Mobile"),
        ("Desarrollador Mobile Senior", "Frontend & Mobile"),
        ("Ingeniero de Datos (Junior)", "Data & BI"),
        ("Junior Business Intelligence Analyst", "Data & BI"),
        ("Senior Machine Learning Engineer", "Machine Learning"),
        ("Site Reliability Engineer", "DevOps"),
        ("Ingeniera DevOps Junior", "DevOps"),
        ("Senior SEO Specialist", "SEO"),
        ("Growth / Paid Media Manager (Lead)", "SEM & Growth"),
        ("Diseñador UX/UI Senior", "UX & Design"),
        ("QA Automation Engineer (Lead)", "QA"),
        ("Product Manager Lead", "Product"),
        # Ordering traps: the first matching pattern wins, so "Product Manager"
        # must be tested before "design" and "Data Engineer" before "backend".
        ("Product Designer (Lead)", "UX & Design"),
        ("Senior Product Manager", "Product"),
        ("Underwater Basket Weaver", "Other"),
        (None, "Other"),
    ],
)
def test_role_group_buckets_free_text_titles(role, expected):
    from cvscreener.branding import role_group

    assert role_group(role) == expected


def test_every_candidate_falls_in_a_named_group():
    """No CV should land in the catch-all: the strip is a summary, not a shrug."""
    from cvscreener.branding import role_group
    from cvscreener.rag.aggregate import load_candidates

    groups = load_candidates()["current_role"].apply(role_group)
    assert "Other" not in set(groups)
    assert len(groups) == 50


def test_group_colours_are_distinct_and_from_the_brand_set():
    """Two groups sharing a colour would make the tile strip lie."""
    from cvscreener.branding import DISCIPLINE_COLOURS, _GROUP_DISCIPLINE, group_colour

    backgrounds = [group_colour(group)[0] for group in _GROUP_DISCIPLINE]
    assert len(set(backgrounds)) == len(backgrounds)
    assert set(backgrounds) <= set(DISCIPLINE_COLOURS.values())


def test_the_palette_holds_up_against_the_black_canvas():
    """The chart sequence was validated against #140C29, then the surface moved.

    Only the contrast check depends on the surface, and black can only raise it,
    but "can only raise it" is worth asserting rather than assuming.
    """
    from cvscreener.branding import CHART_SEQUENCE_DARK, CHART_SURFACE

    def relative_luminance(hex_colour: str) -> float:
        channels = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    surface = relative_luminance(CHART_SURFACE)
    for colour in CHART_SEQUENCE_DARK:
        ratio = (relative_luminance(colour) + 0.05) / (surface + 0.05)
        assert ratio >= 4.5, f"{colour} only reaches {ratio:.2f}:1 on {CHART_SURFACE}"


# --------------------------------------------------------------------------
# Chart dimension verification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,dimension",
    [
        ("Genera un histograma de las edades de los candidatos que sepan Python", "age"),
        ("Muestra un gráfico de candidatos por ciudad", "city"),
        ("Reparte por seniority en un gráfico circular", "seniority"),
        ("Pie chart of candidates by university", "university"),
        ("Bar chart of candidates by skill", "skills"),
        ("Gráfico de candidatos por idioma del CV", "cv_language"),
        ("Distribución por años de experiencia", "years_experience"),
        ("Gráfico de candidatos por puesto", "current_role"),
    ],
)
def test_legitimate_dimensions_are_accepted(question, dimension):
    """The guard must not block the charts that do work."""
    from cvscreener.rag.router import dimension_supported_by

    assert dimension_supported_by(question, dimension)


@pytest.mark.parametrize(
    "question",
    [
        "Make me a pie chart of the candidates by gender",
        "Haz un diagrama sectorial de los candidatos por género",
        "Gráfico circular por sexo",
        "Pie chart of candidates by salary",
    ],
)
def test_absent_fields_match_no_dimension(question):
    """A field the table does not hold must not look like any field it does.

    The bug this covers: `Dimension` is an enum in the JSON Schema, so
    constrained decoding cannot emit "gender" and returns the nearest legal
    value instead - measured as `cv_language` in English and `seniority` in
    Spanish. Every legal dimension has to reject these questions, not just the
    two that happened to be picked.
    """
    from cvscreener.rag.router import DIMENSION_CUES, dimension_supported_by

    matched = [d for d in DIMENSION_CUES if dimension_supported_by(question, d)]
    assert not matched, f"{question!r} was read as {matched}"


def test_unsupported_dimension_produces_no_chart():
    """Withholding the figure is the point: a wrong chart outranks no chart."""
    from cvscreener.rag.aggregate import run_aggregate
    from cvscreener.rag.router import QueryPlan

    plan = QueryPlan(intent="chart", chart_type="pie", dimension="unsupported")
    result = run_aggregate(plan)
    assert result.chart is None
    # The text answer is still computed - only the plot is withheld.
    assert result.text
    assert len(result.matched) == 50


@pytest.mark.parametrize(
    "question",
    [
        "Make me a pie chart of the candidates by gender",
        "Genera un diagrama sectorial por género de los candidatos",
    ],
)
def test_router_refuses_to_chart_a_field_it_does_not_have(question):
    from cvscreener.llm import client
    from cvscreener.rag.router import route

    if not client.is_up():
        pytest.skip("Ollama not reachable")

    plan = route(question)
    assert plan.dimension == "unsupported", f"routed to {plan.dimension}"


def test_chart_answers_carry_the_exact_breakdown():
    """The narrator must be handed the counts, never left to derive them.

    Without this the prompt for a chart question said only "there are 50
    candidates" and attached a table truncated to 40 rows, so gemma2 counted the
    rows itself: it reported 40% Junior / 50% Mid-level / 10% Senior against a
    true 20 / 28 / 52. The chart was right and the prose beside it was invented.
    """
    from cvscreener.rag.aggregate import load_candidates, run_aggregate
    from cvscreener.rag.router import QueryPlan

    frame = load_candidates()
    truth = frame["seniority"].value_counts()

    result = run_aggregate(
        QueryPlan(intent="chart", chart_type="pie", dimension="seniority")
    )
    assert result.chart is not None
    # The payload the UI plots and the sentence the model is given must agree
    # with the table, and with each other.
    plotted = dict(zip(result.chart["categories"], result.chart["values"]))
    assert plotted == truth.to_dict()
    for level, count in truth.items():
        assert f"{level} {count}" in result.text


def test_truncated_tables_announce_themselves():
    """An unannounced truncation invites the model to count the rows shown."""
    from cvscreener.rag.aggregate import load_candidates
    from cvscreener.rag.answer import _table_context

    frame = load_candidates()
    assert len(frame) > 10
    assert "truncated" in _table_context(frame, limit=10)
    assert "truncated" not in _table_context(frame, limit=len(frame))


# --------------------------------------------------------------------------
# University membership
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,stored",
    [
        # The same institution, however either side happens to be written.
        ("UPC", "UPC"),
        ("upc", "UPC"),
        ("Universitat Politècnica de Catalunya", "UPC"),
        ("universitat politecnica de catalunya", "UPC"),
        # The extractor read each CV alone and normalised inconsistently, so
        # these pairs are one university stored two ways.
        ("TUB", "Technische Universität Berlin"),
        ("Technische Universität Berlin", "TUB"),
        ("Deusto", "Universidad de Deusto"),
        ("Danmarks Tekniske Universitet (DTU)", "DTU"),
    ],
)
def test_university_forms_are_treated_as_one(query, stored):
    from cvscreener.rag.aggregate import _university_matches

    assert _university_matches(query, stored)


@pytest.mark.parametrize(
    "query,stored",
    [
        # Near-miss acronyms that share a prefix must stay apart.
        ("UPC", "UPF"),
        ("UPC", "UPM"),
        ("UPM", "UPV"),
        ("UB", "UAB"),
        # Substring traps: "UB" sits inside "TUB", "US" inside many names.
        ("TUB", "UB"),
        ("UB", "TUB"),
        ("US", "Universitat Politecnica de Catalunya"),
    ],
)
def test_different_universities_stay_apart(query, stored):
    from cvscreener.rag.aggregate import _university_matches

    assert not _university_matches(query, stored)


def test_upc_filter_matches_the_source_table():
    """The brief's own sample question, answered exhaustively.

    Six of the 50 studied at UPC, so top-k=5 retrieval cannot answer it in
    full - it is set membership over a recorded field. Before this the router
    was taught (by a few-shot example of mine) to send it to `retrieve`, and
    the reply was "all candidates listed graduated from UPC", naming nobody.
    """
    from cvscreener.rag.aggregate import _university_matches, load_candidates, run_aggregate
    from cvscreener.rag.router import QueryPlan

    frame = load_candidates()
    expected = {
        row["full_name"]
        for _, row in frame.iterrows()
        if _university_matches("UPC", row["university"])
    }
    assert len(expected) == 6

    result = run_aggregate(QueryPlan(intent="aggregate", university="UPC"))
    assert set(result.matched["full_name"]) == expected
    assert result.filters == {"university": "UPC"}


def test_every_stored_university_finds_at_least_its_own_rows():
    """No filter may lose a candidate it should have matched."""
    from cvscreener.rag.aggregate import load_candidates, run_aggregate
    from cvscreener.rag.router import QueryPlan

    frame = load_candidates()
    for university in frame["university"].dropna().unique():
        own = set(frame.loc[frame["university"] == university, "full_name"])
        got = set(run_aggregate(QueryPlan(intent="aggregate", university=university)).matched["full_name"])
        assert own <= got, f"{university!r} lost {own - got}"


@pytest.mark.parametrize(
    "question",
    ["Which candidate graduated from UPC?", "¿Qué candidatos estudiaron en la UPC?"],
)
def test_university_questions_route_to_the_whole_table(question):
    from cvscreener.llm import client
    from cvscreener.rag.router import route

    if not client.is_up():
        pytest.skip("Ollama not reachable")

    plan = route(question)
    assert plan.intent == "aggregate", f"routed to {plan.intent}"
    assert plan.university, "no university filter extracted"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
@pytest.fixture
def transcript(tmp_path, monkeypatch):
    """Point settings.logs_dir at a temp dir for the duration of a test.

    logs_dir is a property, so it is patched on the class rather than on the
    singleton instance.
    """
    from cvscreener.config import Settings

    monkeypatch.setattr(Settings, "logs_dir", property(lambda self: tmp_path))
    return tmp_path / "chat.jsonl"


def test_a_question_and_its_answer_are_recorded(transcript):
    import json

    from cvscreener.logs import log_chat

    log_chat(
        "¿Qué candidatos estudiaron en la UPC?",
        "Seis candidatos: Núria Badia Roldán y otros.",
        model="gemma2:9b",
        intent="aggregate",
        elapsed_s=4.2,
        citations=["Núria Badia Roldán"],
    )

    record = json.loads(transcript.read_text(encoding="utf-8").strip())
    assert record["question"] == "¿Qué candidatos estudiaron en la UPC?"
    assert "Núria Badia Roldán" in record["answer"]
    assert record["intent"] == "aggregate"
    assert record["error"] is None
    assert record["ts"].endswith("+00:00")


def test_failures_are_recorded_too(transcript):
    """A question that errored must still be answerable from the transcript."""
    import json

    from cvscreener.logs import log_chat

    log_chat("boom", "", model="gemma2:9b", error="ConnectError: Ollama unreachable")

    record = json.loads(transcript.read_text(encoding="utf-8").strip())
    assert record["error"] == "ConnectError: Ollama unreachable"
    assert record["question"] == "boom"


def test_records_are_one_json_object_per_line(transcript):
    import json

    from cvscreener.logs import log_chat

    for i in range(3):
        log_chat(f"q{i}", f"a{i}\nwith a newline", model="m")

    lines = transcript.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, "an embedded newline must not split a record"
    assert [json.loads(line)["question"] for line in lines] == ["q0", "q1", "q2"]


def test_logging_never_breaks_a_request(tmp_path, monkeypatch):
    """A log that cannot be written must not take the answer down with it."""
    from cvscreener.config import Settings
    from cvscreener.logs import log_chat

    # A file where the directory should be: mkdir and open both fail.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    monkeypatch.setattr(Settings, "logs_dir", property(lambda self: blocked))

    log_chat("q", "a", model="m")  # must not raise


# --------------------------------------------------------------------------
# API input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cv_id",
    [
        # The vector that actually worked: a backslash is not a URL path
        # separator, so %5C survives Starlette's routing, and on Windows it is
        # still a filesystem separator. GET /cv/..%5Cdecoy served a planted
        # file from outside the corpus with HTTP 200.
        r"..\decoy",
        r"..\..\data\decoy",
        "../decoy",
        "../../etc/passwd",
        # Glob metacharacters would turn one id into a wildcard read.
        "cv_*",
        "cv_0?",
        "cv_[0-9]",
        # A prefix match would pass these: the check has to be a fullmatch.
        "cv_01/../../decoy",
        r"cv_01\..\decoy",
        "",
    ],
)
def test_cv_endpoint_rejects_anything_but_an_identifier(cv_id):
    from fastapi import HTTPException

    from cvscreener.api.main import cv_pdf

    with pytest.raises(HTTPException) as raised:
        cv_pdf(cv_id)
    # 400, never 404: a rejected identifier is a bad request, and answering 404
    # would leak whether the traversed path happened to exist.
    assert raised.value.status_code == 400


@index_built
def test_cv_endpoint_still_serves_a_real_cv():
    """The lock must not also lock out the legitimate caller."""
    from cvscreener.api.main import cv_pdf
    from cvscreener.config import settings

    served = cv_pdf("cv_01")
    assert Path(served.path).parent.resolve() == settings.cvs_dir.resolve()
    assert Path(served.path).suffix == ".pdf"


def test_chat_request_bounds_its_inputs():
    """Unbounded input reaches the embedding model and the prompt builder."""
    from pydantic import ValidationError

    from cvscreener.api.main import MAX_QUESTION_CHARS, MAX_TOP_K, ChatRequest

    ChatRequest(question="fine", top_k=5)  # the ordinary case still works

    with pytest.raises(ValidationError):
        ChatRequest(question="a" * (MAX_QUESTION_CHARS + 1))
    with pytest.raises(ValidationError):
        ChatRequest(question="fine", top_k=MAX_TOP_K + 1)
    with pytest.raises(ValidationError):
        ChatRequest(question="fine", top_k=0)


def test_a_model_override_does_not_leak_between_requests():
    """`settings.chat_model = req.model` made one request poison the next.

    Reproduced against a running server: a request naming a model Ollama does
    not have left /health advertising that model, and the *following* request -
    which sent no model at all - failed with the same 404. The model is now a
    per-request argument, so the global is only ever read as a default.
    """
    import ast
    import inspect
    import textwrap

    from cvscreener.api import main
    from cvscreener.rag.answer import stream_aggregate_answer, stream_retrieval_answer
    from cvscreener.rag.router import route

    # Parsed, not grepped: the function's comments quote the offending line, so
    # a substring check on the source would fail on the explanation of the fix.
    tree = ast.parse(textwrap.dedent(inspect.getsource(main._chat_events)))
    assigned = [
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    ]
    assert "chat_model" not in assigned, "the global is being mutated again"

    # Every consumer must accept the model rather than reach for the global.
    for function in (route, stream_retrieval_answer, stream_aggregate_answer):
        assert "model" in inspect.signature(function).parameters, function.__name__


def test_the_default_model_still_comes_from_settings():
    """Removing the mutation must not remove the default."""
    import inspect

    from cvscreener.api import main

    assert "req.model or settings.chat_model" in inspect.getsource(main._chat_events)
