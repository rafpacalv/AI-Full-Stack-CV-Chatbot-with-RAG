"""Tests for the parts that are deterministic and cheap to check.

Anything that needs a live model is skipped when Ollama is not reachable, so
`pytest` stays fast and green on a machine without it. The retrieval tests do
need the built index - they are the ones that actually pin down the claims this
project makes about cross-lingual and exact-token search.
"""

from __future__ import annotations

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
