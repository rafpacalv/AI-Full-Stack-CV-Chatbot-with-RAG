"""Verify that the literal terms a user asked for actually exist in the corpus.

The problem this solves, observed in use: asked *"¿Qué candidatos tienen
conocimientos en CNN?"* - a term that appears in **zero** CVs - the system
answered *"Guillem Roca Prats: menciona Computer Vision. Las CNN son una
arquitectura comúnmente utilizada en Computer Vision."*

Nothing in that is factually false, but the headline claim is an overclaim: it
attributes a skill to a candidate who never listed it, by inferring from an
adjacent technology. A recruiter skimming the answer takes away "Guillem knows
CNNs". Worse, the behaviour was inconsistent - the same question in English was
correctly refused, so whether the user got a warning depended on the language
they asked in.

The cause is structural. Dense retrieval is *supposed* to match meaning rather
than characters: "CNN" and "computer vision" are genuinely close in embedding
space, so the retriever is working exactly as designed. But when a recruiter
types a specific technology, they mean that technology, and semantic
neighbourhood is not a substitute for the CV saying so.

So this module checks the corpus, not the retrieved chunks. "Absent from the
five chunks we happened to retrieve" is weak and could just mean poor
retrieval; "absent from all 375 chunks" is a definitive statement that can be
shown to the user and enforced in the prompt.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..ingest.index import tokenize
from ..textutils import fold_accents

# Acronyms and product names the user expects to find verbatim: CNN, UPC, AWS,
# ETL, CI/CD, K8S. Two to eight characters, allowing internal dots and slashes.
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}(?:[./][A-Z0-9]{1,7})?\b")

# Tokens with tech punctuation that survive tokenisation: node.js, c++, scikit-learn.
TECHY_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[.+#-][a-zA-Z0-9+#]+)+\b")

# Uppercase words that are ordinary language, not technologies. Without this a
# question typed in capitals, or a sentence-initial "QUIEN", trips the check.
ACRONYM_STOPWORDS = {
    "CV", "CVS", "IA", "AI", "EL", "LA", "LOS", "LAS", "UN", "UNA", "DE", "EN",
    "QUE", "QUIEN", "COMO", "CUAL", "WHO", "WHAT", "HOW", "AND", "THE", "FOR",
    "OR", "TO", "IN", "IS", "ARE", "ME", "MI", "SU", "AL", "SI", "NO", "OK",
}


@lru_cache(maxsize=1)
def corpus_vocabulary() -> frozenset[str]:
    """Every token that appears anywhere in the indexed CVs.

    Built with the same tokenizer BM25 uses, so "present in the vocabulary" and
    "findable by lexical search" mean exactly the same thing - no chance of
    telling the user a term is absent when the retriever can actually match it.
    """
    from .retrieve import load_index

    vocabulary: set[str] = set()
    for chunk in load_index().chunks:
        vocabulary.update(tokenize(chunk.embedding_text))
    return frozenset(vocabulary)


def extract_query_terms(question: str, *, extra: str = "") -> list[str]:
    """Pull out the terms the user expects to be matched literally.

    ``extra`` is the skill the router extracted, which catches spelled-out
    names ("convolutional neural networks") that no acronym pattern would.
    """
    terms: list[str] = []

    for match in ACRONYM_RE.findall(question):
        if match.upper() not in ACRONYM_STOPWORDS:
            terms.append(match)

    terms.extend(TECHY_RE.findall(question))

    if extra:
        terms.append(extra)

    # De-duplicate case-insensitively, preserving the user's own spelling.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = fold_accents(term).casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def missing_from_corpus(question: str, *, skill: str = "") -> list[str]:
    """Terms from the question that appear in no CV at all.

    A term counts as present if *every* token it tokenises to is in the corpus
    vocabulary. Multi-word terms are handled that way so "machine learning"
    passes only when both words occur, while a single unknown token like "cnn"
    is reported.
    """
    vocabulary = corpus_vocabulary()
    absent: list[str] = []

    for term in extract_query_terms(question, extra=skill):
        tokens = tokenize(term)
        if not tokens:
            continue
        if not all(token in vocabulary for token in tokens):
            absent.append(term)
    return absent


def warning_sentence(missing: list[str], language: str) -> str:
    """A prompt fragment telling the model what it must not infer."""
    if not missing:
        return ""
    quoted = ", ".join(f'"{term}"' for term in missing)
    if language == "es":
        return (
            f"\n\nAVISO IMPORTANTE: {quoted} no aparece en NINGÚN CV de la base de "
            "datos. Empieza la respuesta diciéndolo claramente. NO atribuyas ese "
            "conocimiento a ningún candidato, y NO lo deduzcas a partir de "
            "tecnologías relacionadas. Puedes, después, mencionar qué candidatos "
            "trabajan en áreas próximas, dejando claro que es una sugerencia tuya "
            "y no algo que ponga en su CV."
        )
    return (
        f"\n\nIMPORTANT: {quoted} appears in NO CV in the database. Begin your "
        "answer by saying so plainly. Do NOT attribute that knowledge to any "
        "candidate, and do NOT infer it from related technologies. You may then "
        "mention which candidates work in adjacent areas, making clear that this "
        "is your suggestion and not something their CV states."
    )
