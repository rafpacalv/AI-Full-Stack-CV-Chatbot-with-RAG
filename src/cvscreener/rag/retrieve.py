"""Hybrid retrieval: dense vectors + BM25, fused by Reciprocal Rank Fusion.

**Why hybrid.** Dense embeddings understand meaning but blur exact tokens; BM25
matches exact tokens but understands nothing. CV screening needs both at once:
"who studied at UPC?" is a literal-token question, while "¿quién sabe
aprendizaje automático?" must reach an English CV that says "machine learning".
Each retriever alone fails one of those.

**Why RRF rather than a weighted score blend.** Measured on this corpus, bge-m3
scores an unrelated ES/EN pair at 0.475 and a genuinely relevant one at 0.580 -
a high floor and a narrow band. BM25 scores, meanwhile, are unbounded and
corpus-dependent. Normalising two such differently-shaped distributions onto a
common scale means inventing constants that would not survive a change of
corpus. RRF sidesteps the problem: it consumes only *ranks*, so both retrievers
contribute on equal footing and no thresholds are tuned.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import settings
from ..ingest.chunk import Chunk
from ..ingest.index import tokenize
from ..llm import client


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "cv_id": self.chunk.cv_id,
            "candidate": self.chunk.candidate,
            "section": self.chunk.section,
            "source_file": self.chunk.source_file,
            "text": self.chunk.text,
            "rrf_score": round(self.rrf_score, 5),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_score": None if self.dense_score is None else round(self.dense_score, 4),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 3),
        }


class IndexNotBuilt(RuntimeError):
    pass


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: object

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@lru_cache(maxsize=1)
def load_index() -> SearchIndex:
    """Load the on-disk index once per process."""
    d: Path = settings.index_dir
    emb_path, chunk_path, bm25_path = (
        d / "embeddings.npy",
        d / "chunks.jsonl",
        d / "bm25.pkl",
    )
    if not (emb_path.exists() and chunk_path.exists() and bm25_path.exists()):
        raise IndexNotBuilt(
            f"Index missing in {d}. Run: python -m cvscreener.ingest.index"
        )

    chunks = [
        Chunk(**json.loads(line))
        for line in chunk_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embeddings = np.load(emb_path)
    with bm25_path.open("rb") as fh:
        bm25 = pickle.load(fh)
    return SearchIndex(chunks=chunks, embeddings=embeddings, bm25=bm25)


def _ranks(scores: np.ndarray, pool: int, *, positive_only: bool = False) -> list[int]:
    """Indices of the ``pool`` highest scores, best first.

    ``positive_only`` drops non-matching documents entirely. BM25 gives a zero
    to every document sharing no term with the query; without this they would
    still collect rank mass from the tail of the pool purely for existing.
    """
    order = np.argsort(-scores)[:pool]
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return list(order)


def _language_balanced_pool(
    dense_scores: np.ndarray, languages: list[str], pool: int
) -> list[int]:
    """Dense candidates, with each language guaranteed a share of the pool.

    Measured on this corpus, a Spanish query scores **zero** BM25 against every
    English chunk - lexical matching cannot cross a language boundary. Only the
    dense retriever can, so RRF systematically halves cross-lingual hits: a
    Spanish chunk collects rank mass from both retrievers, an English chunk can
    only ever collect it from one.

    Taking the dense pool per language rather than globally means the strongest
    English chunks reach the fusion stage instead of being crowded out before
    it. They still have to earn their final position - nothing is reordered,
    only admitted.
    """
    # Bucket every chunk index by the language of the CV it came from.
    groups: dict[str, list[int]] = {}
    for i, lang in enumerate(languages):
        groups.setdefault(lang, []).append(i)

    # Monolingual corpus: nothing to balance, behave like a plain top-N.
    if len(groups) < 2:
        return _ranks(dense_scores, pool)

    # Give each language an equal slice of the pool (2 languages, pool 25 -> 12
    # each) and fill it with that language's own best dense matches.
    share = max(1, pool // len(groups))
    selected: list[int] = []
    for indices in groups.values():
        idx = np.asarray(indices)
        ordered = idx[np.argsort(-dense_scores[idx])][:share]
        selected.extend(int(i) for i in ordered)

    # Merge the per-language slices back into one list ordered by true
    # similarity. The quota decided who got in; it does not decide the order.
    selected.sort(key=lambda i: -dense_scores[i])
    return selected


def embed_query(query: str) -> np.ndarray:
    """Embed and L2-normalise a query vector.

    Exposed separately so callers can start it early: on this hardware the
    embedding model and the chat model are both resident in VRAM, so embedding
    the query can overlap with the routing call instead of queueing behind it.
    """
    vector = np.asarray(client.embed([query])[0], dtype=np.float32)
    return vector / (np.linalg.norm(vector) or 1.0)


# How close to the best dense match in the index an other-language chunk must
# sit before it displaces a fused result.
#
# This is a *fairness* gate, not a relevance filter, and the distinction matters.
# The ratio is relative within a single query, so it answers "is the other
# language competitive here?" - not "is this chunk any good?". Measured on this
# corpus, an off-topic query ("recetas de paella valenciana") yields ratios of
# 0.91-1.00, indistinguishable from a genuinely relevant one (0.94-1.00), simply
# because when everything is irrelevant both languages are equally irrelevant.
#
# Filtering irrelevance is not attempted here. A retriever always returns its
# top-k; refusing to answer belongs to the generation prompt, which is built to
# say "the CVs do not contain this" and is tested for it.
CROSS_LINGUAL_MIN_RATIO = 0.90


def _reserve_cross_lingual_slots(
    ordered: list[tuple[int, float]],
    dense_scores: np.ndarray,
    languages: list[str],
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    """Keep genuinely competitive other-language chunks in the final slate.

    RRF rewards a document for appearing in *both* retrievers' lists, but a
    lexical retriever can never match across a language boundary - a Spanish
    query scores exactly zero BM25 against every English chunk. So an English
    chunk is structurally capped at roughly half the fused score of a Spanish
    one, however relevant it actually is.

    On this corpus that suppressed correct answers outright: for "¿Qué candidato
    se dedica al diseño de producto y accesibilidad?" the single best dense
    match in the whole index was an English Product Designer, beaten out of the
    results by Spanish chunks scoring 0.90 and 0.89 of his similarity.

    So slots are reserved - but only for chunks whose *dense* similarity stands
    up on its own. Nothing is reordered on a blended score; the comparison stays
    within one metric.
    """
    # Which languages actually made the cut?
    top = ordered[:top_k]
    present = {languages[i] for i, _ in top}
    missing = {lang for lang in set(languages) if lang not in present and lang != "??"}

    # Every language already represented, or too few results to spare a slot:
    # leave the fusion's own ordering completely alone.
    if not missing or len(top) < top_k:
        return top

    # The bar: 90% of the best dense similarity anywhere in the index. Compared
    # cosine-to-cosine, never against a BM25 score.
    threshold = float(dense_scores.max()) * CROSS_LINGUAL_MIN_RATIO
    reserve = max(1, top_k // 3)  # at most a third of the slots, so 1 of 5

    # Walk down the fused list past the cut-off, looking for the missing
    # language's best chunks that clear the bar.
    promoted: list[tuple[int, float]] = []
    for idx, score in ordered[top_k:]:
        if len(promoted) >= reserve:
            break
        if languages[idx] in missing and dense_scores[idx] >= threshold:
            promoted.append((idx, score))

    if not promoted:
        return top
    return top[: top_k - len(promoted)] + promoted


def search(
    query: str,
    *,
    top_k: int | None = None,
    candidate_pool: int = 25,
    cv_ids: list[str] | None = None,
    query_vector: np.ndarray | None = None,
) -> list[RetrievedChunk]:
    """Return the best chunks for ``query``, fused across both retrievers.

    ``cv_ids`` restricts the search to specific candidates, which the router
    uses to answer "summarise the profile of X" without the rest of the corpus
    competing for the top slots. ``query_vector`` accepts an embedding computed
    ahead of time (see :func:`embed_query`).
    """
    top_k = top_k or settings.top_k
    index = load_index()

    # STEP 1 - optional filter to specific CVs.
    # Scores are masked to -inf rather than the arrays being sliced, so every
    # index stays aligned with index.chunks and no bookkeeping is needed to map
    # positions back afterwards.
    mask: np.ndarray | None = None
    if cv_ids:
        wanted = set(cv_ids)
        mask = np.array([c.cv_id in wanted for c in index.chunks], dtype=bool)
        if not mask.any():
            mask = None  # nothing matched: fall back to searching everything

    # STEP 2 - dense retrieval (semantic).
    # Both the stored vectors and the query are L2-normalised, so this single
    # matmul IS the cosine similarity of the query against all ~370 chunks.
    # One multiplication, no loop, sub-millisecond.
    query_vec = embed_query(query) if query_vector is None else query_vector
    dense_scores = index.embeddings @ query_vec

    # STEP 3 - lexical retrieval (exact tokens).
    # Catches what embeddings blur: acronyms, surnames, product names.
    bm25_scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float32)

    if mask is not None:
        dense_scores = np.where(mask, dense_scores, -np.inf)
        bm25_scores = np.where(mask, bm25_scores, -np.inf)

    # STEP 4 - turn both score lists into RANKED shortlists.
    # From here on only positions matter, never the raw scores: that is the
    # whole point of RRF, since cosine and BM25 are not on comparable scales.
    languages = [c.metadata.get("language", "??") for c in index.chunks]
    dense_order = _language_balanced_pool(dense_scores, languages, candidate_pool)
    bm25_order = _ranks(bm25_scores, candidate_pool, positive_only=True)

    # chunk index -> its position in each shortlist (0 = best)
    dense_rank = {idx: r for r, idx in enumerate(dense_order)}
    bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}

    # STEP 5 - Reciprocal Rank Fusion.
    # Each shortlist a chunk appears in contributes 1/(k + rank + 1). With
    # k=60 the curve is nearly flat, so *being in both lists* matters far more
    # than the exact position in either: appearing twice at rank 0 scores
    # 2/61 = 0.0328, once at rank 0 only 1/61 = 0.0164.
    k = settings.rrf_k
    fused: dict[int, float] = {}
    for idx, rank in dense_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for idx, rank in bm25_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # STEP 6 - and that doubling is exactly why cross-lingual hits need
    # protecting: BM25 cannot match the other language at all, so those chunks
    # can only ever collect from one list. See the function below.
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    best = _reserve_cross_lingual_slots(
        ordered, dense_scores, languages, top_k=top_k
    )

    # STEP 7 - return the chunks with every score kept, so the UI's Pipeline
    # tab can show exactly why each one was picked.
    return [
        RetrievedChunk(
            chunk=index.chunks[idx],
            rrf_score=score,
            dense_rank=dense_rank.get(idx),
            bm25_rank=bm25_rank.get(idx),
            dense_score=(
                float(dense_scores[idx]) if np.isfinite(dense_scores[idx]) else None
            ),
            bm25_score=float(bm25_scores[idx]) if np.isfinite(bm25_scores[idx]) else None,
        )
        for idx, score in best
    ]
