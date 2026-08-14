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
    groups: dict[str, list[int]] = {}
    for i, lang in enumerate(languages):
        groups.setdefault(lang, []).append(i)

    if len(groups) < 2:
        return _ranks(dense_scores, pool)

    share = max(1, pool // len(groups))
    selected: list[int] = []
    for indices in groups.values():
        idx = np.asarray(indices)
        ordered = idx[np.argsort(-dense_scores[idx])][:share]
        selected.extend(int(i) for i in ordered)

    # Re-sort the union so ranks still reflect true dense similarity.
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


# A cross-lingual hit must be this close to the best dense match in the corpus
# before it displaces a fused result. Chosen from measurement, not taste: the
# English chunks being suppressed on this corpus scored 0.93-1.00 of the top
# dense similarity, while genuinely off-topic content sits near 0.47/0.53 ~ 0.89
# once normalised. 0.90 admits the former and excludes the latter.
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
    top = ordered[:top_k]
    present = {languages[i] for i, _ in top}
    missing = {lang for lang in set(languages) if lang not in present and lang != "??"}
    if not missing or len(top) < top_k:
        return top

    threshold = float(dense_scores.max()) * CROSS_LINGUAL_MIN_RATIO
    reserve = max(1, top_k // 3)

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

    mask: np.ndarray | None = None
    if cv_ids:
        wanted = set(cv_ids)
        mask = np.array([c.cv_id in wanted for c in index.chunks], dtype=bool)
        if not mask.any():
            mask = None

    # -- dense
    query_vec = embed_query(query) if query_vector is None else query_vector
    dense_scores = index.embeddings @ query_vec

    # -- lexical
    bm25_scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float32)

    if mask is not None:
        dense_scores = np.where(mask, dense_scores, -np.inf)
        bm25_scores = np.where(mask, bm25_scores, -np.inf)

    languages = [c.metadata.get("language", "??") for c in index.chunks]
    dense_order = _language_balanced_pool(dense_scores, languages, candidate_pool)
    bm25_order = _ranks(bm25_scores, candidate_pool, positive_only=True)

    dense_rank = {idx: r for r, idx in enumerate(dense_order)}
    bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}

    k = settings.rrf_k
    fused: dict[int, float] = {}
    for idx, rank in dense_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for idx, rank in bm25_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    best = _reserve_cross_lingual_slots(
        ordered, dense_scores, languages, top_k=top_k
    )
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
