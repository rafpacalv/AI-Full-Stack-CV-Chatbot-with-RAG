"""Build the search index: dense vectors, a BM25 model and the candidate table.

    python -m cvscreener.ingest.index [--force]

Deliberately *not* a vector database. The corpus is ~28 CVs, a few hundred
chunks; exact cosine similarity is a single ``numpy`` matmul over a
(300, 1024) array, which is sub-millisecond and returns the true nearest
neighbours rather than an approximation. Adding Chroma or FAISS here would mean
another dependency, another process and an ANN index, to make an already
instantaneous exact search slower and approximate.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from ..config import settings
from ..llm import client
from ..textutils import fold_accents
from .chunk import Chunk, chunk_document
from .enrich import enrich_document
from .extract import extract_all

log = logging.getLogger("index")

EMBED_BATCH = 16
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./_-]*")


def tokenize(text: str) -> list[str]:
    """Tokenizer for BM25, shared by indexing and querying.

    Accents are folded so a query for "formacion" matches "formación", and
    tokens keep ``+ # . /`` so "c++", "node.js" and "ci/cd" survive intact.
    """
    return [t.strip("._-/") for t in TOKEN_RE.findall(fold_accents(text).lower()) if len(t) > 1]


def build(force: bool = False) -> dict:
    settings.ensure_dirs()
    started = time.time()

    print("1/4  Extracting text from PDFs ...")
    docs = extract_all()
    if not docs:
        sys.exit(f"No CVs found in {settings.cvs_dir}. Run the generation pipeline first.")
    print(f"     {len(docs)} PDFs, {sum(d.n_chars for d in docs):,} characters")

    print("2/4  Extracting structured facts (LLM) ...")
    t0 = time.time()
    facts = []
    for i, doc in enumerate(docs, 1):
        row = enrich_document(
            cv_id=doc.cv_id, source_file=doc.source_file, text=doc.text, force=force
        )
        facts.append(row)
        print(
            f"     [{i:>2}/{len(docs)}] {row.cv_id} {row.full_name:<24.24} "
            f"{row.seniority:<10} {row.years_experience:>2}y  "
            f"{len(row.skills_normalised):>2} skills",
            flush=True,
        )
    enrich_secs = time.time() - t0

    print("3/4  Chunking ...")
    by_id = {f.cv_id: f for f in facts}
    chunks: list[Chunk] = []
    for doc in docs:
        row = by_id[doc.cv_id]
        chunks.extend(
            chunk_document(
                cv_id=doc.cv_id,
                source_file=doc.source_file,
                text=doc.text,
                candidate=row.full_name,
                metadata={
                    "language": row.cv_language,
                    "role": row.current_role,
                    "seniority": row.seniority,
                    "city": row.city,
                },
            )
        )
    print(f"     {len(chunks)} chunks (avg {len(chunks) / len(docs):.1f} per CV)")

    print("4/4  Embedding + BM25 ...")
    t0 = time.time()

    # Note `embedding_text`, not `text`: each chunk is prefixed with
    # "<Candidate> - <Section>" so the vector carries who it is about.
    texts = [c.embedding_text for c in chunks]

    # Batched because one HTTP round-trip per chunk would dominate the runtime.
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(client.embed(texts[i : i + EMBED_BATCH]))
        print(f"     embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", flush=True)
    embed_secs = time.time() - t0

    matrix = np.asarray(vectors, dtype=np.float32)
    # L2-normalise once, here, so cosine similarity at query time is a plain dot
    # product. Normalising every row on every query would repeat this work for
    # no reason. `.clip` guards against a division by zero on an empty vector.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-9)

    # The lexical half of the index, over the same texts and the same tokenizer
    # the query side uses - they must agree or nothing will ever match.
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    # --- persist ---------------------------------------------------------
    # Four artefacts: the vectors, the chunk texts and metadata, the BM25 model,
    # and the candidate table that the aggregate path queries with pandas.
    np.save(settings.index_dir / "embeddings.npy", matrix)
    with (settings.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    with (settings.index_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump(bm25, fh)

    frame = pd.DataFrame([f.model_dump() for f in facts])
    frame.to_parquet(settings.index_dir / "candidates.parquet", index=False)

    stats = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cvs": len(docs),
        "n_chunks": len(chunks),
        "n_pages": sum(d.n_pages for d in docs),
        "n_chars": sum(d.n_chars for d in docs),
        "embedding_model": settings.embed_model,
        "embedding_dim": int(matrix.shape[1]),
        "extraction_model": settings.chat_model,
        "languages": frame["cv_language"].value_counts().to_dict(),
        "enrich_seconds": round(enrich_secs, 1),
        "embed_seconds": round(embed_secs, 1),
        "total_seconds": round(time.time() - started, 1),
    }
    (settings.index_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nIndex ready in {stats['total_seconds']}s -> {settings.index_dir}\n"
        f"  {stats['n_cvs']} CVs · {stats['n_chunks']} chunks · "
        f"{matrix.shape[0]}x{matrix.shape[1]} embeddings · "
        f"languages {stats['languages']}"
    )
    return stats


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the CV search index")
    ap.add_argument("--force", action="store_true", help="re-run LLM fact extraction")
    build(force=ap.parse_args().force)


if __name__ == "__main__":
    main()
