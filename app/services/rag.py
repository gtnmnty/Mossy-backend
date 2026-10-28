# Feature 6: RAG ("Ask Mossy about my documents")
# Retrieval step: turns a question into the few document passages most likely to answer it.
# Flow: load the index that scripts/ingest.py built -> search it two ways (meaning and
# keywords) -> merge both rankings into one list -> return the top passages as plain dicts.
# The search_documents tool (tools/documents.py) calls retrieve() and hands the result to Claude.

import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import voyageai
from rank_bm25 import BM25Okapi

from app.core.config import settings

ROOT = Path(__file__).resolve().parents[2]        # the backend folder
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"

_vo = voyageai.Client(api_key=settings.voyage_api_key)
tokenize = lambda s: re.findall(r"\w+", s.lower())


# Loads the ingested chunks and their embeddings from disk, once, and keeps them in memory
# (lru_cache(maxsize=1) means this body only runs on the first call). Returns None when no
# ingestion has run yet, so retrieve() can answer "nothing indexed" instead of crashing.
@lru_cache(maxsize=1)
def _index():
    cp, ep = DATA_DIR / "chunks.json", DATA_DIR / "embeddings.npy"
    if not cp.exists() or not ep.exists():
        return None
    chunks = json.loads(cp.read_text(encoding="utf-8"))
    return {
        "chunks": chunks,
        "vecs": np.load(ep),
        "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
    }


def _rrf(rankings, k: int = 60) -> list[int]:
    """Reciprocal rank fusion: merges ranked lists without tuning any scores."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0) + 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


# Hybrid search: finds the best passages for a question two different ways, then combines
# them, because meaning-based and keyword-based search each catch things the other misses.
# Flow: embed the question -> rank all chunks by meaning (vector similarity) -> rank all
# chunks by keyword overlap (BM25) -> merge the two rankings with RRF -> return the top chunks.
async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    idx = _index()
    if idx is None:
        return []
    res = await asyncio.to_thread(                 # the Voyage call is blocking, so keep it off the event loop
        _vo.embed, [query], model=settings.embed_model, input_type="query")
    qvec = np.array(res.embeddings[0], dtype=np.float32)

    sims = idx["vecs"] @ qvec                      # Voyage vectors are normalized: dot product = cosine
    vec_rank = np.argsort(-sims)[:20].tolist()
    bm = idx["bm25"].get_scores(tokenize(query))
    bm_rank = [i for i in np.argsort(-bm)[:20].tolist() if bm[i] > 0]

    return [idx["chunks"][i] for i in _rrf([vec_rank, bm_rank])[:top_k]]