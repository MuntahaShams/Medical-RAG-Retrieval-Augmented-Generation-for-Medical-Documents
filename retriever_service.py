"""
retriever_service.py — ChromaDB-based Retriever
================================================
Uses Gemini API key for query embedding.
Uses local ChromaDB for vector search.
Uses local id_map.json for metadata lookup.

No GCS. No GCP auth. Just GEMINI_API_KEY.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from fastapi import APIRouter, HTTPException
from google import genai
from pydantic import BaseModel, Field

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATA_DIR    = Path(os.getenv("DATA_DIR", "data"))
ID_MAP_PATH = Path(os.getenv("ID_MAP_PATH", str(DATA_DIR / "mapping" / "id_map.json")))
ID_MAP_TTL  = int(os.getenv("ID_MAP_TTL_SECONDS", "300"))

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSION   = int(os.getenv("EMBED_DIMENSION", "768"))

CHROMA_PATH       = str(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_rag")

# ─────────────────────────────────────────
# Router
# ─────────────────────────────────────────
router = APIRouter(tags=["retriever"])


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=200)
    expand_neighbors: int = Field(0, ge=0, le=5)
    dedupe_by_page: bool = Field(True)


class PageHit(BaseModel):
    id: str
    score: float
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    image_path: Optional[str] = None
    text_path: Optional[str] = None
    source_pdf: Optional[str] = None
    content_type: Optional[str] = None
    has_images: Optional[bool] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    text_chars: Optional[int] = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    expand_neighbors: int
    dedupe_by_page: bool
    results: List[PageHit]


# ─────────────────────────────────────────
# Globals
# ─────────────────────────────────────────
_gemini_client: Optional[genai.Client]   = None
_chroma_collection                        = None

_id_map_cache: Dict[str, Any] = {}
_id_map_cache_ts: float        = 0.0
_page_index_cache: Dict[Tuple[str, int], List[str]] = {}
_page_index_cache_ts: float    = 0.0


def init_clients():
    global _gemini_client, _chroma_collection
    if _gemini_client and _chroma_collection:
        return

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    chroma_client  = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────
# id_map loading
# ─────────────────────────────────────────
def load_id_map() -> Dict[str, Any]:
    global _id_map_cache, _id_map_cache_ts

    now = time.time()
    if _id_map_cache and (now - _id_map_cache_ts) < ID_MAP_TTL:
        return _id_map_cache

    if not ID_MAP_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"id_map.json not found at {ID_MAP_PATH}. Run vector.py first.",
        )

    _id_map_cache    = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
    _id_map_cache_ts = now
    return _id_map_cache


def _build_page_index(id_map: Dict[str, Any]) -> Dict[Tuple[str, int], List[str]]:
    global _page_index_cache, _page_index_cache_ts

    now = time.time()
    if _page_index_cache and (now - _page_index_cache_ts) < ID_MAP_TTL:
        return _page_index_cache

    index: Dict[Tuple[str, int], List[str]] = {}
    for dp_id, meta in id_map.items():
        doc_id  = meta.get("doc_id")
        page_no = meta.get("page_number")
        if not doc_id or page_no is None:
            continue
        key = (str(doc_id), int(page_no))
        index.setdefault(key, []).append(dp_id)

    for key in index:
        index[key].sort()

    _page_index_cache    = index
    _page_index_cache_ts = now
    return index


# ─────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────
def get_embedding(text: str) -> List[float]:
    if not _gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")

    resp = _gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={"output_dimensionality": EMBED_DIMENSION},
    )
    return resp.embeddings[0].values


# ─────────────────────────────────────────
# ChromaDB search
# ─────────────────────────────────────────
def search(vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    if not _chroma_collection:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized.")

    results = _chroma_collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    hits: List[Dict[str, Any]] = []
    ids       = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for dp_id, dist, meta in zip(ids, distances, metadatas):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score (1 = best)
        score = 1 - dist
        hits.append({"id": dp_id, "score": score, **(meta or {})})

    return hits


# ─────────────────────────────────────────
# Deduplication + neighbor expansion
# ─────────────────────────────────────────
def dedupe_by_page(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for h in hits:
        doc_id  = h.get("doc_id")
        page_no = h.get("page_number")
        if not doc_id or page_no is None:
            out.append(h)
            continue
        key = (str(doc_id), int(page_no))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def expand_neighbors(
    hits: List[Dict[str, Any]],
    expand_n: int,
    id_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if expand_n <= 0:
        return hits

    page_index = _build_page_index(id_map)
    expanded: Dict[str, Dict[str, Any]] = {h["id"]: h for h in hits}

    for h in hits:
        doc_id  = h.get("doc_id")
        page_no = h.get("page_number")
        if not doc_id or page_no is None:
            continue

        for delta in range(-expand_n, expand_n + 1):
            if delta == 0:
                continue
            p2 = int(page_no) + delta
            if p2 <= 0:
                continue
            for dp_id in page_index.get((str(doc_id), p2), []):
                if dp_id in expanded:
                    continue
                meta = id_map.get(dp_id)
                if not meta:
                    continue
                expanded[dp_id] = {"id": dp_id, "score": h.get("score", 0.0), **meta}

    # Original order first, neighbors appended after
    ordered: List[Dict[str, Any]] = []
    seen: set = set()
    for h in hits:
        if h["id"] not in seen:
            ordered.append(h)
            seen.add(h["id"])
    for k, v in expanded.items():
        if k not in seen:
            ordered.append(v)
            seen.add(k)
    return ordered


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.get("/retriever_healthz")
def retriever_healthz():
    init_clients()
    count = _chroma_collection.count() if _chroma_collection else "N/A"
    return {
        "ok":                 True,
        "embed_model":        EMBED_MODEL,
        "embed_dimension":    EMBED_DIMENSION,
        "chroma_path":        CHROMA_PATH,
        "chroma_collection":  CHROMA_COLLECTION,
        "vectors_in_db":      count,
        "id_map_entries":     len(load_id_map()),
    }


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    init_clients()

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    id_map = load_id_map()
    vec    = get_embedding(q)
    hits   = search(vec, req.top_k)

    if not hits:
        return RetrieveResponse(
            query=q, top_k=req.top_k,
            expand_neighbors=req.expand_neighbors,
            dedupe_by_page=req.dedupe_by_page,
            results=[],
        )

    if req.dedupe_by_page:
        hits = dedupe_by_page(hits)

    hits = expand_neighbors(hits, req.expand_neighbors, id_map)

    return RetrieveResponse(
        query=q,
        top_k=req.top_k,
        expand_neighbors=req.expand_neighbors,
        dedupe_by_page=req.dedupe_by_page,
        results=[PageHit(**h) for h in hits],
    )