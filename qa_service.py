"""
qa_service.py — Medical RAG QA Service (Local)
===============================================
Uses Gemini API key for:
  - Query embedding (gemini-embedding-001)
  - Answer generation (gemini-2.0-flash)
Uses local ChromaDB for vector search.
Uses local id_map.json + local text files.

No GCS. No GCP auth. Just GEMINI_API_KEY.
"""

import json
import os
import random
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
GEN_MODEL         = os.getenv("GEN_MODEL", "gemini-3-pro-preview")

CHROMA_PATH       = str(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_rag")

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS_PER_SOURCE", "10000"))

GEN_TEMPERATURE    = float(os.getenv("GEN_TEMPERATURE", "0.1"))
GEN_MAX_TOKENS     = int(os.getenv("GEN_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))

REASONING_DEPTH_MAP = {
    "fast":     {"top_k": 10, "expand_neighbors": 0, "max_sources": 5},
    "balanced": {"top_k": 20, "expand_neighbors": 1, "max_sources": 10},
    "deep":     {"top_k": 50, "expand_neighbors": 2, "max_sources": 20},
}

APPROVED_WEB_DOMAINS = [
    "uptodate.com", "msdmanuals.com", "who.int", "nejm.org",
    "mdcalc.com", "radiopaedia.org", "litfl.com", "amboss.com",
    "flexikon.doccheck.com", "medix.ch", "smartermedicine.ch",
]

LANG_CACHE_TTL = int(os.getenv("LANG_CACHE_TTL_SECONDS", "3600"))

# ─────────────────────────────────────────
# Router
# ─────────────────────────────────────────
router = APIRouter(tags=["qa"])


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class QAMode(str, Enum):
    LECTURES = "lectures"
    WEB      = "web"


class ReasoningDepth(str, Enum):
    FAST     = "fast"
    BALANCED = "balanced"
    DEEP     = "deep"


class QARequest(BaseModel):
    question:         str            = Field(..., min_length=1)
    mode:             QAMode         = QAMode.LECTURES
    reasoning_depth:  ReasoningDepth = ReasoningDepth.BALANCED
    top_k:            Optional[int]  = Field(None, ge=1, le=200)
    expand_neighbors: Optional[int]  = Field(None, ge=0, le=5)
    max_sources:      Optional[int]  = Field(None, ge=1, le=50)


class Citation(BaseModel):
    pdf_name:    str
    page_number: int
    source_pdf:  Optional[str] = None
    image_path:  Optional[str] = None
    text_path:   Optional[str] = None
    doc_id:      Optional[str] = None
    url:         Optional[str] = None
    kind:        Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class QASource(BaseModel):
    id:           str
    doc_id:       str
    page_number:  int
    image_path:   str = ""
    text_path:    str = ""
    source_pdf:   Optional[str]   = None
    text_excerpt: Optional[str]   = None
    score:        Optional[float] = None
    content_type: Optional[str]   = None
    has_images:   Optional[bool]  = None
    chunk_index:  Optional[int]   = None
    total_chunks: Optional[int]   = None


class WebSource(BaseModel):
    url:   str
    quote: str


class QAResponse(BaseModel):
    language_hint:    Optional[str]       = None
    answer:           str
    citations:        List[Citation]
    sources:          List[QASource]      = []
    used_sources:     List[QASource]      = []
    web_sources:      List[WebSource]     = []
    retrieval_config: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────
# Globals
# ─────────────────────────────────────────
_gemini_client: Optional[genai.Client] = None
_chroma_collection                      = None

_id_map_cache: Dict[str, Any]  = {}
_id_map_cache_ts: float         = 0.0
_lang_cache: Dict[str, Tuple[str, float]] = {}


def init_clients():
    global _gemini_client, _chroma_collection
    if _gemini_client and _chroma_collection:
        return

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_collection = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────
# id_map
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
def chroma_search(vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    if not _chroma_collection:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized.")

    results = _chroma_collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    hits: List[Dict[str, Any]] = []
    for dp_id, dist, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        score = 1 - dist  # convert cosine distance → similarity
        hits.append({"id": dp_id, "score": score, **(meta or {})})

    return hits


# ─────────────────────────────────────────
# Source building
# ─────────────────────────────────────────
def build_sources(
    hits: List[Dict[str, Any]],
    id_map: Dict[str, Any],
    max_sources: int,
) -> List[QASource]:
    out: List[QASource] = []
    seen: set = set()

    for h in hits:
        dp_id  = h["id"]
        # Metadata is already in h (from ChromaDB payload)
        # Fall back to id_map if needed
        meta   = h if h.get("doc_id") else id_map.get(dp_id, {})
        if not meta:
            continue

        doc_id  = str(meta.get("doc_id") or dp_id)
        page_no = int(meta.get("page_number", 0))
        key     = f"{doc_id}_p{page_no}"

        if key in seen:
            continue
        seen.add(key)

        out.append(QASource(
            id=dp_id,
            doc_id=doc_id,
            page_number=page_no,
            image_path=meta.get("image_path") or "",
            text_path=meta.get("text_path") or "",
            source_pdf=meta.get("source_pdf"),
            score=h.get("score"),
            content_type=meta.get("content_type"),
            has_images=bool(meta.get("has_images", False)),
            chunk_index=meta.get("chunk_index"),
            total_chunks=meta.get("total_chunks"),
        ))

        if len(out) >= max_sources:
            break

    return out


# ─────────────────────────────────────────
# Load text excerpt (local file)
# ─────────────────────────────────────────
def load_text_excerpt(source: QASource) -> str:
    if not source.text_path:
        return ""
    p = Path(source.text_path)
    if not p.exists():
        return ""
    try:
        raw = p.read_text(encoding="utf-8").strip()
        return raw[:MAX_TEXT_CHARS] + "..." if len(raw) > MAX_TEXT_CHARS else raw
    except Exception:
        return ""


# ─────────────────────────────────────────
# Neighbor expansion
# ─────────────────────────────────────────
def _build_page_index(id_map: Dict[str, Any]) -> Dict[Tuple[str, int], List[str]]:
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
    return index


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
# Language detection (no API call)
# ─────────────────────────────────────────
def detect_language(question: str) -> str:
    q = question.strip().lower()
    if not q:
        return "en"

    now = time.time()
    if q in _lang_cache and (now - _lang_cache[q][1]) < LANG_CACHE_TTL:
        return _lang_cache[q][0]

    lower = f" {q} "
    fr = sum(1 for w in [" le ", " la ", " les ", " des ", " une ", " un ", " et ", " ou ", " pas ", " pour "] if w in lower)
    de = sum(1 for w in [" der ", " die ", " das ", " und ", " nicht ", " ein ", " eine ", " für ", " mit "] if w in lower)
    lang = "fr" if fr > de and fr >= 2 else "de" if de > fr and de >= 2 else "en"
    _lang_cache[q] = (lang, now)
    return lang


# ─────────────────────────────────────────
# Citation helpers
# ─────────────────────────────────────────
def _pdf_filename(source_pdf: Optional[str], fallback: str = "") -> str:
    if not source_pdf:
        return fallback
    return Path(source_pdf).name or fallback


def _citation_tag(s: QASource) -> str:
    return f"[{_pdf_filename(s.source_pdf, fallback=str(s.doc_id))}, p.{s.page_number}]"


def build_sources_block(sources: List[QASource]) -> str:
    blocks: List[str] = []
    for i, s in enumerate(sources, start=1):
        note = " (contains image description)" if s.content_type == "image" else ""
        blocks.append(
            f"[Source {i}]{note}\n"
            f"CITATION_TAG: {_citation_tag(s)}\n"
            f"EXCERPT:\n{s.text_excerpt or ''}\n"
        )
    return "INDEXED_SOURCES:\n" + "\n".join(blocks)


# ─────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────
def get_system_prompt(lang: str, mode: QAMode) -> str:
    allowed = ", ".join(APPROVED_WEB_DOMAINS)

    table = {
        "fr": (
            "| Aspect | Condition A | Condition B |\n"
            "|--------|-------------|-------------|\n"
            "| Épidémiologie | ... | ... |\n"
            "| Étiologie | ... | ... |\n"
            "| Symptômes | ... | ... |\n"
            "| Diagnostics | **Gold standard** | **Gold standard** |\n"
            "| Thérapie | ... | ... |\n"
        ),
        "de": (
            "| Aspekt | Zustand A | Zustand B |\n"
            "|--------|-----------|----------|\n"
            "| Epidemiologie | ... | ... |\n"
            "| Ätiologie | ... | ... |\n"
            "| Symptome | ... | ... |\n"
            "| Diagnostik | **Goldstandard** | **Goldstandard** |\n"
            "| Therapie | ... | ... |\n"
        ),
    }.get(lang, (
        "| Aspect | Condition A | Condition B |\n"
        "|--------|-------------|-------------|\n"
        "| Epidemiology | ... | ... |\n"
        "| Etiology | ... | ... |\n"
        "| Symptoms | ... | ... |\n"
        "| Diagnostics | **Gold standard** | **Gold standard** |\n"
        "| Therapy | ... | ... |\n"
    ))

    base = (
        f"You are a Senior Medical Tutor. Output language: {lang.upper()}.\n\n"
        "FORMAT:\n"
        "- Write the answer body first with NO inline citations.\n"
        "- Bold key findings and gold standards.\n"
        "- Use comparison tables when comparing conditions:\n"
        f"{table}\n"
        "- End with a SOURCES: section.\n"
        "- Lecture source line: [Source N] \"<short quote>\" <CITATION_TAG>\n"
        "- Web source line:     [Web] \"<short quote>\" <URL>\n\n"
        "RULES:\n"
        "- Only cite evidence you actually used.\n"
        "- Do NOT invent facts not in the evidence.\n"
        f"- Web sources MUST be from: {allowed}\n"
    )

    if mode == QAMode.LECTURES:
        return base + "\nUse ONLY INDEXED_SOURCES. Do NOT use web.\n"
    return base + "\nUse web search. Cite at least 1 allowed-domain source.\n"


# ─────────────────────────────────────────
# Gemini generation
# ─────────────────────────────────────────
def call_gemini(
    question: str,
    lang: str,
    mode: QAMode,
    sources: List[QASource],
) -> str:
    if not _gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")

    system_prompt = get_system_prompt(lang, mode)

    if sources:
        user_content = f"QUESTION: {question}\n\n{build_sources_block(sources)}"
    else:
        user_content = f"QUESTION: {question}\n\nINDEXED_SOURCES: (none)"

    use_web = mode == QAMode.WEB

    def _is_rate_limit(e: Exception) -> bool:
        err = str(e).lower()
        return "429" in err or "quota" in err or "resource" in err or "exhausted" in err

    attempt = 0
    while True:
        try:
            if use_web:
                response = _gemini_client.models.generate_content(
                    model=GEN_MODEL,
                    contents=user_content,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=GEN_TEMPERATURE,
                        max_output_tokens=GEN_MAX_TOKENS,
                        tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
                    ),
                )
            else:
                response = _gemini_client.models.generate_content(
                    model=GEN_MODEL,
                    contents=user_content,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=GEN_TEMPERATURE,
                        max_output_tokens=GEN_MAX_TOKENS,
                    ),
                )

            if response.text:
                return response.text.strip()

            raise ValueError("Empty response from Gemini")

        except HTTPException:
            raise  # don't swallow our own HTTP exceptions

        except Exception as e:
            if _is_rate_limit(e):
                attempt += 1
                print(f"[RATE LIMIT] Gemini 429 on generation (attempt {attempt}). Waiting 30s...")
                time.sleep(30)
                continue  # retry forever
            # Non-429 error — fail immediately
            raise HTTPException(status_code=502, detail=f"Gemini generation failed: {e}")


# ─────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────
def _split_sources(answer: str) -> Tuple[str, str]:
    m = re.search(r"(^|\n)\s*SOURCES\s*:\s*", answer, flags=re.IGNORECASE)
    if not m:
        return answer.strip(), ""
    return answer[: m.start()].strip(), answer[m.start() :].strip()


def _strip_inline_cites(body: str) -> str:
    body = re.sub(r"\[\s*Source\s+\d+[^\]]*\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\[[^\[\]]+?\.pdf\s*,\s*p\.\s*\d+\s*\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"https?://\S+", "", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _extract_used_indices(sources_section: str, max_idx: int) -> List[int]:
    used: List[int] = []
    for m in re.finditer(r"\[\s*Source\s+(\d+)\s*\]", sources_section or "", flags=re.IGNORECASE):
        try:
            n = int(m.group(1))
            if 1 <= n <= max_idx and n not in used:
                used.append(n)
        except Exception:
            pass
    return used


def _extract_web_sources(sources_section: str) -> List[WebSource]:
    out: List[WebSource] = []
    for m in re.finditer(r'\[\s*Web\s*\]\s*"([^"]+)"\s*(https?://\S+)', sources_section, flags=re.IGNORECASE):
        url   = m.group(2).strip().rstrip(')"\'.') 
        quote = m.group(1).strip()
        try:
            host = urlparse(url).netloc.lower().lstrip("www.")
            if any(host == d or host.endswith("." + d) for d in APPROVED_WEB_DOMAINS):
                out.append(WebSource(url=url, quote=quote))
        except Exception:
            pass
    return out


def postprocess(
    answer: str,
    mode: QAMode,
    sources: List[QASource],
) -> Tuple[str, List[int], List[WebSource]]:
    body, sources_section = _split_sources(answer)
    body = _strip_inline_cites(body)

    if not sources_section and sources:
        # Fallback: build a basic sources section
        lines = ["SOURCES:"]
        for i, s in enumerate(sources[:3], start=1):
            q = (s.text_excerpt or "").replace("\n", " ")[:200]
            lines.append(f"[Source {i}] \"{q}\" {_citation_tag(s)}")
        sources_section = "\n".join(lines)

    used_indices = _extract_used_indices(sources_section, max_idx=len(sources))
    web_sources  = _extract_web_sources(sources_section)

    final = f"{body}\n\n{sources_section.strip()}".strip()
    return final, used_indices, web_sources


def build_citations(
    used_indices: List[int],
    sources: List[QASource],
    web_sources: List[WebSource],
) -> List[Citation]:
    out: List[Citation] = []
    for i in used_indices:
        if 1 <= i <= len(sources):
            s = sources[i - 1]
            out.append(Citation(
                pdf_name=_pdf_filename(s.source_pdf, fallback=str(s.doc_id)),
                page_number=s.page_number,
                source_pdf=s.source_pdf,
                image_path=s.image_path,
                text_path=s.text_path,
                doc_id=s.doc_id,
                kind="lecture",
                chunk_index=s.chunk_index,
                total_chunks=s.total_chunks,
            ))
    for ws in web_sources:
        out.append(Citation(pdf_name=ws.url, page_number=0, url=ws.url, kind="web"))
    return out


# ─────────────────────────────────────────
# Retrieval config resolution
# ─────────────────────────────────────────
def resolve_config(req: QARequest) -> Dict[str, Any]:
    cfg = REASONING_DEPTH_MAP[req.reasoning_depth].copy()
    if req.top_k is not None:            cfg["top_k"] = req.top_k
    if req.expand_neighbors is not None: cfg["expand_neighbors"] = req.expand_neighbors
    if req.max_sources is not None:      cfg["max_sources"] = req.max_sources
    cfg["reasoning_depth"] = req.reasoning_depth
    return cfg


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.on_event("startup")
def on_startup():
    init_clients()


@router.get("/qa_healthz")
def qa_healthz():
    return {
        "ok":               True,
        "embed_model":      EMBED_MODEL,
        "gen_model":        GEN_MODEL,
        "chroma_path":      CHROMA_PATH,
        "chroma_collection": CHROMA_COLLECTION,
    }


@router.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest) -> QAResponse:
    init_clients()

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    cfg              = resolve_config(req)
    top_k            = cfg["top_k"]
    expand_n         = cfg["expand_neighbors"]
    max_sources      = cfg["max_sources"]
    lang             = detect_language(question)
    retrieved: List[QASource] = []

    if req.mode == QAMode.LECTURES:
        try:
            id_map = load_id_map()
            vec    = get_embedding(question)
            hits   = chroma_search(vec, top_k)
            hits   = expand_neighbors(hits, expand_n, id_map)
            retrieved = build_sources(hits, id_map, max_sources)
            for s in retrieved:
                s.text_excerpt = load_text_excerpt(s)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    sources_for_model = retrieved if req.mode == QAMode.LECTURES else []

    if req.mode == QAMode.LECTURES and not sources_for_model:
        return QAResponse(
            language_hint=lang,
            answer="Insufficient evidence in the provided documents.",
            citations=[], sources=[], used_sources=[], web_sources=[],
            retrieval_config=cfg,
        )

    raw_answer = call_gemini(question, lang, req.mode, sources_for_model)

    final_answer, used_indices, web_sources = postprocess(
        raw_answer, req.mode, sources_for_model
    )

    used_sources = [sources_for_model[i - 1] for i in used_indices if 1 <= i <= len(sources_for_model)]
    citations    = build_citations(used_indices, sources_for_model, web_sources)

    return QAResponse(
        language_hint=lang,
        answer=final_answer,
        citations=citations,
        sources=sources_for_model,
        used_sources=used_sources,
        web_sources=web_sources,
        retrieval_config=cfg,
    )