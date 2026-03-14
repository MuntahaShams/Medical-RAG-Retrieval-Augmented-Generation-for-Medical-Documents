"""
vector.py — Embedding Generation + ChromaDB Ingestion
======================================================
Reads processed metadata from DATA_DIR/processed/metadata/
Reads page text from DATA_DIR/processed/page_text/
Generates embeddings via Gemini API (gemini-embedding-001)
Stores vectors in local ChromaDB (no Docker needed)
Saves id_map.json locally

No GCS. No GCP auth. Just a GEMINI_API_KEY.

Setup:
    pip install chromadb google-genai

Set env vars:
    GEMINI_API_KEY=your_key_here

Run:
    python vector.py
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import chromadb
from google import genai

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATA_DIR    = Path(os.getenv("DATA_DIR", "data"))
META_DIR    = DATA_DIR / "processed" / "metadata"
MAPPING_DIR = DATA_DIR / "mapping"
ID_MAP_PATH = MAPPING_DIR / "id_map.json"

# ChromaDB stores its data here (persistent, local folder)
CHROMA_PATH       = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_rag")

# Gemini embedding
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "768"))  # 768, 1536, or 3072

# Chunking
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "15000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))
MIN_CHUNK     = int(os.getenv("MIN_CHUNK_SIZE", "100"))
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"

# How many chunks to upsert at once into ChromaDB
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))


# ─────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────
def _get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")
    return genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────
def get_embedding(client: genai.Client, text: str) -> List[float]:
    """Call Gemini embedding API. Retries on rate limit."""
    if len(text) > 30000:
        text = text[:30000]

    for attempt in range(4):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config={"output_dimensionality": EMBED_DIMENSION},
            )
            return resp.embeddings[0].values
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "rate" in err or "429" in err:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("Embedding failed after retries.")


# ─────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────
def chunk_text(text: str) -> List[Tuple[str, int, int]]:
    """Split text into overlapping chunks. Returns [(chunk, start, end)]."""
    if len(text) <= CHUNK_SIZE:
        return [(text, 0, len(text))]

    chunks: List[Tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            search_start = end - int(CHUNK_SIZE * 0.2)
            snippet = text[search_start:end]
            last_break = max(
                snippet.rfind(". "),
                snippet.rfind("! "),
                snippet.rfind("? "),
                snippet.rfind("\n"),
            )
            if last_break != -1:
                end = search_start + last_break + 1

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append((chunk, start, end))

        prev_end = chunks[-1][2] if chunks else 0
        start = end - CHUNK_OVERLAP
        if start <= prev_end:
            start = end

    return chunks if chunks else [(text, 0, len(text))]


# ─────────────────────────────────────────
# Per-page processing
# ─────────────────────────────────────────
def process_page(
    client: genai.Client,
    page_data: Dict[str, Any],
    doc_id: str,
    source_pdf: str,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield chunk dicts for one page:
    {datapoint_id, embedding, metadata, text}
    """
    page_number = page_data["page_number"]
    text_path   = page_data.get("text_path", "")

    if not text_path or not Path(text_path).exists():
        return

    raw  = Path(text_path).read_text(encoding="utf-8")
    text = " ".join(raw.split())  # normalize whitespace

    if not text.strip():
        return

    if ENABLE_CHUNKING and len(text) > CHUNK_SIZE:
        chunks = chunk_text(text)
        total  = len(chunks)
        for idx, (chunk_str, start_char, end_char) in enumerate(chunks):
            emb = get_embedding(client, chunk_str)
            dp_id = f"{doc_id}_p{page_number:04d}_c{idx:02d}"
            yield {
                "datapoint_id": dp_id,
                "embedding":    emb,
                "text":         chunk_str,
                "metadata": {
                    "datapoint_id":    dp_id,
                    "doc_id":          doc_id,
                    "page_number":     page_number,
                    "chunk_index":     idx,
                    "total_chunks":    total,
                    "chunk_start_char": start_char,
                    "chunk_end_char":  end_char,
                    "content_type":    page_data.get("content_type", "text"),
                    "has_images":      page_data.get("has_images", False),
                    "image_path":      page_data.get("image_path") or "",
                    "text_path":       text_path,
                    "text_chars":      len(chunk_str),
                    "source_pdf":      source_pdf,
                },
            }
    else:
        emb   = get_embedding(client, text)
        dp_id = f"{doc_id}_p{page_number:04d}"
        yield {
            "datapoint_id": dp_id,
            "embedding":    emb,
            "text":         text,
            "metadata": {
                "datapoint_id":    dp_id,
                "doc_id":          doc_id,
                "page_number":     page_number,
                "chunk_index":     0,
                "total_chunks":    1,
                "chunk_start_char": 0,
                "chunk_end_char":  len(text),
                "content_type":    page_data.get("content_type", "text"),
                "has_images":      page_data.get("has_images", False),
                "image_path":      page_data.get("image_path") or "",
                "text_path":       text_path,
                "text_chars":      len(text),
                "source_pdf":      source_pdf,
            },
        }


# ─────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────
def get_chroma_collection(persist_path: str, collection_name: str):
    """Return (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        # ChromaDB stores our own embeddings — disable its built-in embedding fn
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_batch(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
):
    """Upsert a batch into ChromaDB."""
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("VECTOR EMBEDDING GENERATION (Gemini → ChromaDB)")
    print("=" * 60)
    print(f"Data dir:    {DATA_DIR.resolve()}")
    print(f"Embed model: {EMBED_MODEL} ({EMBED_DIMENSION}d)")
    print(f"ChromaDB:    {CHROMA_PATH}  collection={CHROMA_COLLECTION}")
    print(f"Chunking:    {'ON' if ENABLE_CHUNKING else 'OFF'}")
    print("=" * 60)

    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var.")

    gemini     = _get_gemini_client()
    collection = get_chroma_collection(CHROMA_PATH, CHROMA_COLLECTION)

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing id_map (resume support)
    if ID_MAP_PATH.exists():
        id_map: Dict[str, Any] = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
        print(f"Loaded existing id_map: {len(id_map)} entries (skipping already indexed)")
    else:
        id_map = {}

    meta_files = sorted(META_DIR.glob("*.json"))
    if not meta_files:
        raise RuntimeError(f"No metadata JSON in {META_DIR}. Run main.py first.")

    print(f"\nFound {len(meta_files)} document(s)")

    total_new  = 0
    total_skip = 0

    # Batch buffers
    buf_ids:   List[str]             = []
    buf_embs:  List[List[float]]     = []
    buf_docs:  List[str]             = []
    buf_metas: List[Dict[str, Any]]  = []

    def flush():
        nonlocal total_new
        if buf_ids:
            upsert_batch(collection, buf_ids, buf_embs, buf_docs, buf_metas)
            total_new += len(buf_ids)
            buf_ids.clear(); buf_embs.clear()
            buf_docs.clear(); buf_metas.clear()
            # Save id_map after each flush
            ID_MAP_PATH.write_text(
                json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    for meta_idx, meta_file in enumerate(meta_files):
        meta       = json.loads(meta_file.read_text(encoding="utf-8"))
        doc_id     = meta["doc_id"]
        source_pdf = meta.get("source_pdf", meta.get("filename", ""))
        pages      = meta.get("pages", [])

        print(f"\n[{meta_idx + 1}/{len(meta_files)}] {meta.get('filename', doc_id)}")
        print(f"  pages: {len(pages)}")

        for page_data in pages:
            try:
                for chunk in process_page(gemini, page_data, doc_id, source_pdf):
                    dp_id = chunk["datapoint_id"]

                    if dp_id in id_map:
                        total_skip += 1
                        continue

                    # ChromaDB metadata values must be str/int/float/bool
                    safe_meta = {
                        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                        for k, v in chunk["metadata"].items()
                    }

                    buf_ids.append(dp_id)
                    buf_embs.append(chunk["embedding"])
                    buf_docs.append(chunk["text"])
                    buf_metas.append(safe_meta)

                    id_map[dp_id] = chunk["metadata"]

                    if len(buf_ids) >= BATCH_SIZE:
                        flush()
                        print(f"  ✓ {total_new} chunks indexed so far...")

            except Exception as e:
                print(f"  [WARN] Page {page_data.get('page_number')}: {e}")
                continue

    flush()  # Final flush

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"New chunks indexed : {total_new}")
    print(f"Chunks skipped     : {total_skip}")
    print(f"Total in id_map    : {len(id_map)}")
    print(f"ChromaDB path      : {CHROMA_PATH}")
    print(f"id_map saved to    : {ID_MAP_PATH}")
    print("=" * 60)
    print("Next step: uvicorn app:app --reload")


if __name__ == "__main__":
    main()