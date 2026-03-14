"""
main.py — PDF Ingestion Pipeline
=======================================
Reads PDFs from DATA_DIR/pdfs/
Outputs to:
  DATA_DIR/processed/page_images/<doc_id>/page_XXXX.png
  DATA_DIR/processed/page_text/<doc_id>/page_XXXX.txt
  DATA_DIR/processed/metadata/<doc_id>.json
"""

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from google import genai
from google.genai import types as genai_types

# Config
DATA_DIR       = Path(os.getenv("DATA_DIR", "data"))
PDF_DIR        = DATA_DIR / "pdfs"
OUT_DIR        = DATA_DIR / "processed"
OUT_IMAGES_DIR = OUT_DIR / "page_images"
OUT_TEXT_DIR   = OUT_DIR / "page_text"
OUT_META_DIR   = OUT_DIR / "metadata"
OUT_CSV_DIR    = OUT_DIR / "csv"

MAX_PAGES = int(os.getenv("MAX_PAGES", "2000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

IMAGE_SIZE_THRESHOLD  = int(os.getenv("IMAGE_SIZE_THRESHOLD", "10000"))
IMAGE_COUNT_THRESHOLD = int(os.getenv("IMAGE_COUNT_THRESHOLD", "1"))

# Gemini client
_gemini_client: Optional[genai.Client] = None


def _get_gemini_client() -> Optional[genai.Client]:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            print("[WARN] GEMINI_API_KEY not set — image pages will skip description.")
            return None
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# Helpers
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_id(pdf_path: Path) -> str:
    return hashlib.sha1(str(pdf_path.resolve()).encode()).hexdigest()


def _ensure_dirs():
    for d in [OUT_IMAGES_DIR, OUT_TEXT_DIR, OUT_META_DIR, OUT_CSV_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Image detection
def detect_images_on_page(page: fitz.Page) -> Tuple[bool, int]:
    image_list = page.get_images(full=True)
    if not image_list:
        return False, 0

    significant = 0
    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            w = base_image.get("width", 0)
            h = base_image.get("height", 0)
            if w * h >= IMAGE_SIZE_THRESHOLD:
                significant += 1
        except Exception:
            continue

    return significant >= IMAGE_COUNT_THRESHOLD, significant


def extract_page_image_as_png(page: fitz.Page, zoom: float = 2.0) -> bytes:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


# Gemini image description
def describe_image_with_gemini(image_bytes: bytes, page_text: str = "") -> str:
    client = _get_gemini_client()
    if not client:
        return "[Image description unavailable: GEMINI_API_KEY not set]"

    if page_text.strip():
        prompt = (
            "Describe this medical lecture slide/image in detail. Focus on:\n"
            "- Diagrams, charts, tables, or anatomical illustrations\n"
            "- Key medical concepts shown visually\n"
            "- Any text visible in the image that is part of diagrams or labels\n\n"
            f"Additional text context from this page:\n{page_text[:500]}\n\n"
            "Provide a comprehensive description suitable for medical students."
        )
    else:
        prompt = (
            "Describe this medical lecture slide/image in detail. Focus on:\n"
            "- Diagrams, charts, tables, or anatomical illustrations\n"
            "- Key medical concepts shown visually\n"
            "- Any text visible in the image that is part of diagrams or labels\n\n"
            "Provide a comprehensive description suitable for medical students."
        )

    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            return response.text.strip() if response.text else "[ERROR: Empty description]"

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "resource" in err or "exhausted" in err:
                attempt += 1
                print(f"\n  [RATE LIMIT] Gemini 429 on image description (attempt {attempt}). Waiting 30s...")
                time.sleep(30)
                continue  # retry forever until it succeeds
            # Non-429 error — return error string, don't crash the whole pipeline
            return f"[ERROR: {e}]"


# Page processing
def process_page(page: fitz.Page, page_no: int, doc_id: str) -> Dict[str, Any]:
    img_dir = OUT_IMAGES_DIR / doc_id
    txt_dir = OUT_TEXT_DIR / doc_id
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    has_images, img_count = detect_images_on_page(page)

    try:
        page_text = page.get_text("text") or ""
    except Exception:
        page_text = ""

    page_data: Dict[str, Any] = {
        "page_number":  page_no,
        "has_images":   has_images,
        "image_count":  img_count,
        "text_chars":   len(page_text),
        "content_type": None,
        "image_path":   None,
        "text_path":    None,
        "description":  None,
    }

    if has_images:
        page_data["content_type"] = "image"

        img_bytes = extract_page_image_as_png(page)
        img_path  = img_dir / f"page_{page_no:04d}.png"
        img_path.write_bytes(img_bytes)
        page_data["image_path"] = str(img_path)

        description = describe_image_with_gemini(img_bytes, page_text)
        page_data["description"] = description

        combined = f"[IMAGE DESCRIPTION]\n{description}\n\n[PAGE TEXT]\n{page_text}"
        txt_path = txt_dir / f"page_{page_no:04d}.txt"
        txt_path.write_text(combined, encoding="utf-8")
        page_data["text_path"] = str(txt_path)

    else:
        page_data["content_type"] = "text"
        txt_path = txt_dir / f"page_{page_no:04d}.txt"
        txt_path.write_text(page_text, encoding="utf-8")
        page_data["text_path"] = str(txt_path)

    return page_data


# CSV index
def save_csv_index(doc_id: str, pdf_path: Path, pages: List[Dict[str, Any]]):
    csv_path = OUT_CSV_DIR / f"{doc_id}.csv"
    fieldnames = [
        "doc_id", "source_pdf", "page_number", "content_type",
        "has_images", "image_count", "text_chars", "image_path", "text_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pages:
            writer.writerow({
                "doc_id":       doc_id,
                "source_pdf":   str(pdf_path),
                "page_number":  p["page_number"],
                "content_type": p["content_type"],
                "has_images":   p["has_images"],
                "image_count":  p["image_count"],
                "text_chars":   p["text_chars"],
                "image_path":   p.get("image_path", ""),
                "text_path":    p.get("text_path", ""),
            })


# ─────────────────────────────────────────
# PDF processor
# ─────────────────────────────────────────
def process_pdf(pdf_path: Path) -> Dict[str, Any]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {pdf_path}")

    doc_id    = _doc_id(pdf_path)
    meta_path = OUT_META_DIR / f"{doc_id}.json"

    if meta_path.exists():
        print(f"  [SKIP] Already processed: {pdf_path.name}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"\nProcessing: {pdf_path.name}")
    print(f"  doc_id: {doc_id}")

    meta: Dict[str, Any] = {
        "doc_id":         doc_id,
        "source_pdf":     str(pdf_path),
        "filename":       pdf_path.name,
        "created_at_utc": _utc_now_iso(),
        "pages":          [],
        "warnings":       [],
        "stats":          {"total_pages": 0, "image_pages": 0, "text_pages": 0},
    }

    pdf       = fitz.open(str(pdf_path))
    num_pages = pdf.page_count
    meta["num_pages"]            = num_pages
    meta["stats"]["total_pages"] = num_pages

    if num_pages > MAX_PAGES:
        raise RuntimeError(f"PDF too large: {num_pages} pages (max {MAX_PAGES})")

    for i in range(num_pages):
        page    = pdf.load_page(i)
        page_no = i + 1
        print(f"  Page {page_no}/{num_pages}...", end="\r")

        try:
            page_data = process_page(page, page_no, doc_id)
            meta["pages"].append(page_data)
            if page_data["content_type"] == "image":
                meta["stats"]["image_pages"] += 1
            else:
                meta["stats"]["text_pages"] += 1
        except Exception as e:
            meta["warnings"].append({"page": page_no, "error": str(e)})
            print(f"\n  [WARN] Page {page_no}: {e}")

    pdf.close()

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv_index(doc_id, pdf_path, meta["pages"])

    print(f"\n  Done: {meta['stats']}")
    return meta


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Local PDF ingestion pipeline")
    parser.add_argument("--pdf", type=str, default=None, help="Process a single PDF file")
    args = parser.parse_args()

    _ensure_dirs()

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {PDF_DIR.resolve()}")
            return

    print(f"Found {len(pdfs)} PDF(s) | Output: {OUT_DIR.resolve()}")
    print("=" * 60)

    success, failed = 0, 0
    for pdf_path in pdfs:
        try:
            process_pdf(pdf_path)
            success += 1
        except Exception as e:
            print(f"\n[ERROR] {pdf_path.name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"DONE: {success} succeeded, {failed} failed")
    print("Next step: python vector.py")


if __name__ == "__main__":
    main()