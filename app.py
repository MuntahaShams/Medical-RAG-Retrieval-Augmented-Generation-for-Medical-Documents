from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import retriever_service as retriever_service
import qa_service as qa_service

app = FastAPI(title="Medical RAG", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],  # includes Authorization
)

# -----------------------------
# Serve UI (static)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Mount /static for assets (optional, but good practice)
# Your index.html will still be served at "/"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def ui_index():
    """
    Serves the UI.
    Put index.html in ./static/index.html
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Helpful error if you forgot to add the file
        return {"error": "UI not found. Place static/index.html next to the service."}
    return FileResponse(str(index_path))

# (Optional) convenience: /ui also points to same index
@app.get("/ui", include_in_schema=False)
def ui_alias():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"error": "UI not found. Place static/index.html next to the service."}
    return FileResponse(str(index_path))

@app.on_event("startup")
def startup_all():
    # Initialize clients/caches for each module
    retriever_service.init_clients()
    qa_service.init_clients()

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "medical-rag"}

app.include_router(retriever_service.router)
app.include_router(qa_service.router)