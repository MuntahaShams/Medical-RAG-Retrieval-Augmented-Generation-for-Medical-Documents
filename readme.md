# Medical RAG — Setup & Usage Guide
---

## Requirements
- Python 3.10+
- **Gemini API key**

---

## Installation

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your Gemini API key**

On Mac / Linux:
```bash
export GEMINI_API_KEY="api_key"
```

On Windows:
```cmd
set GEMINI_API_KEY=api_key
```

---

## Folder Structure

```
project/
├── main.py
├── vector.py
├── retriever_service.py
├── qa_service.py
├── app.py
├── requirements.txt
├── static/
│   └── index.html
└── data/
    └── chroma_db/
    └── mapping/
    └── pdfs/
    └── processed/

```

---

```bash
uvicorn app:app --reload --port 8000
```

---

## Accessing the UI

Once the app is running, open your browser and go to:

**http://127.0.0.1:8000/ui**

You will see the Medical RAG interface.

## Scripts Overview

| Script | Purpose |
|---|---|
| `main.py` | Reads PDFs, extracts text per page, calls Gemini to describe image/diagram pages, saves results locally |
| `vector.py` | Generates embeddings for all pages using Gemini and indexes them into a local ChromaDB vector database |
| `retriever_service.py` | FastAPI service that handles vector search — converts a query into an embedding and finds the most relevant pages |
| `qa_service.py` | FastAPI service that takes a question, retrieves relevant pages, and calls Gemini to generate a cited answer |
| `app.py` | Main FastAPI entry point — starts the server, serves the UI, and connects all services |

---

## API Endpoints (for developers)

| Method | URL | Description |
|---|---|---|
| `GET` | `/healthz` | Check if the server is running |
| `GET` | `/retriever_healthz` | Check retriever status and index size |
| `GET` | `/qa_healthz` | Check QA service status |
| `POST` | `/qa` | Ask a question programmatically |
| `POST` | `/retrieve` | Run a raw vector search (debug) |

**Example API call:**
```bash
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of pulmonary embolism?", "mode": "lectures", "reasoning_depth": "balanced"}'
```

---

## Troubleshooting

**UI shows "not found"**
Make sure `static/index.html` exists in the project folder.

**Port already in use**
Change the port: `uvicorn app:app --reload --port 8001` and access the UI at `http://127.0.0.1:8001/ui`