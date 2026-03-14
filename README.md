
# Medical RAG – Retrieval Augmented Generation for Medical Documents

A Retrieval-Augmented Generation (RAG) system designed to answer questions from medical documents using semantic search and LLM-based reasoning.

The system processes medical PDFs, builds a vector database, retrieves relevant context for user queries, and generates grounded responses using an LLM.

This project demonstrates how modern AI systems combine **vector search + LLMs + APIs** to build domain-specific knowledge assistants.

---

# Project Demo

🎥 Demo Video:

A short demo of the Medical RAG system processing medical PDFs and answering user queries using retrieval-augmented generation.

medical_rag.mp4
---

# Key Features

* Medical document question answering
* Retrieval-Augmented Generation pipeline
* Semantic search using vector embeddings
* PDF ingestion and processing
* Image/diagram description using Gemini
* FastAPI backend
* Web-based UI for interaction
* ChromaDB vector storage

---

# System Architecture

The system follows a typical RAG pipeline:

1. **Document Ingestion**

   * Medical PDFs are loaded and processed
   * Text is extracted page by page

2. **Image Understanding**

   * Pages containing diagrams/images are processed with Gemini
   * Image descriptions are added to the document text

3. **Embedding Generation**

   * Text chunks are converted into embeddings

4. **Vector Database**

   * Embeddings are stored in **ChromaDB**

5. **Query Processing**

   * User question is converted to embedding
   * Relevant chunks are retrieved from vector database

6. **LLM Response Generation**

   * Retrieved context is sent to the LLM
   * The model generates a grounded answer

---

# Example Workflow

User Question

```
What are the symptoms of hypertension?
```

Pipeline Steps

```
User Query
    ↓
Embedding Generation
    ↓
Vector Search (ChromaDB)
    ↓
Retrieve Relevant Medical Context
    ↓
LLM Answer Generation (Gemini)
    ↓
Final Response
```

---

# Project Structure

```
medical-rag/
│
├── main.py
├── vector.py
├── retriever_service.py
├── qa_service.py
├── app.py
├── requirements.txt
│
├── static/
│   └── index.html        # Web interface
│
├── data/
│   ├── pdfs/             # Source medical documents
│   ├── processed/        # Processed text files
│   ├── mapping/
│   └── chroma_db/        # Vector database
```

---

# Tech Stack

Python
FastAPI
ChromaDB (Vector Database)
Gemini API (LLM + Vision)
Sentence Embeddings
HTML / JavaScript UI

---

# Installation

Clone the repository

```bash
git clone https://github.com/MuntahaShams/medical-rag.git
cd medical-rag
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Setup

Set your Gemini API key.

Mac / Linux

```bash
export GEMINI_API_KEY="your_api_key"
```

Windows

```cmd
set GEMINI_API_KEY=your_api_key
```

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload --port 8000
```

---

# Access the UI

Open your browser and go to:

```
http://127.0.0.1:8000/ui
```

You will see the Medical RAG interface where you can ask questions about the medical documents.

---

# Scripts Overview

| Script                 | Purpose                                                 |
| ---------------------- | ------------------------------------------------------- |
| `main.py`              | Processes PDFs and extracts text and image descriptions |
| `vector.py`            | Generates embeddings and builds the vector database     |
| `retriever_service.py` | Retrieves relevant chunks from ChromaDB                 |
| `qa_service.py`        | Sends context + query to Gemini for answer generation   |
| `app.py`               | FastAPI application and API endpoints                   |

---

# Challenges

Some challenges in building domain-specific RAG systems include:

* handling long medical documents
* maintaining context relevance
* processing images and diagrams
* reducing hallucinations
* ensuring accurate retrieval

This project addresses these using semantic retrieval and structured document processing.

---

# Future Improvements

* citation support in answers
* better chunking strategies
* evaluation metrics for RAG quality
* UI improvements
* support for multiple medical datasets
* hybrid search (BM25 + vector)

---

# Author

**Muntaha Shams**

AI Engineer – LLMs | NLP | Computer Vision | Document AI

GitHub
[https://github.com/MuntahaShams](https://github.com/MuntahaShams)

Portfolio
[https://muntahashams.github.io/portfolio/projects](https://muntahashams.github.io/portfolio/projects)

---
