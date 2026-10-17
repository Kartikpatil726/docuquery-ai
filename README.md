# DocuQuery AI: Enterprise Retrieval-Augmented Generation (RAG) System

Upload PDF documents and ask questions in natural language. Answers are grounded in your documents, with page-level citations and a confidence score. When the documents do not cover a question, the assistant answers from general knowledge and says so.

## 1. Architecture and Data Flow

```
            ┌────────────┐   PDF    ┌──────────────────────────────────────┐
            │  Streamlit │ ───────► │ Ingestion                            │
            │     UI     │          │ load (pypdf) → chunk → embed → store │
            └─────┬──────┘          └───────────────────┬──────────────────┘
                  │ question                            ▼
                  │                              ┌─────────────┐
                  ▼                              │  ChromaDB   │
        ┌───────────────────┐  top-k chunks      │ vector store│
        │   RAG pipeline    │ ◄───────────────── └─────────────┘
        │ embed question    │
        │ retrieve → guard  │ ── prompt + sources ──► Gemini LLM
        │ validate citations│ ◄───── JSON answer ───────┘
        └─────────┬─────────┘
                  ▼
     answer + citations + confidence + latency
```

**Ingestion:** a PDF is read page by page, split into overlapping chunks (default 1000 characters, 200 overlap), embedded, and stored in ChromaDB with filename and page metadata. Re-uploading the same file is detected and skipped.

**Query:** the question is embedded and the top-k most similar chunks are retrieved. Suspicious chunks (prompt-injection patterns) are dropped, and the remaining chunks are sent to Gemini as labelled sources (S1, S2, ...). The model returns structured JSON (`answerable`, `answer`, `sources`).

**Guardrails:**
1. Retrieved text is treated as untrusted data, never as instructions.
2. Citations are validated against the retrieved chunks and must meet `MIN_SIMILARITY`.
3. Citation text, filenames and page numbers come from the stored chunks, never from model output.
4. If the document does not support the answer, the response is marked `answerable: false`, has no citations, and is answered from general knowledge with a clear note.
5. Confidence is derived from the retrieval scores of the cited chunks.

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API | FastAPI, Uvicorn |
| UI | Streamlit |
| PDF parsing | pypdf |
| Chunking | langchain-text-splitters (recursive) |
| Vector database | ChromaDB (embedded) |
| Embeddings | Local `sentence-transformers/all-MiniLM-L6-v2`, or Gemini embeddings |
| LLM | Google Gemini (via `google-genai`), structured JSON output |
| Validation / settings | Pydantic, pydantic-settings |
| Testing | pytest, `scripts/evaluate.py` |
| Packaging | Docker, docker-compose, GitHub Actions CI |

## 3. Project Directory Structure

```
docuquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py            # /health /upload /query /documents
│   │   ├── core/
│   │   │   ├── config.py            # settings from .env
│   │   │   ├── security.py          # X-API-Key check
│   │   │   └── retry.py             # retry/backoff for 429/5xx
│   │   ├── models/schemas.py        # request/response models
│   │   └── services/
│   │       ├── document_loader.py   # PDF → pages
│   │       ├── chunking.py          # pages → chunks
│   │       ├── embeddings.py        # text → vectors
│   │       ├── vector_store.py      # ChromaDB wrapper
│   │       ├── guardrails.py        # anti-hallucination logic
│   │       ├── llm.py               # Gemini client
│   │       ├── ingestion.py         # upload pipeline
│   │       └── rag_pipeline.py      # question answering
│   ├── tests/
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── requirements.txt
│   └── requirements-local.txt       # optional local-embedding libs
├── frontend/
│   ├── app.py                       # Streamlit UI (talks to the API)
│   ├── .streamlit/config.toml
│   ├── Dockerfile
│   └── requirements.txt
├── cloud/
│   ├── streamlit_app.py             # single-app deployment (no separate API)
│   └── requirements.txt
├── scripts/                         # sample PDF, model checks, evaluation
├── data/                            # never committed (raw_docs/, chroma/)
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── requirements.txt                 # developer install
├── docker-compose.yml
├── render.yaml
└── README.md
```

## 4. Local Setup and Installation

```bash
git clone <your-repo-url> docuquery-ai
cd docuquery-ai
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Edit `.env` and set at least:

```
GEMINI_API_KEY=your-key-from-aistudio
GEMINI_MODEL=<a model available to your key>
API_KEY=<any long random string>
EMBEDDING_PROVIDER=local           # or gemini
```

Find a usable model name with `python scripts/probe_models.py`. Never commit `.env`.

Other settings (defaults): `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `TOP_K=5`, `MIN_SIMILARITY=0.25`, `MAX_UPLOAD_MB=25`, `LLM_MAX_OUTPUT_TOKENS`, `CORS_ORIGINS=http://localhost:8501`, `RATE_LIMIT=30/minute`.

## 5. Execution Instructions

Run the API and UI in two terminals, both with the virtual environment active.

```bash
# Terminal 1: backend
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: frontend (from the project root)
streamlit run frontend/app.py --server.port 8501
```

Open http://localhost:8501, upload a PDF in the sidebar, click **Index document**, then ask questions. Generate a sample document with `python scripts/make_sample_pdf.py`.

**Docker:** `docker compose up --build`

**Single-app mode (no API server):** `streamlit run cloud/streamlit_app.py`. This imports the backend code directly and uses Gemini embeddings. Set `GEMINI_API_KEY`, `GEMINI_MODEL` and `EMBEDDING_PROVIDER=gemini` in the environment or in Streamlit secrets.

## 6. Testing and Evaluation

```bash
# Unit tests (chunking and guardrails)
cd backend
pytest -q

# End-to-end evaluation (backend running; run from the project root)
python scripts/evaluate.py
```

`evaluate.py` runs a fixed question set against the sample document and reports pass/fail per case, citation page recall and latency. If you hit free-tier limits, set `EVAL_PAUSE=5` in `.env`. Because out-of-scope questions are now answered from general knowledge, evaluation cases for such questions should expect `answerable == false` with no citations rather than a refusal message.

## 7. API Specification

All endpoints except `/health` require the header `X-API-Key: <API_KEY>`. Interactive docs are served at `/docs` while the server runs.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, indexed chunk count, embedding provider, LLM configured |
| POST | `/upload` | Multipart PDF upload (field `file`); indexes the document |
| POST | `/query` | Ask a question |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Delete a document and its chunks |

**POST /query**

```json
{ "question": "How many days to pay invoices?", "doc_id": null }
```
`question` is 3 to 1000 characters. `doc_id` optionally restricts the search to one document.

```json
{
  "answer": "Invoices are payable within 45 days [S1].",
  "answerable": true,
  "confidence": 0.82,
  "citations": [
    { "source_id": "S1", "filename": "acme_msa.pdf", "page": 2, "quote": "...", "score": 0.71 }
  ],
  "latency_ms": 1430
}
```

**Status codes:** 400 invalid PDF, 401 bad API key, 404 unknown document, 413 file too large, 415 not a PDF, 422 unreadable PDF, 502 model or embedding provider failure, 503 service not ready.

## 8. Deployment Architecture

**Option A: Single Streamlit app (simplest).** Deploy `cloud/streamlit_app.py` on Streamlit Community Cloud. The RAG pipeline runs inside the app, with Gemini for both embeddings and answers. Secrets (`GEMINI_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_PROVIDER`, `LLM_MAX_OUTPUT_TOKENS`) are set in the app settings. The index is stored in temporary storage and is lost when the app restarts.

**Option B: Separate services.** The FastAPI backend (Docker) and the Streamlit frontend are deployed separately (`render.yaml`, `docker-compose.yml`). The frontend needs `BACKEND_URL` and `API_KEY`, and the backend needs `CORS_ORIGINS` set to the frontend's origin. Persist the `data/chroma` directory on a volume, and run a single Uvicorn worker because ChromaDB is an embedded database.

## 9. Security and Operational Hardening

- Secrets live only in `.env` or the platform's secret store, and are never committed.
- API-key authentication on all data endpoints (`X-API-Key`).
- Upload checks: `.pdf` extension, `%PDF-` file signature, size limit (`MAX_UPLOAD_MB`).
- Prompt-injection defence: retrieved text is untrusted, suspicious chunks are dropped, and the system prompt forbids following instructions found in sources.
- Citations are built from stored chunks and validated, so page numbers and quotes cannot be invented by the model.
- Request validation with Pydantic (question length limits, typed responses).
- Configurable CORS origins and rate limit (`RATE_LIMIT`).
- Retry with exponential backoff for transient provider errors (429, 5xx).
- Provider failures are converted to one error type and returned as clean 502/503 responses without leaking internals.
- Public deployments spend your LLM quota. Restrict viewers where possible and monitor usage.

## 10. License

MIT License. Copyright (c) 2026 YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
