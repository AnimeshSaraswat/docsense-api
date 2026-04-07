# DocSense API

A production-grade RAG (Retrieval-Augmented Generation) API for document Q&A.  
Upload PDFs or text files, ask questions in natural language, get AI-powered answers with source attribution.

**Live API:** https://docsense-api-hhcq.onrender.com  
**Docs:** https://docsense-api-hhcq.onrender.com/docs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI (Python 3.11) |
| Vector Store | ChromaDB (in-process, persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` |
| Validation | Pydantic v2 |
| Deployment | Render (API) · Vercel (Frontend) |

---

## Architecture
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   Extract   │────▶│    Chunk    │────▶│    Embed    │
│  PDF / TXT  │     │    Text     │     │ 500c / 50ov │     │   OpenAI    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
│                                                                  │
▼                                                                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   GPT-4o    │◀────│  Top-K      │◀────│  ChromaDB   │
│ + Sources   │     │    mini     │     │  Retrieval  │     │ Vector Store│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

**Upload flow:**
1. File saved to disk
2. Text extracted via `pypdf` (PDF) or direct read (TXT)
3. Text split into overlapping chunks (500 chars, 50 char overlap)
4. Chunks embedded via OpenAI `text-embedding-3-small`
5. Embeddings stored in ChromaDB with metadata

**Query flow:**
1. Question embedded using same model
2. Cosine similarity search against document's ChromaDB collection
3. Top-5 chunks retrieved and passed as context
4. GPT-4o-mini generates answer grounded in retrieved context
5. Answer returned with source chunks and relevance scores

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload PDF or TXT — extract, chunk, embed, index |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/documents/{id}/query` | Ask a question, get answer + source chunks |
| `DELETE` | `/documents/{id}` | Remove document and its vector index |
| `GET` | `/health` | Health check |

---

## Local Setup
```bash
git clone https://github.com/AnimeshSaraswat/docsense-api.git
cd docsense-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn src.main:app --reload
```

Swagger UI: http://localhost:8000/docs

---

## Example Usage

**Upload:**
```bash
curl -X POST https://docsense-api-hhcq.onrender.com/documents/upload \
  -F "file=@report.pdf"
```

**Query:**
```bash
curl -X POST https://docsense-api-hhcq.onrender.com/documents/{doc_id}/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

**Response shape:**
```json
{
  "doc_id": "...",
  "question": "What are the key findings?",
  "answer": "The key findings include...",
  "sources": [
    { "index": 2, "text": "...", "score": 0.87 },
    { "index": 3, "text": "...", "score": 0.81 }
  ]
}
```

---

## Limitations

- ChromaDB runs in-process — data resets on Render free tier restarts
- Render free tier spins down after inactivity — first request may take ~30s
- Not designed for concurrent heavy load on free tier