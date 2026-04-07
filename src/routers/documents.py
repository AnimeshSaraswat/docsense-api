import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from src.config import get_settings
from src.schemas.documents import (
    DocumentMeta,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from src.services.chunker import chunk_text
from src.services.embedder import get_embeddings
from src.services.extractor import SUPPORTED_TYPES, extract_text
from src.services.generator import generate_answer
from src.services.vector_store import (
    collection_exists,
    delete_collection,
    query_chunks,
    store_chunks,
)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile):
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: PDF, TXT.",
        )

    settings = get_settings()
    doc_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text(dest, file.content_type)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {e}")

    if not text.strip():
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="Document appears to be empty or unreadable."
        )

    dest.with_suffix(".txt").write_text(text, encoding="utf-8")

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = get_embeddings([c.text for c in chunks])
    store_chunks(doc_id, chunks, embeddings)

    metadata = {
        "doc_id": doc_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (UPLOAD_DIR / f"{doc_id}.json").write_text(json.dumps(metadata), encoding="utf-8")

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        content_type=file.content_type,
        char_count=len(text),
        message=f"Uploaded, chunked into {len(chunks)} chunks, and indexed.",
    )


@router.post("/{doc_id}/query", response_model=QueryResponse)
def query_document(doc_id: str, body: QueryRequest):
    if not collection_exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    query_embedding = get_embeddings([body.question])[0]

    settings = get_settings()
    chunks = query_chunks(doc_id, query_embedding, top_k=settings.max_retrieved_chunks)

    if not chunks:
        raise HTTPException(
            status_code=422, detail="No chunks found for this document."
        )

    answer = generate_answer(body.question, [c["text"] for c in chunks])

    return QueryResponse(
        doc_id=doc_id,
        question=body.question,
        answer=answer,
        sources=[
            SourceChunk(index=c["index"], text=c["text"], score=c["score"])
            for c in chunks
        ],
    )


@router.get("", response_model=list[DocumentMeta])
def list_documents():
    docs = []
    for meta_file in sorted(
        UPLOAD_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True
    ):
        try:
            docs.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return docs


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    if not collection_exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_collection(doc_id)

    for f in UPLOAD_DIR.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)
