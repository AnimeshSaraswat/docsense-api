from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    content_type: str
    char_count: int
    message: str


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    content_type: str
    char_count: int
    chunk_count: int
    uploaded_at: str


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    sources: list[SourceChunk]
