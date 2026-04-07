from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.documents import router as documents_router

app = FastAPI(
    title="DocSense API",
    description="RAG-based document Q&A API",
    version="1.0.0",
)

app.include_router(documents_router, prefix="/documents", tags=["documents"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok", "service": "docsense-api", "version": "1.0.0"}
