from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(doc_id: str) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(doc_id: str, chunks: list, embeddings: list[list[float]]) -> None:
    collection = get_or_create_collection(doc_id)
    collection.add(
        ids=[f"{doc_id}_chunk_{c.index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"index": c.index, "char_start": c.char_start, "char_end": c.char_end}
            for c in chunks
        ],
    )


def delete_collection(doc_id: str) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(f"doc_{doc_id}")
    except Exception:
        pass  # Already gone — fine


def collection_exists(doc_id: str) -> bool:
    client = get_chroma_client()
    try:
        client.get_collection(f"doc_{doc_id}")
        return True
    except Exception:
        return False


def query_chunks(doc_id: str, query_embedding: list[float], top_k: int) -> list[dict]:
    collection = get_or_create_collection(doc_id)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append(
            {
                "text": doc,
                "index": results["metadatas"][0][i]["index"],
                "score": round(1 - results["distances"][0][i], 4),
            }
        )

    return sorted(chunks, key=lambda x: x["index"])
