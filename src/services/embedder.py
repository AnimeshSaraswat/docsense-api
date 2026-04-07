from openai import OpenAI

from src.config import get_settings


def get_embeddings(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    # Response preserves input order
    return [item.embedding for item in response.data]
