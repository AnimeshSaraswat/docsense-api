from openai import OpenAI
from src.config import get_settings


def generate_answer(question: str, chunks: list[str]) -> str:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant answering questions about a document.
Use the context below as your primary source. You may make reasonable inferences 
from the context even if the exact terminology differs (e.g. "qualifications" and 
"education" refer to the same thing). If a question truly cannot be answered from 
the context, say so briefly.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
