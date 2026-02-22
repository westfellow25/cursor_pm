from __future__ import annotations

import os

from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    if not texts:
        return []

    client = get_openai_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
