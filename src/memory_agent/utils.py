"""Utility functions used in our graph."""

import requests
from typing import List, Dict

def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}


def call_custom_llm(api_url: str, api_key: str, model: str, messages: List[Dict]) -> str:
    """Send a chat completion request to a custom LLM endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,  # GCore-style headers
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    resp = requests.post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_custom_embeddings(api_url: str, api_key: str, model: str, text: str) -> List[float]:
    """Send a request to a custom embeddings endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }
    payload = {"model": model, "input": text}
    resp = requests.post(f"{api_url}/embeddings", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


from langgraph.store.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
    """Adapter for custom embeddings."""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    async def aembed(self, text: str):
        # async wrapper around sync utils.call_custom_embeddings
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: call_custom_embeddings(self.api_url, self.api_key, self.model, text),
        )

    def embed(self, text: str):
        # direct sync fallback
        return call_custom_embeddings(self.api_url, self.api_key, self.model, text)
