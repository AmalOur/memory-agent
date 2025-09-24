"""Utility functions used in our graph."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def split_model_and_provider(fully_specified_name: str) -> dict:
    """Split provider/model string into parts."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}


class GCoreChat(ChatOpenAI):
    """OpenAI-compatible chat model but using X-API-Key instead of Bearer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inject GCore headers
        self.client.default_headers["X-API-Key"] = self.openai_api_key
        # Remove Authorization header
        self.client.api_key = None


class GCoreEmbeddings(OpenAIEmbeddings):
    """OpenAI-compatible embeddings model but using X-API-Key instead of Bearer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inject GCore headers
        self.client.default_headers["X-API-Key"] = self.openai_api_key
        # Remove Authorization header
        self.client.api_key = None
