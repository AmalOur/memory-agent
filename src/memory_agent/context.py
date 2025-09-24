"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="custom/Qwen/Qwen3-30B-A3B",
        metadata={"description": "Provider/model-name for LLM."},
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    # Custom inference configs
    llm_api: str = field(default_factory=lambda: os.environ.get("LLM_API", ""))
    llm_key: str = field(default_factory=lambda: os.environ.get("LLM_KEY", ""))
    embed_api: str = field(default_factory=lambda: os.environ.get("EMBED_API", ""))
    embed_key: str = field(default_factory=lambda: os.environ.get("EMBED_KEY", ""))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", ""))

    def __post_init__(self):
        for f in fields(self):
            if not f.init:
                continue
            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))

    def get_embedder(self):
        """Return the embedder object configured in context."""
        if self.embed_model.startswith("Alibaba-NLP") or self.embed_api:
            return CustomEmbeddings(
                api_url=self.embed_api,
                api_key=self.embed_key,
                model=self.embed_model,
            )
        return None 