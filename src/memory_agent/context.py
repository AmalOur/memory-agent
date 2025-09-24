"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    # Custom LLM Configuration
    llm_api_key: str = field(
        default="a",
        metadata={"description": "API key for the custom LLM endpoint"}
    )
    
    llm_base_url: str = field(
        default="https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev",
        metadata={"description": "Base URL for the custom LLM API"}
    )
    
    llm_model_name: str = field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={"description": "Name of the custom LLM model"}
    )
    
    llm_temperature: float = field(
        default=0.7,
        metadata={"description": "Temperature for LLM generation"}
    )
    
    llm_max_tokens: int = field(
        default=1000,
        metadata={"description": "Maximum tokens for LLM generation"}
    )

    # Custom Embeddings Configuration
    embed_api_key: str = field(
        default="Y0DNDULH6SJskvTl",
        metadata={"description": "API key for the custom embedding endpoint"}
    )
    
    embed_base_url: str = field(
        default="https://inference-instance-gte-qwen2-ust2hkbr.ai.gcore.dev/v1/embeddings",
        metadata={"description": "Base URL for the custom embedding API"}
    )
    
    embed_model_name: str = field(
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        metadata={"description": "Name of the custom embedding model"}
    )

    # Legacy model field (kept for compatibility but not used with custom models)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="custom/qwen3-30b",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                # Map field names to environment variable names
                env_var_name = f.name.upper()
                setattr(self, f.name, os.environ.get(env_var_name, f.default))