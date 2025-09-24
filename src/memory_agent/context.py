"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields
from typing import Optional

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    # Custom LLM Configuration - loaded from environment variables
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B"))
    """The model name to use for the custom LLM."""
    
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    """API key for the custom LLM endpoint."""
    
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1"))
    """Base URL for the custom LLM endpoint."""
    
    llm_extra_headers: Optional[dict] = field(default=None)
    """Extra headers to send with requests to the custom LLM."""
    
    # Custom Embeddings Configuration - loaded from environment variables
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"))
    """The embedding model name to use."""
    
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    """API key for the custom embedding endpoint."""
    
    embedding_base_url: str = field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "https://inference-instance-gte-qwen2-ust2hkbr.ai.gcore.dev/v1/embeddings"))
    """Base URL for the custom embedding endpoint."""
    
    # Legacy model field for compatibility (deprecated)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="custom/qwen3-30b",
        metadata={
            "description": "Legacy model field. Use model_name, llm_api_key, and llm_base_url instead."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Post-initialization setup and environment variable handling."""
        
        # Set up extra headers with API key from environment
        if self.llm_extra_headers is None:
            x_api_key = os.getenv("X_API_KEY") or self.llm_api_key
            if x_api_key:
                self.llm_extra_headers = {"X-API-Key": x_api_key}
            else:
                self.llm_extra_headers = {}
        
        # CRITICAL: Set dummy OPENAI_API_KEY if not present (required for LangGraph platform)
        # This prevents the OpenAI initialization error without affecting our custom endpoints
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-langgraph-platform-compatibility"