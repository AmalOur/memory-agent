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

    # Custom LLM Configuration
    model_name: str = "Qwen/Qwen3-30B-A3B"
    """The model name to use for the custom LLM."""
    
    llm_api_key: str = "a"
    """API key for the custom LLM endpoint."""
    
    llm_base_url: str = "https://inference-instance-qwen3-30b-ust2hkbr.ai.gcore.dev/v1"
    """Base URL for the custom LLM endpoint."""
    
    llm_extra_headers: Optional[dict] = field(default_factory=lambda: {"X-API-Key": "a"})
    """Extra headers to send with requests to the custom LLM."""
    
    # Legacy model field for compatibility (deprecated)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="custom/qwen3-30b",
        metadata={
            "description": "Legacy model field. Use model_name, llm_api_key, and llm_base_url instead."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                # Map environment variable names
                env_name = f.name.upper()
                setattr(self, f.name, os.environ.get(env_name, f.default))
        
        # Handle special case for extra headers
        if self.llm_extra_headers == {"X-API-Key": "a"}:
            # Update X-API-Key with the actual API key
            self.llm_extra_headers = {"X-API-Key": self.llm_api_key}