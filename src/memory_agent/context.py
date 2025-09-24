from memory_agent.utils import GCoreChat, GCoreEmbeddings

@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gcore/Qwen/Qwen3-30B-A3B",
        metadata={"description": "The model to use (gcore/... for GCore inference)."},
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    # GCore configs (from .env)
    llm_api: str = field(default_factory=lambda: os.environ.get("LLM_API", ""))
    llm_key: str = field(default_factory=lambda: os.environ.get("LLM_KEY", ""))
    embed_api: str = field(default_factory=lambda: os.environ.get("EMBED_API", ""))
    embed_key: str = field(default_factory=lambda: os.environ.get("EMBED_KEY", ""))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", ""))

    def get_chat(self):
        """Return a GCore chat model instance."""
        return GCoreChat(
            model=self.model.split("/", 1)[-1],
            openai_api_key=self.llm_key,
            openai_api_base=self.llm_api,
        )

    def get_embedder(self):
        """Return a GCore embeddings model instance."""
        return GCoreEmbeddings(
            model=self.embed_model,
            openai_api_key=self.embed_key,
            openai_api_base=self.embed_api,
        )
