"""Graphs that extract memories on a schedule."""

import asyncio
import logging
import os
from datetime import datetime
from typing import cast, List
import aiohttp
import requests

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)


class CustomEmbeddings(Embeddings, BaseModel):
    """Custom embeddings implementation for the memory agent using your GTE-Qwen2 endpoint."""
    
    model: str = Field(default="")
    base_url: str = Field(default="")
    api_key: str = Field(default="")
    
    class Config:
        extra = "forbid"

    def __init__(self, context: Context):
        super().__init__(
            model=context.embedding_model,
            base_url=context.embedding_base_url,
            api_key=context.embedding_api_key
        )

    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """Make request to custom embedding endpoint."""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            }
            
            payload = {
                "input": texts[0] if len(texts) == 1 else texts,
                "model": self.model
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "data" in result:
                return [item["embedding"] for item in result["data"]]
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            # Return dummy embeddings to prevent complete failure
            return [[0.0] * 1536 for _ in texts]

    async def _amake_request(self, texts: List[str]) -> List[List[float]]:
        """Make async request to custom embedding endpoint."""
        try:
            headers = {
                "Content-Type": "application/json", 
                "x-api-key": self.api_key
            }
            
            payload = {
                "input": texts[0] if len(texts) == 1 else texts,
                "model": self.model
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if "data" in result:
                        return [item["embedding"] for item in result["data"]]
                    else:
                        raise ValueError(f"Unexpected response format: {result}")
                        
        except Exception as e:
            logger.error(f"Async embedding request failed: {e}")
            # Return dummy embeddings to prevent complete failure
            return [[0.0] * 1536 for _ in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self._make_request(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        result = self._make_request([text])
        return result[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs."""
        return await self._amake_request(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query text."""
        result = await self._amake_request([text])
        return result[0]


def get_custom_llm(context: Context) -> ChatOpenAI:
    """Initialize a custom LLM using the context configuration."""
    return ChatOpenAI(
        model=context.model_name,
        api_key=context.llm_api_key,
        base_url=context.llm_base_url,
        default_headers=context.llm_extra_headers or {},
        temperature=0.7,
        max_tokens=1000,
    )


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    system_prompt = runtime.context.system_prompt
    
    # Initialize the custom LLM
    llm = get_custom_llm(runtime.context)

    try:
        # Retrieve the most recent memories for context
        memories = await cast(BaseStore, runtime.store).asearch(
            ("memories", user_id),
            query=str([m.content for m in state.messages[-3:]]),
            limit=10,
        )
    except Exception as e:
        logger.warning(f"Failed to retrieve memories: {e}")
        memories = []

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    try:
        # Invoke the language model with the prepared prompt and tools
        msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
            [{"role": "system", "content": sys}, *state.messages]
        )
        return {"messages": [msg]}
    except Exception as e:
        logger.error(f"Error calling custom LLM: {e}")
        from langchain_core.messages import AIMessage
        fallback_msg = AIMessage(content=f"I apologize, but I encountered an error while processing your message.")
        return {"messages": [fallback_msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    """Store memories from tool calls."""
    tool_calls = getattr(state.messages[-1], "tool_calls", [])
    
    if not tool_calls:
        return {"messages": []}

    try:
        # Concurrently execute all upsert_memory calls
        saved_memories = await asyncio.gather(
            *(
                tools.upsert_memory(
                    **tc["args"],
                    user_id=runtime.context.user_id,
                    store=cast(BaseStore, runtime.store),
                )
                for tc in tool_calls
            )
        )

        # Format the results of memory storage operations
        results = [
            {
                "role": "tool",
                "content": mem,
                "tool_call_id": tc["id"],
            }
            for tc, mem in zip(tool_calls, saved_memories)
        ]
        return {"messages": results}
        
    except Exception as e:
        logger.error(f"Error storing memories: {e}")
        results = [
            {
                "role": "tool",
                "content": f"Memory storage failed: {str(e)}",
                "tool_call_id": tc["id"],
            }
            for tc in tool_calls
        ]
        return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "store_memory"
    return END


# Create the graph + all nodes
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")

graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]