"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

# LLM will be initialized per-request using runtime context
llm = None


def get_custom_llm(runtime: Runtime[Context]) -> ChatOpenAI:
    """Initialize a custom LLM using the runtime context."""
    context = runtime.context
    
    # Create ChatOpenAI with custom endpoint configuration
    custom_llm = ChatOpenAI(
        model=context.model_name,
        api_key=context.llm_api_key,
        base_url=context.llm_base_url,
        default_headers=context.llm_extra_headers or {},
        temperature=0.7,  # You can make this configurable too
        max_tokens=1000,  # You can make this configurable too
    )
    
    return custom_llm


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    system_prompt = runtime.context.system_prompt
    
    # Initialize the custom LLM
    llm = get_custom_llm(runtime)

    # Retrieve the most recent memories for context
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

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
    # This helps the model understand the context and temporal relevance
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    try:
        msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
            [{"role": "system", "content": sys}, *state.messages]
        )
        return {"messages": [msg]}
    except Exception as e:
        logger.error(f"Error calling custom LLM: {e}")
        # Return a fallback message
        from langchain_core.messages import AIMessage
        fallback_msg = AIMessage(content=f"I apologize, but I encountered an error while processing your message: {str(e)}")
        return {"messages": [fallback_msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    # Extract tool calls from the last message
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

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
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        # If there are tool calls, we need to store memories
        return "store_memory"
    # Otherwise, finish; user can send the next message
    return END


# Create the graph + all nodes
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]