"""Graphs that extract memories on a schedule."""

import logging
from datetime import datetime
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

# ✅ Initialize the chat model using OpenAI-compatible endpoint
llm = init_chat_model()


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    ctx = runtime.context
    user_id = ctx.user_id
    model = ctx.model
    system_prompt = ctx.system_prompt

    # Retrieve memories
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format them into a string
    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})"
        for mem in memories
    )
    if formatted:
        formatted = f"<memories>\n{formatted}\n</memories>"

    sys = system_prompt.format(
        user_info=formatted,
        time=datetime.now().isoformat(),
    )

    # ✅ Call the LLM with OpenAI-compatible API
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        context=utils.split_model_and_provider(model),
    )

    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    tool_calls = getattr(state.messages[-1], "tool_calls", [])
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
    results = [
        {"role": "tool", "content": mem, "tool_call_id": tc["id"]}
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "store_memory"
    return END


builder = StateGraph(State, context_schema=Context)
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
