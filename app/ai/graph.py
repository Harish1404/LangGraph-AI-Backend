"""
The LangGraph chatbot — a StateGraph that replaces the old hand-wired ChatService.

How to read this file (for LangGraph beginners):
─────────────────────────────────────────────────
1. **State**  : ChatState is a TypedDict that every node can read and write.
                The `messages` key uses LangGraph's `add_messages` reducer,
                which means new messages are *appended*, never overwritten.

2. **Nodes**  : Plain async functions.  Each one receives the current state,
                does one job, and returns a dict of the fields it changed.

3. **Edges**  : The wires between nodes.  A normal edge always goes one way;
                a *conditional* edge calls a tiny Python function to pick the
                next node at runtime.

4. **ToolNode**: A built-in LangGraph helper that runs every tool the LLM
                 asked for and appends ToolMessages — no manual loop needed.

5. **Compile**: Turns the graph definition into a runnable object.  The
                `MemorySaver` checkpointer stores conversation history in
                memory, keyed by the `thread_id` you pass at runtime.

Graph shape (see implementation_plan.md for the Mermaid diagram):

    START → route_query
              ├── RAG    → retrieve           → generate → END
              ├── TOOL   → call_llm_with_tools ⇄ tools  → END
              ├── BOTH   → retrieve_for_both  → call_llm_with_tools ⇄ tools → END
              └── DIRECT → generate           → END
"""

import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.ai.chat import _build_models, content_to_text
from app.ai.router import query_router
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_context_text
from app.prompts.router_prompt import (
    BOTH_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
)
from app.rag.rag_pipeline import rag_pipeline
from app.tools.weather import get_weather

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Define the State
# ═══════════════════════════════════════════════════════════════════════════════
# Everything that flows through the graph lives here.
# `add_messages` is a *reducer*: when a node returns {"messages": [new_msg]},
# LangGraph appends it to the existing list instead of replacing it.

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history (auto-appended)
    route: str                               # "RAG" | "TOOL" | "BOTH" | "DIRECT"
    search_query: str                        # rewritten question from the router
    context: str                             # retrieved RAG chunks (empty if unused)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Define the Nodes (each node is a plain async function)
# ═══════════════════════════════════════════════════════════════════════════════

async def route_query(state: ChatState) -> dict:
    """
    Node 1: Classify the user's question and rewrite it.

    The router decides which path (RAG / TOOL / BOTH / DIRECT) the question
    should take, and also rewrites follow-up questions like "and where did he
    study?" into standalone ones the vector store can actually search for.
    """
    # The last message is always the user's latest question
    last_message = state["messages"][-1]
    user_prompt = last_message.content

    # History for the router = everything except the latest message
    history = state["messages"][:-1]

    decision = await query_router.route(user_prompt, history)

    logger.info(
        f"Router selected: {decision.route} for query: {user_prompt!r} "
        f"(searching for: {decision.standalone_question!r})"
    )

    return {
        "route": decision.route,
        "search_query": decision.standalone_question,
    }


async def retrieve(state: ChatState) -> dict:
    """
    Node 2a: Retrieve resume chunks for the RAG path.

    Searches the vector store using the router's rewritten question,
    then stores the results as context text in the state.
    """
    retrieved_chunks = await rag_pipeline.retrieve(state["search_query"])
    context_text = build_context_text(retrieved_chunks)

    return {"context": context_text}


async def retrieve_for_both(state: ChatState) -> dict:
    """
    Node 2b: Retrieve resume chunks for the BOTH path.

    Same retrieval as the RAG node — the only difference is where the
    graph goes *after* this (to the tool node, not to generate).
    """
    retrieved_chunks = await rag_pipeline.retrieve(state["search_query"])
    context_text = build_context_text(retrieved_chunks)

    return {"context": context_text}


async def call_llm_with_tools(state: ChatState) -> dict:
    """
    Node 3: Call the LLM with tools bound.

    Used by the TOOL and BOTH paths.  The LLM can either:
      - answer directly (no tool calls) → the graph ends, or
      - request tool calls → the graph loops through the ToolNode.

    For the BOTH path, the retrieved context is included in the prompt
    so the model can read a city out of the resume before calling the
    weather tool.
    """
    _, llm_with_tools = _build_models(500)

    # Build the message list for this call
    user_prompt = state["messages"][-1].content
    context = state.get("context", "")
    route = state.get("route", "TOOL")

    # Pick the right system prompt
    if route == "BOTH" and context:
        system_prompt = BOTH_SYSTEM_PROMPT
        user_content = f"Resume context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        system_prompt = TOOL_SYSTEM_PROMPT
        user_content = user_prompt

    # Collect conversation history (exclude the last HumanMessage — we rebuild it)
    history = [msg for msg in state["messages"][:-1]
               if not isinstance(msg, SystemMessage)]

    messages = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=user_content),
    ]

    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def generate(state: ChatState) -> dict:
    """
    Node 4: Generate a final answer (no tools).

    Used by the RAG and DIRECT paths.
    - RAG:    answers from retrieved resume chunks
    - DIRECT: answers from the model's own knowledge
    """
    llm_with_fallbacks, _ = _build_models(500)

    user_prompt = state["messages"][-1].content
    route = state.get("route", "DIRECT")
    context = state.get("context", "")

    # Pick system prompt and build the user message
    if route == "RAG" and context:
        system_prompt = RAG_SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        system_prompt = DIRECT_SYSTEM_PROMPT
        user_content = user_prompt

    # Collect conversation history (exclude the last HumanMessage — we rebuild it)
    history = [msg for msg in state["messages"][:-1]
               if not isinstance(msg, SystemMessage)]

    messages = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=user_content),
    ]

    response = await llm_with_fallbacks.ainvoke(messages)

    return {"messages": [response]}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Define Conditional Edge Functions
# ═══════════════════════════════════════════════════════════════════════════════
# These are tiny functions that look at the state and return the name of the
# next node.  LangGraph uses them at the diamond-shaped decision points.

def pick_route(state: ChatState) -> Literal["retrieve", "call_llm_with_tools", "retrieve_for_both", "generate"]:
    """
    After route_query: which node should run next?

    Reads state["route"] (set by the router) and maps it to a node name.
    """
    route = state.get("route", "DIRECT")

    if route == "RAG":
        return "retrieve"
    elif route == "TOOL":
        return "call_llm_with_tools"
    elif route == "BOTH":
        return "retrieve_for_both"
    else:
        return "generate"


def should_continue(state: ChatState) -> Literal["tools", "end"]:
    """
    After call_llm_with_tools: did the LLM ask for a tool, or is it done?

    If the last message has tool_calls → go to the ToolNode.
    Otherwise → the answer is ready, go to END.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build the ToolNode
# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph's built-in ToolNode automatically:
#   1. Reads tool_calls from the last AIMessage
#   2. Executes each tool
#   3. Appends ToolMessage(s) to state["messages"]
# This replaces the entire hand-written _run_tool_loop from the old chat.py.

tools = ToolNode([get_weather])


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Assemble the Graph
# ═══════════════════════════════════════════════════════════════════════════════

graph = StateGraph(ChatState)

# ── Add all nodes ──
graph.add_node("route_query", route_query)
graph.add_node("retrieve", retrieve)
graph.add_node("retrieve_for_both", retrieve_for_both)
graph.add_node("call_llm_with_tools", call_llm_with_tools)
graph.add_node("generate", generate)
graph.add_node("tools", tools)

# ── Add edges ──

# Entry point: every request starts at route_query
graph.add_edge(START, "route_query")

# After routing, pick the right path based on the route
graph.add_conditional_edges(
    "route_query",
    pick_route,
    {
        "retrieve": "retrieve",
        "call_llm_with_tools": "call_llm_with_tools",
        "retrieve_for_both": "retrieve_for_both",
        "generate": "generate",
    },
)

# RAG path: retrieve → generate → END
graph.add_edge("retrieve", "generate")

# BOTH path: retrieve_for_both → call_llm_with_tools
graph.add_edge("retrieve_for_both", "call_llm_with_tools")

# After the LLM call: either loop through tools or finish
graph.add_conditional_edges(
    "call_llm_with_tools",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

# After tools run, go back to the LLM so it can use the results
graph.add_edge("tools", "call_llm_with_tools")

# After generate, we're done
graph.add_edge("generate", END)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Compile with a Checkpointer (this is the memory)
# ═══════════════════════════════════════════════════════════════════════════════
# MemorySaver stores the full conversation state in memory, keyed by thread_id.
# At runtime, you pass: config={"configurable": {"thread_id": conversation_id}}
# and LangGraph automatically loads/saves the conversation history for you.
#
# This replaces the entire app/memory/window.py module.

checkpointer = MemorySaver()

chat_graph = graph.compile(checkpointer=checkpointer)
