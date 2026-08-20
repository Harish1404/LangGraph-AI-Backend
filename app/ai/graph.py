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
                checkpointer stores conversation state in MongoDB, keyed by the
                `thread_id` you pass at runtime.  Because that needs a live
                database connection, the compile happens in the FastAPI
                lifespan (init_chat_graph) rather than at import time.

6. **Interrupt**: `approve_tools` can pause the whole graph mid-run and wait for
                a human.  The pause is just a checkpoint that was never
                committed, which is why it needs a durable checkpointer to
                survive the round-trip to the browser.

Graph shape (see implementation_plan.md for the Mermaid diagram):

    START → route_query
              ├── RAG    → retrieve           → generate → END
              ├── TOOL   → call_llm_with_tools → approve_tools ⇄ tools → END
              ├── BOTH   → retrieve_for_both  → call_llm_with_tools → …
              └── DIRECT → generate           → END

    approve_tools ── approved ──→ tools ──→ back to call_llm_with_tools
                  └─ rejected ──→ generate (forced DIRECT) ──→ END
"""

from functools import lru_cache
import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.ai.chat import _build_models, content_to_text
from app.ai.router import query_router
from app.core.config import settings
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_context_text
from app.prompts.router_prompt import (
    BOTH_SYSTEM_PROMPT,
    DENIED_TOOL_NOTE,
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
    user_prompt: str                         # this turn's question, verbatim
    denied_tools: list[str]                  # tools the user refused this turn

# A note on the last two fields.
#
# `user_prompt` exists because messages[-1] is NOT reliably the user's question.
# After a trip through the tool node the last message is a ToolMessage, and on
# the rejection path it is a *denial* ToolMessage.  Nodes that need the question
# read this instead of guessing from the tail of the list.
#
# Every field except `messages` is per-turn and must be cleared by route_query.
# With a durable checkpointer the whole state is reloaded on the next turn, so a
# stale `context` or `denied_tools` would otherwise leak into an unrelated
# question. The old MemorySaver hid this by forgetting everything on restart.


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
        "user_prompt": user_prompt,
        # Cleared, not left alone: these are reloaded from the checkpoint and
        # belong to the *previous* turn. See the note under ChatState.
        "context": "",
        "denied_tools": [],
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

    # Build the message list for this call.
    # NOT messages[-1]: on the second pass round the tool loop that is a
    # ToolMessage, and using it here would ask the model to answer its own
    # tool output instead of the question.
    user_prompt = state.get("user_prompt") or state["messages"][-1].content
    context = state.get("context", "")
    route = state.get("route", "TOOL")

    # Pick the right system prompt
    if route == "BOTH" and context:
        system_prompt = BOTH_SYSTEM_PROMPT
        user_content = f"Resume context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        system_prompt = TOOL_SYSTEM_PROMPT
        user_content = user_prompt

    history = [msg for msg in state["messages"]
               if not isinstance(msg, SystemMessage)]

    if isinstance(state["messages"][-1], ToolMessage):
        # Second pass, coming back from the tool node. History already ends with
        # the AIMessage that requested the tools followed by the ToolMessages
        # answering them, and that pairing has to survive intact — Gemini and
        # Groq both reject a request where an assistant tool_call has no
        # matching tool result. So replay it verbatim; do not rebuild anything.
        messages = [SystemMessage(content=system_prompt), *history]
    else:
        # First pass. Drop the raw question and rebuild it with the RAG context
        # folded in, which is the whole point of the BOTH route.
        messages = [
            SystemMessage(content=system_prompt),
            *history[:-1],
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

    # Same reason as in call_llm_with_tools: on the tool-rejection path the last
    # message is a denial ToolMessage, not the question.
    user_prompt = state.get("user_prompt") or state["messages"][-1].content
    route = state.get("route", "DIRECT")
    context = state.get("context", "")
    denied_tools = state.get("denied_tools") or []

    # Pick system prompt and build the user message
    if route == "RAG" and context:
        system_prompt = RAG_SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        system_prompt = DIRECT_SYSTEM_PROMPT
        user_content = user_prompt

    # Arriving here after a refused tool call. Without this the model sees a
    # ToolMessage saying "declined" and a system prompt that knows nothing about
    # tools, and tends to either invent the answer or ask to try again.
    if denied_tools:
        system_prompt += DENIED_TOOL_NOTE.format(tools=", ".join(denied_tools))

    history = [msg for msg in state["messages"]
               if not isinstance(msg, SystemMessage)]

    if isinstance(state["messages"][-1], ToolMessage):
        # Rejection path. The AIMessage/ToolMessage pairing has to stay intact,
        # so append the question rather than replacing the tail.
        messages = [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=user_content),
        ]
    else:
        messages = [
            SystemMessage(content=system_prompt),
            *history[:-1],
            HumanMessage(content=user_content),
        ]

    response = await llm_with_fallbacks.ainvoke(messages)

    return {"messages": [response]}


async def approve_tools(state: ChatState) -> Command[Literal["tools", "generate"]]:
    """
    Node 3b: The human-in-the-loop gate. Runs before any tool does.

    Only tools listed in HITL_TOOLS are gated; everything else falls straight
    through, so the common case costs one set lookup.

    `interrupt()` does not block. It raises, LangGraph checkpoints the graph
    exactly where it stands, and the whole run unwinds. When the caller resumes
    with Command(resume=...), **this node re-executes from its first line** and
    the interrupt() call returns the resume value instead of raising. That is
    why nothing above it may have a side effect: it all runs twice.

    Returns a Command rather than a dict so it can pick its own next node.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    gated = [tc for tc in tool_calls if tc["name"] in settings.HITL_TOOLS]

    if not gated:
        return Command(goto="tools")

    logger.info(f"Tool approval required for: {[tc['name'] for tc in gated]}")

    decision = interrupt({
        "type": "tool_approval",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in gated
        ],
    })

    # Anything that is not an explicit "accept" is a refusal. Fail closed: a
    # malformed resume payload must not be able to run a gated tool.
    action = decision.get("action") if isinstance(decision, dict) else None

    if action == "accept":
        logger.info(f"Tool approval GRANTED for: {[tc['name'] for tc in gated]}")
        return Command(goto="tools")

    # ── Refused ──────────────────────────────────────────────────────────────
    reason = decision.get("reason") if isinstance(decision, dict) else None
    denial = "The user declined to run this tool, so it produced no result."
    if reason:
        denial += f" Their reason: {reason}"

    logger.info(f"Tool approval DENIED for: {[tc['name'] for tc in gated]}")

    # One ToolMessage per *pending* call, not just the gated ones. An AIMessage
    # whose tool_calls are not all answered is an invalid history: the next
    # provider call fails with a 400 rather than degrading. So every id gets
    # closed out, whether it was the one the user objected to or not.
    return Command(
        goto="generate",
        update={
            "messages": [
                ToolMessage(
                    content=denial,
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ],
            # Answer from the model's own knowledge instead. The tool is not
            # retried — the user already said no once.
            "route": "DIRECT",
            "denied_tools": [tc["name"] for tc in gated],
        },
    )


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
graph.add_node("approve_tools", approve_tools)
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

# After the LLM call: either head for the tools or finish.
# Note the target: tool calls go to the approval gate first, never straight to
# the ToolNode. approve_tools is what forwards them on (or refuses).
graph.add_conditional_edges(
    "call_llm_with_tools",
    should_continue,
    {
        "tools": "approve_tools",
        "end": END,
    },
)

# approve_tools has no static edges — it returns Command(goto=...), and LangGraph
# reads its possible destinations off the Command[Literal["tools", "generate"]]
# return annotation.

# After tools run, go back to the LLM so it can use the results
graph.add_edge("tools", "call_llm_with_tools")

# After generate, we're done
graph.add_edge("generate", END)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Compile with a Checkpointer (this is the memory)
# ═══════════════════════════════════════════════════════════════════════════════
# The checkpointer stores the full graph state in MongoDB, keyed by thread_id.
# At runtime you pass config={"configurable": {"thread_id": conversation_id}}
# and LangGraph loads and saves that thread's history for you.
#
# This replaces the entire app/memory/window.py module for the text path.
#
# The compile cannot happen at import time any more: the checkpointer needs a
# live MongoDB connection, and app/db/mongodb.py has not connected yet when this
# module is first imported. So the lifespan in app/main.py calls
# init_chat_graph() once, and everything else goes through get_chat_graph().

_chat_graph = None


@lru_cache(maxsize=4)
def _compile_graph(checkpointer_instance):
    """
    Cached compilation of the StateGraph against a checkpointer instance.
    Prevents redundant compilation latency on graph retrieval.
    """
    return graph.compile(checkpointer=checkpointer_instance)


def init_chat_graph(checkpointer) -> None:
    """Compile the graph against a checkpointer. Called once, from the lifespan."""
    global _chat_graph
    _chat_graph = _compile_graph(checkpointer)
    logger.info("LangGraph compiled and cached with %s.", type(checkpointer).__name__)


def get_chat_graph():
    """
    The compiled graph.

    The MemorySaver fallback exists so that importing this module outside the
    app (a test, a script, a notebook) still works. It is loud because in a
    running server it means state is silently not being persisted, and — worse
    — that tool approvals cannot survive the round-trip to the browser.
    """
    global _chat_graph

    if _chat_graph is None:
        logger.warning(
            "get_chat_graph() called before init_chat_graph(); falling back to "
            "MemorySaver. Conversation state will NOT persist and tool approval "
            "will not survive a restart."
        )
        _chat_graph = _compile_graph(MemorySaver())

    return _chat_graph
