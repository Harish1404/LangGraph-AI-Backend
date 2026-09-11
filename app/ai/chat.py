import asyncio
import json
import logging
import traceback
from functools import lru_cache

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

from app.ai.models import log_chain, ordered_models

from app.core.tracing import (
    summarize_messages,
    join_tokens,
    set_run_inputs,
    set_run_metadata,
)
from app.memory.store import conversation_store
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_context_text
from app.prompts.router_prompt import (
    TOOL_SYSTEM_PROMPT,
    BOTH_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
)
from app.prompts.voice_prompt import VOICE_SYSTEM_PROMPT
from app.core.config import settings
from app.rag.rag_pipeline import rag_pipeline
from app.ai.agents.router import query_router
from app.ai.agents.tool_agent.toolbelt import TOOLS
from app.ai.agents.tool_agent.tools.weather import get_weather

# logger lets us print debug/error messages to the console with proper labels
logger = logging.getLogger(__name__)

# Strong references to detached history writes. asyncio only keeps a weak
# reference to a running task, so without this a fire-and-forget write can be
# garbage collected mid-flight and silently never happen.
_pending_writes: set[asyncio.Task] = set()


@lru_cache(maxsize=4)
def _build_models(light_tokens: int, reasoning_tokens: int):
    """The model chain for a pair of token budgets, built once per process.

    Ordered fastest-first — see app/ai/models.py for the measurements behind it:

        mistral-small  →  gpt-oss-20b  →  gemini-3.5-flash-lite

    Two budgets rather than one because the chain mixes two kinds of model.
    `light_tokens` goes to the two that emit only visible text; the Groq model
    reasons, and its hidden tokens come out of the same allowance, so it gets
    `reasoning_tokens` instead. Voice passes the same small number for both.

    Constructing these is expensive and, crucially, *not* a one-off cold start:
    every client benefits from being a singleton. They are stateless HTTP
    clients, so one set per budget pair is safe to share across requests, cached
    via @lru_cache.
    """
    primary, *fallbacks = ordered_models(light_tokens, reasoning_tokens)

    llm_with_fallbacks = primary.with_fallbacks(fallbacks)

    # Note the order: bind_tools() must be applied to each *model* first,
    # because with_fallbacks() returns a RunnableWithFallbacks, which has no
    # bind_tools() method of its own.
    # TOOLS comes from the tool agent's toolbelt, so the model's bindings and
    # the graph's ToolNode can never drift apart.
    llm_with_tools = primary.bind_tools(TOOLS).with_fallbacks(
        [model.bind_tools(TOOLS) for model in fallbacks]
    )

    return llm_with_fallbacks, llm_with_tools


def warm_up_models() -> None:
    """Build both budget variants at startup rather than mid-request."""
    log_chain()
    for budgets in (
        (settings.light_max_tokens, settings.reasoning_max_tokens),
        (settings.voice_max_tokens, settings.voice_max_tokens),
    ):
        _build_models(*budgets)


async def warm_up_llm() -> None:
    """Open the connection to Groq's chat endpoint before the first real turn.

    Same reasoning as the STT warm-up in app/ai/voice.py, and measurably worth
    it: the first turn of a fresh process saw first-audio at ~2.4s against
    ~1.3s for every turn after, and the difference was almost entirely the
    first chat/completions call paying for TLS setup.
    """
    try:
        llm, _ = _build_models(
            settings.voice_max_tokens, settings.voice_max_tokens
        )
        async for _ in llm.astream([HumanMessage(content="hi")]):
            break  # one token is enough to establish the connection
        logger.info("LLM connection warmed up")
    except Exception as e:
        logger.warning(f"LLM warm-up skipped: {e}")


def content_to_text(content) -> str:
    """
    Flattens a message's content down to plain text.

    Groq hands back a plain string, but Gemini — which is what the fallback
    switches to whenever Groq rate-limits — can return a list of content
    parts instead. A raw list breaks in two places at once: Starlette calls
    .encode() on whatever the stream yields, and the history accumulator
    joins the tokens into one answer. Both need a str.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                pieces.append(part.get("text") or "")
        return "".join(pieces)

    if content is None:
        return ""

    return str(content)


def format_sse_event(event_type: str, content: str) -> str:
    """Formats event data into standard Server-Sent Events (SSE) string."""
    return f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"


# ─────────────────────────────────────────────────────────
# SECTION 1: ChatService (100% Pure LangChain streaming chat)
# ─────────────────────────────────────────────────────────

class ChatService:
    """
    Handles a single user chat message.

    A router first classifies the query, then exactly one path runs:

        RAG    -> retrieve resume chunks, answer from them (no tools)
        TOOL   -> call the weather tool, answer from its result (no retrieval)
        BOTH   -> retrieve resume chunks AND call the weather tool
        DIRECT -> answer from the model's own knowledge (neither)

    This is the fix for the old behaviour, where retrieval always ran and the
    weather tool was always available, so both fired for every single question.

    The service is also the memory boundary: it is handed the conversation's
    recent history to replay, and it is what writes the finished answer back.
    """

    def __init__(
        self,
        user_prompt: str,
        conversation_id: str,
        history: list[BaseMessage] | None = None,
        voice_mode: bool = False,
    ):
        self.user_prompt = user_prompt
        self.conversation_id = conversation_id
        self.voice_mode = voice_mode

        # The window buffer, already trimmed to the last k turns by the caller.
        self.history = list(history or [])

        # Voice answers are spoken, so they need different shaping than text
        # ones: a few sentences, no markdown, numbers written how they sound.
        # Injecting it at the head of the history rather than editing each of
        # the four branch generators means it applies to RAG, TOOL, BOTH and
        # DIRECT alike, and lands directly after that branch's own system
        # prompt. The cap is also a cost control — on the ElevenLabs free tier
        # a 2000-character reply is a tenth of the month's credits.
        if voice_mode:
            self.history.insert(0, SystemMessage(content=VOICE_SYSTEM_PROMPT))
        # Voice caps every model at the same small number; text uses the
        # two-tier pair so the reasoning model gets room for its hidden tokens.
        budgets = (
            (settings.voice_max_tokens, settings.voice_max_tokens)
            if voice_mode
            else (settings.light_max_tokens, settings.reasoning_max_tokens)
        )

        # Filled in once the router has run.
        self.route: str | None = None

        # What retrieval actually searches for: the question with its
        # back-references resolved. Falls back to the raw prompt.
        self.search_query = user_prompt
        
        # Tool activity, recorded for the stored message's metadata.
        self.tool_calls: list[dict] = []

        self.gemini_key = settings.gemini_api_key
        self.groq_key   = settings.groq_api_key

        # Shared across requests — see _build_models for why this is not done
        # inline here any more.
        self.llm_with_fallbacks, self.llm_with_tools = _build_models(*budgets)

    # ── Entry point ──────────────────────────────────────────────────────────

    @traceable(
        run_type="chain",
        name="chatbot_request",
        reduce_fn=join_tokens,
    )
    async def chat(self):
        """
        Routes the query, then streams plain text tokens from the chosen path.

        This is also the ROOT of the LangSmith trace. Every LLM call, tool call
        and retrieval below it nests under this one span, which is what turns a
        handful of disconnected dashboard rows into a single request tree.
        """
        # This method takes no arguments other than `self`, which langsmith
        # strips — so the question has to be published onto the span by hand,
        # or the dashboard would show the answer with no sign of the input.
        set_run_inputs(
            user_prompt=self.user_prompt,
            conversation_id=self.conversation_id,
        )

        # Nothing else buffers the answer — the tokens are streamed straight to
        # the client — so it is collected here to be written to MongoDB.
        parts: list[str] = []
        completed = False
        abandoned = False

        try:
            decision = await query_router.route(self.user_prompt, self.history)
            self.route = decision.route
            self.search_query = decision.standalone_question

            logger.info(
                f"Router selected: {self.route} for query: {self.user_prompt!r} "
                f"(searching for: {self.search_query!r})"
            )

            # The route is only known now, so it is attached at runtime rather
            # than declared on the decorator. Lets you filter the dashboard by
            # metadata.route to compare paths. thread_id is what groups every
            # turn of one conversation into a single LangSmith thread.
            set_run_metadata(
                route=self.route,
                conversation_id=self.conversation_id,
                thread_id=self.conversation_id,
                standalone_question=self.search_query,
                history_messages=len(self.history),
            )

            if self.route == "RAG":
                stream = self._rag_stream()
            elif self.route == "TOOL":
                stream = self._tool_stream()
            elif self.route == "BOTH":
                stream = self._both_stream()
            else:
                stream = self._direct_stream()

            async for token in stream:
                parts.append(token)
                yield token

            completed = True

        except GeneratorExit:
            # The client went away mid-answer and this generator is being torn
            # down. Flagged rather than handled here, so the finally below can
            # tell an abandoned stream apart from a finished one.
            abandoned = True
            raise

        except Exception as e:
            logger.error(f"Chat pipeline execution failed: {e}\n{traceback.format_exc()}")
            yield f"\n[ERROR: Chat service error — {e}]"

        finally:
            answer = "".join(parts)

            if abandoned:
                # Cannot await here. An abandoned async generator is finalized
                # by the event loop after aclose() has already returned, and an
                # await at that point is cancelled — the write would be issued
                # and then silently dropped. Handing it to an independent task
                # gets it off this dying frame. Best-effort by nature.
                self._schedule_persist(answer)
            else:
                # Normal and error paths still have a live request behind them,
                # so this is awaited: the next turn must not be able to load
                # history before this answer has landed.
                await self._persist_answer(answer, completed)

    # ── Route: RAG ───────────────────────────────────────────────────────────

    @traceable(
        run_type="chain",
        name="rag_answer",
        reduce_fn=join_tokens,
    )
    async def _rag_stream(self):
        """Answers strictly from resume chunks retrieved out of MongoDB Atlas."""
        set_run_inputs(user_prompt=self.user_prompt, search_query=self.search_query)

        # Retrieval uses the router's rewritten question, not the raw one: a
        # follow-up such as "and where did he study?" has nothing to embed.
        retrieved_chunks = await rag_pipeline.retrieve(self.search_query)
        context_text = build_context_text(retrieved_chunks)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            *self.history,
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {self.user_prompt}")
        ]

        async for chunk in self.llm_with_fallbacks.astream(messages):
            text = content_to_text(chunk.content)
            if text:
                yield text

    # ── Route: TOOL ──────────────────────────────────────────────────────────

    @traceable(
        run_type="chain",
        name="tool_answer",
        reduce_fn=join_tokens,
    )
    async def _tool_stream(self):
        """Answers using the weather tool only. No retrieval happens here."""
        set_run_inputs(user_prompt=self.user_prompt)

        # A fresh list every call: _run_tool_loop appends to what it is given,
        # and self.history must never be mutated — it is replayed as-is and
        # would otherwise accumulate tool messages across turns.
        messages = [
            SystemMessage(content=TOOL_SYSTEM_PROMPT),
            *self.history,
            HumanMessage(content=self.user_prompt)
        ]

        async for token in self._run_tool_loop(messages):
            yield token

    # ── Route: BOTH ──────────────────────────────────────────────────────────

    @traceable(
        run_type="chain",
        name="rag_plus_tool_answer",
        reduce_fn=join_tokens,
    )
    async def _both_stream(self):
        """
        Retrieves resume context first (so the model can read a city out of it),
        then runs the same tool loop.
        """
        set_run_inputs(user_prompt=self.user_prompt, search_query=self.search_query)

        retrieved_chunks = await rag_pipeline.retrieve(self.search_query)
        context_text = build_context_text(retrieved_chunks)

        messages = [
            SystemMessage(content=BOTH_SYSTEM_PROMPT),
            *self.history,
            HumanMessage(content=f"Resume context:\n{context_text}\n\nQuestion: {self.user_prompt}")
        ]

        async for token in self._run_tool_loop(messages):
            yield token

    # ── Route: DIRECT ────────────────────────────────────────────────────────

    @traceable(
        run_type="chain",
        name="direct_answer",
        reduce_fn=join_tokens,
    )
    async def _direct_stream(self):
        """Answers from the model's own knowledge. No retrieval, no tools."""
        set_run_inputs(user_prompt=self.user_prompt)

        messages = [
            SystemMessage(content=DIRECT_SYSTEM_PROMPT),
            *self.history,
            HumanMessage(content=self.user_prompt)
        ]

        async for chunk in self.llm_with_fallbacks.astream(messages):
            text = content_to_text(chunk.content)
            if text:
                yield text

    # ── The tool-calling loop, written out by hand ───────────────────────────

    @traceable(
        run_type="chain",
        name="tool_loop",
        process_inputs=summarize_messages,
        reduce_fn=join_tokens,
    )
    async def _run_tool_loop(self, messages: list):
        """
        A single tool-call round-trip, using only langchain-core primitives:

          1. Ask the tool-aware model what to do.
          2. If it asked for tools, run each one and append a ToolMessage.
          3. Stream the final answer from the *plain* model, so it summarizes
             the tool results instead of calling the tool all over again.

        Because we own this loop, ToolMessage content is never yielded to the
        client — the raw webhook JSON can no longer leak into the stream.
        """
        ai_msg = await self.llm_with_tools.ainvoke(messages)

        # The model answered directly without reaching for a tool.
        if not getattr(ai_msg, "tool_calls", None):
            text = content_to_text(ai_msg.content)
            if text:
                yield text
            return

        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            name = tool_call.get("name")
            if name != get_weather.name:
                logger.warning(f"Model requested unknown tool {name!r}; skipping.")
                continue

            logger.info(f"Calling tool {name} with args {tool_call.get('args')}")

            # Kept for the stored message's metadata. Note this is recorded,
            # not replayed — see ConversationStore.append_message.
            self.tool_calls.append({"name": name, "args": tool_call.get("args")})

            try:
                result = await get_weather.ainvoke(tool_call["args"])
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                result = f"Tool error: {e}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        async for chunk in self.llm_with_fallbacks.astream(messages):
            text = content_to_text(chunk.content)
            if text:
                yield text

    # ── Persistence ─────────────────────────────────────────────────────────

    def _schedule_persist(self, answer: str) -> None:
        """
        Saves a cut-off answer from outside this generator's teardown.

        Synchronous on purpose — it only schedules the write. See the note in
        chat()'s finally for why awaiting from a finalizing generator does not
        work: aclose() has already returned by then and the await is cancelled.
        """
        if not answer.strip():
            return

        try:
            task = asyncio.create_task(self._persist_answer(answer, completed=False))
        except RuntimeError as e:
            # No running loop — the app is shutting down. Nothing to be done.
            logger.warning(
                f"Could not schedule partial answer for {self.conversation_id}: {e}"
            )
            return

        _pending_writes.add(task)
        task.add_done_callback(_pending_writes.discard)

    async def _persist_answer(self, answer: str, completed: bool) -> None:
        """
        Writes the finished answer to MongoDB so the next turn can see it.

        Two deliberate choices here. Nothing is stored for an empty answer,
        which keeps a failed turn's error text out of the replayed history.
        And a write failure is logged, never raised — the user already has
        their answer on screen by this point, and turning a storage hiccup
        into a broken response would be a worse trade.
        """
        if not answer.strip():
            return

        try:
            await conversation_store.append_message(
                self.conversation_id,
                role="assistant",
                content=answer,
                route=self.route,
                tool_calls=self.tool_calls,
                partial=not completed,
            )
        except Exception as e:
            logger.error(
                f"Failed to persist assistant message for {self.conversation_id}: {e}"
            )

    # ────────────────────────────────────────────────────────────────────────
    # OPTIONAL: SSE EVENT STREAMING (Uncomment when connecting to a Frontend UI)
    # ────────────────────────────────────────────────────────────────────────
    # async def chat_sse(self):
    #     """
    #     Same routing as chat(), but emits rich SSE events (status updates and
    #     text tokens) instead of bare text, for a Frontend UI to render.
    #     """
    #     try:
    #         yield format_sse_event("status", "Understanding your question...")
    #         route = await query_router.route(self.user_prompt)
    #         yield format_sse_event("status", f"Route selected: {route}")
    #
    #         if route == "RAG":
    #             yield format_sse_event("status", "Searching knowledge base...")
    #             stream = self._rag_stream()
    #         elif route == "TOOL":
    #             yield format_sse_event("status", "Checking the weather...")
    #             stream = self._tool_stream()
    #         elif route == "BOTH":
    #             yield format_sse_event("status", "Searching knowledge base and checking the weather...")
    #             stream = self._both_stream()
    #         else:
    #             yield format_sse_event("status", "Thinking...")
    #             stream = self._direct_stream()
    #
    #         async for token in stream:
    #             yield format_sse_event("token", token)
    #
    #         yield format_sse_event("done", "Response complete")
    #
    #     except Exception as e:
    #         logger.error(f"Chat pipeline execution failed: {e}\n{traceback.format_exc()}")
    #         yield format_sse_event("error", f"Chat service error — {e}")


# ─────────────────────────────────────────────────────────
# SECTION 2: LangGraph Streaming (replaces ChatService for text chat)
# ─────────────────────────────────────────────────────────

# Which graph nodes produce text meant for the user now lives with the agents
# that own those nodes — see ANSWER_NODES in app/ai/agents/graph.py, which
# unions one small set per agent.
#
# It is read through a lazy import inside _stream_graph rather than imported at
# module scope, for the same reason get_chat_graph is: the agents' node modules
# import _build_models from this file, so a top-level import here would close
# the loop into a circular import.


def sse(event: str, data: dict) -> str:
    """
    One Server-Sent Event, properly framed.

    The stream carries more than text now — a run can stop halfway to ask the
    user to approve a tool call — so the client needs to tell the kinds apart.
    Named events do that with no in-band escaping and no sentinel string the
    model could ever produce by accident.

    Event types on this stream:
      token      {"t": "..."}      a piece of the answer
      interrupt  {...}             the run paused; payload says what for
      truncated  {"reason": "length"}  the answer hit the token ceiling
      done       {"conversation_id": "..."}
      error      {"detail": "..."}
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _first_interrupt(snapshot) -> dict | None:
    """
    Walk a state snapshot and its nested subgraph snapshots for an interrupt.

    The recursion is the point. interrupt() is called by approve_tools, which
    lives *inside* the tool agent subgraph, so the pending interrupt hangs off
    the state of the supervisor's `tool_agent` task rather than off the
    supervisor's own task list. Looking only one level deep finds nothing,
    which reads exactly like "this thread is not waiting on anything" — the
    resume endpoint would 409 and tool approval would be dead.
    """
    for task in snapshot.tasks:
        for value in getattr(task, "interrupts", ()) or ():
            return value.value

        # Populated only when aget_state was called with subgraphs=True.
        nested = getattr(task, "state", None)
        if nested is not None and hasattr(nested, "tasks"):
            found = _first_interrupt(nested)
            if found is not None:
                return found

    return None


async def _pending_interrupt(graph, config) -> dict | None:
    """
    What this thread is waiting on, or None if it ran to completion.

    An interrupt is not an event on the stream — it is a state the graph is
    left sitting in — so it has to be read back off the checkpoint after the
    run unwinds.
    """
    snapshot = await graph.aget_state(config, subgraphs=True)

    return _first_interrupt(snapshot)


async def _stream_graph(graph_input, conversation_id: str):
    """
    Drive the graph and frame everything it produces as SSE.

    Shared by the first turn (stream_chat) and by the continuation after an
    approval (resume_chat). The only difference between the two is what goes in
    at the top: a new message, or a Command carrying the user's decision.
    """
    # Lazy import to avoid circular dependency (graph.py imports from chat.py)
    from app.ai.agents.graph import ANSWER_NODES, get_chat_graph

    graph = get_chat_graph()

    # thread_id is what the checkpointer keys conversation state by.
    # Our conversation_id maps directly to it.
    config = {"configurable": {"thread_id": conversation_id}}

    parts: list[str] = []
    interrupted = False
    truncated = False

    try:
        # astream_events gives us fine-grained events from every node.
        # We filter for "on_chat_model_stream" to get LLM tokens.
        async for event in graph.astream_events(graph_input, config, version="v2"):
            if event["event"] != "on_chat_model_stream":
                continue

            # Which node this model call belongs to — see ANSWER_NODES.
            if (event.get("metadata") or {}).get("langgraph_node") not in ANSWER_NODES:
                continue

            chunk = event["data"]["chunk"]

            # Did the model stop because it ran out of budget rather than
            # because it was finished? Checked on every chunk, not at the end:
            # the provider attaches finish_reason to the chunk that carries it
            # and the *final* chunk's metadata is empty, so reading only the
            # last one finds nothing.
            #
            # Filtering by ANSWER_NODES above also keeps the router out of
            # this — its classifier is capped far lower and legitimately runs
            # to its limit without that meaning anything to the user.
            if (getattr(chunk, "response_metadata", None) or {}).get(
                "finish_reason"
            ) == "length":
                truncated = True

            token = content_to_text(chunk.content)
            if token:
                parts.append(token)
                yield sse("token", {"t": token})

        pending = await _pending_interrupt(graph, config)

        if pending:
            interrupted = True
            yield sse("interrupt", pending)
        else:
            # Announced before `done` so the client can attach it to the
            # message it is about to commit.
            if truncated:
                # Which ceiling it hit depends on which model answered, so both
                # are logged rather than guessing.
                logger.warning(
                    "Answer for %s hit its token ceiling "
                    "(LIGHT_MAX_TOKENS=%d, REASONING_MAX_TOKENS=%d).",
                    conversation_id,
                    settings.light_max_tokens,
                    settings.reasoning_max_tokens,
                )
                yield sse("truncated", {"reason": "length"})

            yield sse("done", {"conversation_id": conversation_id})

    except Exception as e:
        logger.error(f"LangGraph chat failed: {e}\n{traceback.format_exc()}")
        yield sse("error", {"detail": f"Chat service error — {e}"})

    finally:
        # Persist the answer to MongoDB so it shows up in the conversation
        # list, transcript, and sidebar. The checkpointer handles LLM memory,
        # but the UI still reads from MongoDB.
        #
        # Not on the interrupt path: the turn is not over, and whatever the
        # model said before asking for approval ("Let me check that for you…")
        # is not the answer. Writing it would leave a stub in the transcript
        # that the real answer then appears *after*.
        answer = "".join(parts)
        if answer.strip() and not interrupted:
            try:
                await conversation_store.append_message(
                    conversation_id,
                    role="assistant",
                    content=answer,
                    # Stored so the "cut off" notice survives a reload — the
                    # transcript is what the UI reads back, and an incomplete
                    # answer should not look finished the second time either.
                    partial=truncated,
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist assistant message for {conversation_id}: {e}"
                )


async def stream_chat(user_prompt: str, conversation_id: str):
    """
    One turn of text chat: POST /chatbot. Voice mode still uses ChatService.

    The input is only the new message — the checkpointer prepends this thread's
    history from MongoDB, so there is no manual memory management here.

    The stream may end in `done` (answer complete) or in `interrupt` (the graph
    paused for tool approval and is waiting on POST /chatbot/{id}/resume).
    """
    async for frame in _stream_graph(
        {"messages": [HumanMessage(content=user_prompt)]},
        conversation_id,
    ):
        yield frame


async def resume_chat(conversation_id: str, decision: dict):
    """
    Continue a thread that paused for tool approval: POST /chatbot/{id}/resume.

    Command(resume=...) does not restart the graph — it reloads the checkpoint,
    re-runs the interrupted node with `decision` as the return value of its
    interrupt() call, and carries on from there. The user's original question is
    still in state; it does not need to be sent again.
    """
    from langgraph.types import Command

    async for frame in _stream_graph(Command(resume=decision), conversation_id):
        yield frame


async def get_pending_approval(conversation_id: str) -> dict | None:
    """
    What this thread is waiting on, without advancing it.

    Backs GET /chatbot/{id}/pending, which is what lets a browser refresh — or
    a server restart — land back on the approval prompt instead of on a
    conversation that appears to have stopped mid-sentence.
    """
    from app.ai.agents.graph import get_chat_graph

    return await _pending_interrupt(
        get_chat_graph(),
        {"configurable": {"thread_id": conversation_id}},
    )

