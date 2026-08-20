import asyncio
import json
import logging
import traceback
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

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
from app.ai.router import query_router
from app.tools.weather import get_weather

# logger lets us print debug/error messages to the console with proper labels
logger = logging.getLogger(__name__)

# Strong references to detached history writes. asyncio only keeps a weak
# reference to a running task, so without this a fire-and-forget write can be
# garbage collected mid-flight and silently never happen.
_pending_writes: set[asyncio.Task] = set()


@lru_cache(maxsize=4)
def _build_models(max_tokens: int):
    """The model trio for a given token budget, built once per process.

    Constructing these is expensive and, crucially, *not* a one-off cold start:
    measured here, ChatGoogleGenerativeAI takes ~740ms and ChatGroq ~490ms
    EVERY time, so building them per request put a flat ~1.2s in front of every
    answer — voice and text alike — before a single byte was sent to any API.

    They are stateless HTTP clients, so one set per token budget is safe to
    share across requests, the same way app/ai/router.py already keeps a
    module-level singleton. Cached on max_tokens because that is the only thing
    that varies (500 for text, ~120 for voice), which means two entries.
    """
    primary_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=settings.groq_api_key,
        temperature=0.7,
        max_tokens=max_tokens,
    )

    fallback_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.7,
        max_output_tokens=max_tokens,
    )

    llm_with_fallbacks = primary_llm.with_fallbacks([fallback_llm])

    # Note the order: bind_tools() must be applied to each *model* first,
    # because with_fallbacks() returns a RunnableWithFallbacks, which has no
    # bind_tools() method of its own.
    llm_with_tools = primary_llm.bind_tools([get_weather]).with_fallbacks(
        [fallback_llm.bind_tools([get_weather])]
    )

    return llm_with_fallbacks, llm_with_tools


def warm_up_models() -> None:
    """Build both token-budget variants at startup rather than mid-request."""
    for budget in (500, settings.voice_max_tokens):
        _build_models(budget)


async def warm_up_llm() -> None:
    """Open the connection to Groq's chat endpoint before the first real turn.

    Same reasoning as the STT warm-up in app/ai/voice.py, and measurably worth
    it: the first turn of a fresh process saw first-audio at ~2.4s against
    ~1.3s for every turn after, and the difference was almost entirely the
    first chat/completions call paying for TLS setup.
    """
    try:
        llm, _ = _build_models(settings.voice_max_tokens)
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
        max_tokens = settings.voice_max_tokens if voice_mode else 500

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
        self.llm_with_fallbacks, self.llm_with_tools = _build_models(max_tokens)

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

async def stream_chat(user_prompt: str, conversation_id: str):
    """
    Invoke the LangGraph and stream tokens back to the caller.

    This is the new entry point for text chat (POST /chatbot).
    Voice mode still uses the ChatService above.

    How it works:
      1. We import the compiled graph (chat_graph) from app/ai/graph.py.
      2. We pass the user message + a thread_id config to astream_events().
      3. LangGraph's MemorySaver checkpointer automatically loads/saves
         conversation history for this thread_id — no manual memory management.
      4. We yield text tokens as they stream from the LLM.
      5. After streaming, we persist the final answer to MongoDB so the
         conversation list and transcript endpoints still work.
    """
    # Lazy import to avoid circular dependency (graph.py imports from chat.py)
    from app.ai.graph import chat_graph

    # thread_id is what the MemorySaver uses to key conversation state.
    # Our conversation_id maps directly to it.
    config = {"configurable": {"thread_id": conversation_id}}

    # The input to the graph: just the new user message.
    # The checkpointer will automatically prepend the conversation history.
    input_state = {"messages": [HumanMessage(content=user_prompt)]}

    # Collect the full answer for persistence
    parts: list[str] = []

    try:
        # astream_events gives us fine-grained events from every node.
        # We filter for "on_chat_model_stream" to get LLM tokens.
        async for event in chat_graph.astream_events(input_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = content_to_text(chunk.content)
                if token:
                    parts.append(token)
                    yield token

    except Exception as e:
        logger.error(f"LangGraph chat failed: {e}\n{traceback.format_exc()}")
        yield f"\n[ERROR: Chat service error — {e}]"

    finally:
        # Persist the answer to MongoDB so it shows up in the conversation
        # list, transcript, and sidebar. The checkpointer handles LLM memory,
        # but the UI still reads from MongoDB.
        answer = "".join(parts)
        if answer.strip():
            try:
                await conversation_store.append_message(
                    conversation_id,
                    role="assistant",
                    content=answer,
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist assistant message for {conversation_id}: {e}"
                )

