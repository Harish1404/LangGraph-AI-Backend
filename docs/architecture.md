# System Architecture

> **How to read this document.** Start with §1 for the one-paragraph summary, then
> §2 for the map. §3–§8 walk through one request end to end. §9 collects the design
> decisions and the reasoning behind each one — that section is the most useful if
> you are learning from this codebase rather than working on it.
>
> **§12 covers authentication.** It is a gate in front of everything in §3–§8: no
> request reaches `ChatService` without an authenticated user, and every stored
> document is owned by one. Read it before changing anything that touches `user_id`.
>
> The spoken path — microphone in, synthesised speech out — has its own walkthrough in
> [voice-mode.md](voice-mode.md).

---

## 1. What this system is

A **retrieval-augmented chatbot with persistent conversation memory**. A user asks a
question; the system decides *how* to answer it (search a document, call a tool, both,
or neither), remembers what was said before, and streams the answer back token by token.

Four ideas do most of the work:

| Idea | In one line |
|---|---|
| **Routing** | Not every question needs a database search. Decide first, then do only that work. |
| **Retrieval** | Answers about the resume come from the resume, not from the model's memory of the internet. |
| **Memory** | The last few exchanges are replayed to the model, so follow-up questions make sense. |
| **Identity** | Every thread belongs to someone. Who that is comes from a signed cookie, never from the request body. |

---

## 2. The map

Two ways in — a streaming HTTP endpoint for typed questions and a WebSocket for spoken
ones — converging on the same `ChatService`, both behind the same auth gate. Everything
above the database is stateless; all state lives in MongoDB, so restarting the server
loses nothing.

```mermaid
graph TD
    CLERK[[Clerk<br/>identity provider]] -.sign-in.-> UI
    CLERK -.webhooks.-> WH(POST /webhooks/clerk<br/>routes/webhooks.py)

    UI[Client<br/>session cookie + conversation_id] --> GATE
    GATE{{get_current_user<br/>api/deps.py}} --> CHATEP(POST /chatbot<br/>routes/chatbot.py)
    GATE --> VOICEEP(WS /ws/voice<br/>routes/voice.py)
    GATE --> CONVEP(/conversations CRUD<br/>routes/conversations.py)
    GATE --> AUTHEP(/auth/me, /auth/refresh<br/>routes/auth.py)

    AUTHEP --> AUTHSVC(AuthService<br/>services/auth_service.py)
    WH --> AUTHSVC
    AUTHSVC --> REPOS(repositories/<br/>user, profile, session)
    REPOS --> DB

    CHATEP --> STREAM(stream_chat<br/>ai/chat.py)
    STREAM --> GRAPH(LangGraph StateGraph<br/>chat_graph<br/>ai/graph.py)

    GRAPH --> ROUTER(QueryRouter<br/>route + rewritten question<br/>ai/router.py)
    GRAPH --> CHECKPOINT[MemorySaver checkpointer<br/>LangGraph state history]
    GRAPH --> PIPE{RAGPipeline.retrieve}
    GRAPH --> TOOLNODE[ToolNode: get_weather]
    GRAPH --> LLM[deepseek-v4-flash -> mistral-small<br/>-> gpt-oss-20b -> gemini-2.5-flash]

    VOICEEP --> STT(Groq Whisper<br/>ai/voice.py)
    STT --> SERVICE(ChatService<br/>legacy voice path<br/>ai/chat.py)
    SERVICE --> ROUTER
    SERVICE --> WINDOW(window.py<br/>last k turns)
    SERVICE --> PIPE
    SERVICE --> HOOK[Weather webhook]
    SERVICE --> LLM
    SERVICE --> TTS(ElevenLabs stream-input<br/>ai/voice.py)
    TTS --> VOICEEP

    ROUTER --> LLM
    PIPE -->|Dense| VEC[(Vector search)]
    PIPE -->|Sparse| KEY[(Keyword search)]
    VEC --> RRF(RRF merge)
    KEY --> RRF
    RRF --> DB[(MongoDB Atlas<br/>rag_db)]

    WINDOW --> STORE(store.py<br/>reads / writes messages)
    GRAPH --> STORE
    CONVEP --> STORE
    STORE --> DB
```

**Why these boundaries?** `store.py` knows about MongoDB and nothing else — no LangChain
types, no LLM calls. `window.py` knows about LangChain messages but not about queries or
collections. That split means you can change how memory is *assembled* without touching
how it is *stored*, and vice versa.

The auth side repeats the pattern one layer deeper: `repositories/` know only Mongo,
`services/` hold the policy, and `api/deps.py` is the only place a route learns who is
calling. A route never queries a user collection directly.

---

## 3. The database

Six collections in `rag_db`: the knowledge base, two for the chat, and three for auth
(documented in §12). `mongodb.py` also exposes a `connector` accessor, but nothing writes
to it yet — it is a placeholder for the MCP work, not a live collection.

### `vector_documents` — the knowledge base
Chunks of the ingested documents, each with a 768-dimensional embedding. Written once at
startup, read on every RAG query. See [workflow.md](workflow.md) §1.

### `conversations` — one document per chat thread

```jsonc
{
  "_id": "conv_9f8b2c1e-4a77-4d1e-9c33-6b0f5d2a1e88",
  "user_id": "usr_1827a0e5-8130-4919-9e58-761999246e75",  // -> users._id
  "title": "Who is Harish",          // auto-derived from the first user message
  "message_count": 6,                 // doubles as the sequence allocator
  "last_message_preview": "He works at ...",
  "created_at": "...", "updated_at": "...",
  "deleted": false                    // soft delete
}
```

### `messages` — one document per message

```jsonc
{
  "_id": "msg_3c7d81a0-55be-42f2-8d94-71ab0c9e2f10",
  "conversation_id": "conv_9f8b2c1e-...",
  "seq": 3,                           // order within the thread: 0, 1, 2, 3, ...
  "role": "user",                     // or "assistant"
  "content": "and where did he study?",
  "route": "RAG",                     // assistant messages only
  "tool_calls": [ ... ],              // metadata only — see §6
  "partial": false,                   // true if the stream was cut short
  "created_at": "..."
}
```

**Indexes**
- `messages`: `(conversation_id, seq)` — **unique**
- `conversations`: `(user_id, updated_at desc)` — powers the sidebar

Auth indexes are listed in §12.

> **Teaching note — the unique index is not just for speed.** It is the last line of
> defence against two concurrent requests in the same thread being handed the same `seq`.
> The database refuses the duplicate rather than silently interleaving the conversation.

> **`user_id` is never taken from the request.** Every read and write above is filtered
> by the id resolved from the session cookie. An earlier version accepted `user_id` as a
> field on `ChatRequest`, which meant the caller declared their own identity — see §12
> for what that allowed and how it was closed.

---

## 4. How a `seq` number is allocated

Every message needs a position in the thread, and two requests may arrive at once. The
allocation is therefore a **single atomic operation**:

```python
conversation = await self.conversations.find_one_and_update(
    {"_id": conversation_id, "deleted": {"$ne": True}},
    {"$inc": {"message_count": 1}, "$set": {"updated_at": now}},
    return_document=ReturnDocument.AFTER,
)
seq = conversation["message_count"] - 1
```

MongoDB guarantees `$inc` is atomic, so each caller gets a distinct number. Reading the
count and then writing `count + 1` in two steps would be a classic race.

---

## 5. The memory layer

### The window buffer

Replaying an entire conversation to the model would grow without bound and cost more on
every turn. Instead only the **last k turns** are replayed — a *window buffer*.

A **turn** is a user message *plus* its answer. The window is measured in turns rather
than raw messages on purpose: a window that cuts between a question and its answer hands
the model a dangling question and produces visibly confused replies.

| Setting | Default | Meaning |
|---|---|---|
| `WINDOW_K` | 4 | turns replayed normally |
| `WINDOW_K_LARGE` | 5 | turns once the thread is long |
| `LARGE_HISTORY_THRESHOLD` | 100 | turn count at which the wider window applies |

All three are read from `.env` via `app/core/config.py`, so tuning needs no code change.

```mermaid
graph LR
    OLD[Turns 1-8<br/>in MongoDB<br/>NOT replayed] -. forgotten .-> PROMPT
    KEEP(Turns 9-12<br/>the window) --> PROMPT
    NEW(Turn 13<br/>the new question) --> PROMPT[Prompt sent to the model]
```

Nothing is ever deleted — turns 1–8 stay in the database and are still returned by
`GET /conversations/{id}`. They are simply not sent to the model.

### What gets replayed — and what does not

Only plain user/assistant **text** is rebuilt into `HumanMessage` / `AIMessage`. Tool
activity is stored in the `tool_calls` field for observability but is **never** replayed.

> **Teaching note — why tool messages must not be replayed.** A tool call is a pair: an
> assistant message saying *"call `get_weather(Chennai)`"* and a matching tool message
> carrying the result. Replay the first without the second and the Groq API rejects the
> whole request with a 400. The finished answer already contains everything the tool
> produced, so nothing is lost by storing only the text.

### Where history goes in the prompt

History is spliced between the system prompt and the current question:

```python
messages = [
    SystemMessage(content=RAG_SYSTEM_PROMPT),   # the rules
    *self.history,                              # what was said before
    HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {self.user_prompt}"),
]
```

The system prompt stays first so its instructions are not buried, and the current question
stays last so it is the most recent thing the model reads.

### Multi-Agent Supervisor, Subgraphs & Checkpointer

For standard text chat (`POST /chatbot`), the work is split across a **supervisor graph and specialist agent subgraphs** under `app/ai/agents/` (entered via `agents/graph.py`):

1. **State Definitions (`agents/state.py`)**:
   ```python
   class ChatState(TypedDict):                  # the supervisor's state
       messages: Annotated[list, add_messages]  # auto-appended message history
       route: str                               # "RAG" | "TOOL" | "BOTH" | "DIRECT"
       search_query: str                        # rewritten question from router
       context: str                             # retrieved RAG context
       user_prompt: str                         # this turn's question, verbatim
       denied_tools: list[str]                  # tools the user refused this turn
   ```
   `RagState` and `ToolState` are **subsets** of `ChatState`, sharing key names and the `add_messages` reducer. That is what lets each compiled subgraph be added to the supervisor as a plain node: LangGraph filters the parent state down on the way in and merges the returned keys back on the way out, with no adapter code in between.

2. **Agent Boundaries**:
   - **Supervisor** — `route_query` classification, `pick_route` dispatch, the `DIRECT` answer, and the sequencing of `BOTH` (`rag_agent` → `tool_agent`).
   - **`rag_agent`** — retrieval and RAG-grounded generation. Reads `route` itself so it can stop after retrieving on `BOTH`.
   - **`tool_agent`** — the toolbelt (`toolbelt.TOOLS`, also what `_build_models` binds), the `ToolNode`, and the `approve_tools` gate.

3. **`MongoDBSaver` Checkpointer**:
   The **supervisor** is compiled with the checkpointer; the subgraphs are compiled bare. The parent's checkpointer propagates down to them, so passing `config={"configurable": {"thread_id": conversation_id}}` persists every agent's state on the same thread — and lets an `interrupt()` raised deep inside `tool_agent` survive the round-trip to the browser. Reading that pending interrupt back therefore requires `aget_state(config, subgraphs=True)` and a recursive walk (`_first_interrupt` in `app/ai/chat.py`); a one-level lookup finds nothing and looks exactly like "not waiting on anything".

4. **Built-In `ToolNode`**:
   Rather than running custom manual loops, `tool_agent` leverages LangGraph's prebuilt `ToolNode(TOOLS)`. When an LLM returns `tool_calls`, a conditional edge (`should_continue`) routes execution to `approve_tools`, which either forwards to `ToolNode` — appending `ToolMessage` results automatically before looping back to the LLM — or, on refusal, ends the agent with `denied_tools` set so the supervisor can answer without tools.

---

## 6. The router — two jobs in one call

The router decides the route **and** rewrites the question so it stands on its own. Both
come back from a single model call:

```python
class RouteDecision(BaseModel):
    route: Literal["RAG", "TOOL", "BOTH", "DIRECT"]
    standalone_question: str
```

| Route | Meaning |
|---|---|
| `RAG` | Answer from the resume. Retrieval only. |
| `TOOL` | Answer from the weather tool. No retrieval. |
| `BOTH` | Needs the resume *and* the tool (e.g. weather in a city named in the resume). |
| `DIRECT` | Greetings, general knowledge — neither. |

### Why the rewrite matters

This is the part that makes memory actually useful for retrieval:

| Turn | User types | Router rewrites to | Searched for |
|---|---|---|---|
| 1 | "Who is Harish?" | "Who is Harish?" | ✅ |
| 2 | "what are his skills?" | "what are Harish Palanivel's skills?" | ✅ |
| 3 | "and where did he study?" | "Where did Harish study?" | ✅ |

Turn 3 on its own contains **no searchable terms at all** — embedding "and where did he
study?" returns nothing useful no matter how good the vector store is. The rewritten
question is what gets embedded.

> **Teaching note — why one call and not two.** The classic LangChain pattern uses a
> separate "condense question" chain before the router. That is cleaner in theory but
> doubles the latency on the critical path, and both jobs need exactly the same input
> (history + the new question). Merging them is free accuracy.

### Structured output with a fallback

```python
llm = primary_llm.with_structured_output(RouteDecision).with_fallbacks(
    [fallback_llm.with_structured_output(RouteDecision)]
)
```

> **Teaching note — the ordering rule.** `with_structured_output()` is applied to each
> *model* first, then the pair is wrapped in `with_fallbacks()`. Doing it the other way
> round fails: `with_fallbacks()` returns a `RunnableWithFallbacks`, which has no
> `with_structured_output()` method of its own. The same rule applies to `bind_tools()`
> in `chat.py`.

The router **never raises**. Any failure — bad JSON, an unrecognised route, a timeout —
degrades to `RAG` with the original question untouched.

---

## 7. Generation

### The model chain

Every model in the app is constructed in one place — [`app/ai/models.py`](../app/ai/models.py) —
because `chat.py` and `agents/router.py` both need the same providers in the same order,
and `chat.py` already imports `router.py`, so the shared code cannot live in either
without closing an import cycle.

Four models, fastest first, with automatic fallback between them:

| # | Model | Provider | Budget | Measured TTFT |
|---|---|---|---|---|
| 1 | `mistral-small-latest` | Mistral | light | **0.44s** min / 0.56s median |
| 2 | `openai/gpt-oss-20b` | Groq | **reasoning** | rate-limited (429) during benchmarking |
| 3 | `deepseek-v4-flash` | OpenRouter | light | 0.55s min / 0.56s median |
| 4 | `gemini-2.5-flash` | Google | light | 3.21s min / 3.30s median |

**Mistral leads on reliability, not raw speed.** It and DeepSeek are effectively tied on
first-token latency, but DeepSeek was observed *stalling part-way through a generation* —
unsurprising given OpenRouter serves it from several different upstream providers, so
stability varies run to run. It stays in the chain as a genuinely independent provider,
just off the critical path, and carries a 60s request timeout so a stall fails over to
Gemini instead of hanging the turn.

Gemini is last at roughly six times the time-to-first-token.

If `OPENROUTER_API_KEY` is absent the chain is built **without** DeepSeek — a three-model
chain, with nothing else shifted. The assembled chain is logged once at startup.

#### DeepSeek goes through OpenRouter, and two settings are load-bearing

(Applies wherever it sits in the chain — these were tuned while it was the primary and
still govern how it behaves as a fallback.)

The direct DeepSeek account has no balance — every completion returns **402 Insufficient
Balance** — so the model is reached through OpenRouter's OpenAI-compatible endpoint with
`ChatOpenAI`. `DEEPSEEK_API_KEY` is kept in config for the day that account is funded.

Two `extra_body` keys were measured, and **deleting either one defeats the purpose of the
model being there at all**:

| Setting | Without | With |
|---|---|---|
| `provider: {sort: "latency"}` | TTFT **2.90s** — default routing is not latency-aware | TTFT **0.55s** |
| `reasoning: {enabled: False}` | Measured at a 600-token cap: **328 tokens** lost to hidden reasoning, ~270 visible | **0** reasoning, all 600 visible — roughly double the answer |

OpenRouter spreads this model across upstream providers (StreamLake, NextBit and Baidu
were all observed), and it *does* reason by default despite being marketed otherwise.

### Answer length — two budgets

The chain mixes two kinds of model, so it carries two ceilings:

- **`LIGHT_MAX_TOKENS`** (default **2500**) — DeepSeek, Mistral, Gemini. These emit only
  visible text, so every token reaches the reader.
- **`REASONING_MAX_TOKENS`** (default **4000**) — `openai/gpt-oss-20b` alone, which spends
  hidden reasoning tokens from the same allowance. A measured deep-dive answer used
  **1375 reasoning tokens of 3617**, which is why it needs materially more than the
  light tier for a comparable answer.

A budget is a **hard ceiling**: the model stops the moment it is reached, the provider
returns `finish_reason: "length"`, and the answer ends mid-word.

The two numbers are paired deliberately: 4000 is the light tier's 2500 visible tokens plus
~1400 of headroom for hidden reasoning, so **both tiers yield a comparable answer**. Left
at 3000, the reasoning model would be the *weaker* one (~1600 visible) and answers would
quietly get shorter whenever the chain fell through to it.

A ceiling is not a target, which is what makes a generous one close to free — measured on
the light tier, a two-sentence answer spends 64 tokens and a 200-word one 212, whatever the
cap is set to. Only answers that genuinely need the room draw on it. Both are env-tunable
without a code change.

Because truncation is otherwise completely silent, `_stream_graph` watches each answer
chunk for `finish_reason == "length"`, emits a `truncated` SSE event before `done`, and
stores the message with `partial=True` so the notice survives a reload.

Voice keeps its own, much smaller budget (`VOICE_MAX_TOKENS`, default 120) applied to
every model in the chain: spoken answers should be short, and it is a real cost control on
the ElevenLabs free tier.

### The tool loop

There is no `AgentExecutor` and no LangGraph — the loop is written by hand in
`_run_tool_loop`, which makes the control flow completely visible:

1. Ask the tool-aware model what to do.
2. If it requested tools, run each one and append a `ToolMessage`.
3. Stream the final answer from the **plain** model, so it summarises the tool result
   instead of calling the tool again.

Because the loop is owned here, `ToolMessage` content is never yielded to the client —
raw webhook JSON cannot leak into the stream.

The tool itself (`app/tools/weather.py`) is an **async** tool built on a single shared
`httpx.AsyncClient`, created lazily and closed in the app's shutdown lifespan.

> **Teaching note — most of a "slow API call" was never the API.** The webhook's own work
> takes ~0.30s, but the tool span was measuring ~1.09s. The missing 0.7s was **connection
> setup**: the original implementation used `requests` with no session, so every single
> call paid a fresh DNS lookup, TCP handshake and TLS handshake before sending a byte.
> Reusing one client dropped the median to **0.36s**.
>
> Note what the fix was *not*. Making the tool `async` was not the win — LangChain already
> runs a sync tool in a thread pool when you `await` it, so it never blocked the event
> loop (measured: worst loop stall 19ms). Async is what makes a *shared connection pool*
> natural; the connection reuse is what saved the time. When something looks slow,
> measure the parts before rewriting the whole thing.

### One subtlety: content is not always a string

```python
def content_to_text(content) -> str:
    """Groq returns a str; Gemini can return a list of content parts."""
```

> **Teaching note.** Groq hands back `content` as a plain string, but Gemini — the
> *fallback* — can return a **list of parts**. Since the fallback fires exactly when Groq
> is rate-limiting, this shows up only under load. A raw list breaks two things at once:
> Starlette calls `.encode()` on whatever the stream yields, and the accumulator joins
> the tokens into one answer. Every yield site goes through `content_to_text()`.

---

## 8. Writing the answer back

The user message is written **before** the stream opens; the assistant message **after**
it finishes.

```mermaid
sequenceDiagram
    participant R as Route handler
    participant S as ChatService
    participant DB as MongoDB

    R->>DB: save user message (seq = n)
    R->>DB: load window (last k turns, excluding that message)
    R->>S: stream the answer
    S-->>R: tokens (also collected into a buffer)
    S->>DB: save assistant message (seq = n+1)
```

Writing the question first means it survives a crash mid-answer, and any failure — an
unknown `conversation_id`, a malformed body — surfaces as a proper JSON `4xx` instead of
an error string buried in the middle of a token stream.

### The abandoned-stream case

If the client closes the tab mid-answer, the partial text is still saved, flagged
`partial: true`. Getting this right required a split:

| Situation | How the write happens |
|---|---|
| Normal completion, or a handled error | **Awaited** — the next turn must not load history before this answer lands |
| Client disconnected (`GeneratorExit`) | Handed to an **independent task** |

> **Teaching note — why not just `await` in the `finally`?** Because it silently does
> nothing. An abandoned async generator is finalised by the event loop *after*
> `aclose()` has already returned, and an `await` at that point is cancelled — the write
> is issued and then dropped, with no error anywhere. Scheduling an independent task gets
> the write off the dying frame. The normal path stays awaited, because making *it*
> async too would let a fast follow-up question race a not-yet-saved answer.

---

## 9. Design decisions and why

| Decision | Reasoning |
|---|---|
| **Hand-written window memory, not `ConversationBufferWindowMemory`** | That class was **deleted in LangChain 1.0** — `langchain.memory` no longer imports. Its MongoDB backend was also blocking `pymongo`, which would stall the event loop inside these async endpoints. |
| **Not `RunnableWithMessageHistory` either** | The modern replacement wraps an **LCEL Runnable**. `ChatService` is a hand-written async generator with a manual tool loop, so adopting it would mean rewriting the streaming path to gain nothing. |
| **Window measured in turns, not messages** | Cutting between a question and its answer produces confused replies. |
| **UUID string ids with a `conv_` / `msg_` prefix** | Stored as plain `str`, so an id travels unchanged from Mongo → Python → JSON → the URL bar. The prefix makes a mistyped id fail fast (`422`, not a confusing `404`) and makes ids self-describing in logs. |
| **Router returns route *and* rewritten question** | One call, no extra latency, and retrieval finally works on follow-ups. |
| **Tool calls stored but never replayed** | Replaying a tool call without its result is a hard API error. |
| **Soft delete** | Messages stay on disk; an accidental click is not permanent. |
| **`X-Conversation-Id` response header** | The body is a raw token stream with nowhere to put metadata, and this keeps the existing response contract unchanged. |
| **State lives only in MongoDB** | The app layer is stateless, so restarts and multiple workers are safe. |
| **Voice uses raw PCM over a WebSocket, not WebRTC** | Push-to-talk on localhost has none of the problems WebRTC solves (echo, jitter, packet loss), and WebRTC hides the media path — the exact part worth teaching. See [voice-mode.md](voice-mode.md) §2. |
| **LLM clients built once per process, not per request** | Constructing `ChatGroq` + `ChatGoogleGenerativeAI` costs ~1.2 s **every time**, not just the first. Doing it per request put that in front of every answer. See [voice-mode.md](voice-mode.md) §8. |
| **Clerk for identity, this backend for sessions** | Clerk runs sign-in and OAuth so none of that is hand-rolled; the backend still issues its own cookies so it can enforce `is_banned` with no third-party round trip on the chat hot path — and so `/ws/voice` has a credential at all, since the browser `WebSocket` API cannot set headers. See §12. |
| **`user_id` deleted from request schemas, not defaulted** | It used to be a field on `ChatRequest` defaulting to `"default_user"`. Making it *dynamic* would not have helped: as long as the client sends it, the client chooses who to be. Removing the field is what makes spoofing impossible. |
| **Refresh tokens hashed with SHA-256, not bcrypt** | bcrypt is deliberately slow to resist brute force against *low-entropy human passwords*. A refresh token is 256 bits from `secrets.token_urlsafe(32)` — not guessable at any hash speed — so bcrypt would add ~100 ms to a hot endpoint and buy nothing. |
| **Refresh tokens rotate, and reuse burns the family** | A token presented twice is either a retry or a theft, and the server cannot tell which. Revoking the whole chain is the only response that is safe in the theft case. |
| **Clerk webhooks are mandatory, not optional** | Two session authorities drift. Without the webhook, a user revoked or deleted in Clerk keeps a working session here until their refresh token expires. |

---

## 10. File map

```
app/
├── main.py                  # FastAPI app, lifespan (LangSmith -> Mongo -> indexes -> ingest), CORS
├── core/
│   ├── config.py            # env settings: memory window knobs, Clerk keys, cookie policy
│   ├── ids.py               # conv_/msg_/usr_/prf_ UUID generation + validation
│   ├── security.py          # access-token signing, refresh hashing, cookie set/clear
│   └── tracing.py           # LangSmith helpers
├── api/
│   └── deps.py              # get_current_user / require_verified / require_admin / _ws
├── repositories/            # Mongo only, no policy
│   ├── user_repo.py         # users: upsert from Clerk, ban, token_version
│   ├── profile_repo.py      # profiles: editable-field allowlist
│   └── session_repo.py      # refresh tokens: rotation, reuse detection, revocation
├── services/                # policy and orchestration
│   ├── auth_service.py      # establish / refresh / logout / me
│   └── clerk_service.py     # token verification, user lookup, OAuth token read
├── db/
│   └── mongodb.py           # Motor client, collection accessors, index creation
├── memory/
│   ├── store.py             # persistence: conversations & messages (MongoDB only)
│   └── window.py            # window buffer: last k turns -> LangChain messages
├── ai/
│   ├── router.py            # QueryRouter -> RouteDecision(route, standalone_question)
│   ├── chat.py              # ChatService: 4 route branches, tool loop, persistence
│   ├── voice.py             # STT (Groq Whisper), TTS (ElevenLabs WS), sentence chunker
│   └── chain.py             # 3-stage LCEL chain: extract -> enrich -> format
├── rag/
│   ├── rag_pipeline.py      # ingest() / retrieve()
│   ├── vector_store.py      # Atlas $vectorSearch + $search + RRF
│   ├── embeddings.py        # Gemini embeddings, 768d
│   └── data_processor.py    # loaders + RecursiveCharacterTextSplitter
├── prompts/                 # router, RAG and per-route system prompts
├── routes/
│   ├── auth.py              # /auth/session, /refresh, /logout, /me, /me/profile
│   ├── webhooks.py          # POST /webhooks/clerk  (Svix-signed, not user-authenticated)
│   ├── chatbot.py           # POST /chatbot
│   ├── voice.py             # WS /ws/voice  (push-to-talk turn loop)
│   └── conversations.py     # conversation CRUD
├── schemas/
│   ├── chat.py              # Pydantic request/response models
│   └── auth.py              # profile update + user/profile output models
└── tools/
    └── weather.py           # @tool get_weather
```

> **Note.** `chain.py` is a self-contained LCEL example and is not reachable from any
> endpoint — it is not part of the request path described above.

---

## 11. Observability

LangSmith traces are grouped per conversation via `thread_id`, so an entire chat reads as
one thread in the dashboard rather than a pile of unrelated runs. Each run also carries
`route`, `standalone_question` and `history_messages` as metadata — enough to answer
"why did this question get routed there?" without reproducing it locally.

Tracing is optional and never load-bearing: it is enabled by `LANGSMITH_TRACING=true`, and
a LangSmith outage cannot break a chat request.

---

## 12. Authentication and sessions

### 12.1 Two systems, one job each

The setup uses Clerk *and* a session layer of its own. That is a deliberate split, not
redundancy:

| | Owns | Answers |
|---|---|---|
| **Clerk** | Identity | *Who is this person?* Runs sign-in, sign-up and OAuth. |
| **This backend** | Sessions | *May they act right now?* Issues cookies, enforces `is_banned` / `is_verified`. |
| **Clerk webhooks** | Sync | Keeps the two from drifting apart. |

Three reasons the local session layer earns its place rather than just forwarding Clerk
tokens on every call:

1. **`/ws/voice` needs it.** The browser `WebSocket` API cannot set an `Authorization`
   header. Cookies ride the handshake automatically. The alternative is a token in the
   query string, which lands in every access log.
2. **Ban checks stay local**, so no third-party round trip sits on the streaming path.
3. **Clerk is called once per sign-in**, not once per request.

> **The third row is not optional.** Two session authorities drift: revoke a user in
> Clerk and, with no webhook, their session here keeps working until the refresh token
> expires. `routes/webhooks.py` is what closes that, and building the session store
> without it is the failure mode of this design.

### 12.2 The flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant C as Clerk
    participant A as FastAPI

    B->>C: sign in (OAuth / password)
    C-->>B: Clerk session token
    B->>A: POST /auth/session  (Bearer <clerk token>)
    A->>C: verify_token (azp pinned to our origins)
    A->>A: upsert users + profiles, check is_banned
    A-->>B: Set-Cookie access (15m, /) + refresh (7d, /auth/refresh)
    Note over B,A: every later call, including WS, rides the cookies
    B->>A: GET /conversations   (cookie)
    A-->>B: only this user's threads
    B->>A: POST /auth/refresh   (on 401)
    A-->>B: rotated pair, or 401 + cleared cookies
```

The Clerk token is sent **exactly once**, at step 3. Everything after is cookie-based.

### 12.3 The two credentials

| | Access token | Refresh token |
|---|---|---|
| Form | Signed JWT | 256 random bits, opaque |
| Lifetime | `ACCESS_TOKEN_TTL_MIN` (15m) | `REFRESH_TOKEN_TTL_DAYS` (7d) |
| Stored | Not stored | SHA-256 hash only |
| Cookie path | `/` | **`/auth/refresh`** |
| Verified by | Signature, no DB read | Indexed hash lookup |

Both are `HttpOnly`, so no script on the page can read either — the reason this is not a
JWT in `localStorage`.

**Why the refresh cookie is path-scoped.** A cookie is only sent to paths it is scoped to,
so the long-lived credential is simply absent from every ordinary API call. It travels on
one endpoint instead of all of them.

**Why the access token is short.** It is verified without touching the database, so its
lifetime is also the worst case delay before a ban bites. The `ver` claim closes even
that: it is compared against the user's `token_version`, which a ban increments, so
existing tokens stop verifying immediately.

### 12.4 Rotation and theft detection

Refresh tokens are single-use. Each rotation marks the old document `replaced_by` and
issues a successor carrying the same `family_id`.

If an already-spent token is presented again, one of two things happened: the legitimate
holder retried, or someone stole it and the real holder already spent it. **The server
cannot distinguish these**, so it takes the only action that is safe in the theft case —
revoke the entire family and force a fresh sign-in.

The consume step is a conditional update (`replaced_by: None` in the filter), so the check
and the write are one atomic operation. Under eight concurrent rotations of the same
token, exactly one wins; a read-then-write would let several through.

> **Client consequence.** A page load fires several requests at once, so an expired token
> produces a burst of 401s. If each retried independently, the second would present an
> already-spent token and get the whole family burned — logging the user out for loading a
> busy page. `src/lib/api.ts` holds one in-flight refresh promise so the burst produces a
> single rotation.

### 12.5 Collections

```jsonc
// users — authorization state, written only by auth decisions
{
  "_id": "usr_...", "clerk_user_id": "user_2ab...", "email": "a@b.com",
  "email_verified": true,
  "is_verified": true,          // the app's own gate, separate from Clerk's
  "is_banned": false, "ban_reason": null, "banned_at": null,
  "role": "user",               // or "admin"
  "token_version": 0,           // bumped to invalidate live access tokens
  "created_at": "...", "updated_at": "...", "last_login_at": "..."
}

// profiles — user-editable, 1:1 with users
{
  "_id": "prf_...", "user_id": "usr_...",
  "username": "harish",         // from the OAuth provider on first sign-in
  "email": "a@b.com",
  "avatar_url": "", "mobile": "", "address": "", "bio": "",   // empty until edited
  "created_at": "...", "updated_at": "..."
}

// refresh_tokens — one per issued token
{
  "_id": "sess_...", "user_id": "usr_...",
  "token_hash": "<sha256>",     // the raw value is never stored
  "family_id": "fam_...",       // shared by every descendant of one login
  "issued_at": "...", "expires_at": "...",
  "revoked_at": null,
  "replaced_by": "<sha256>",    // non-null + presented again == reuse
  "user_agent": "...", "ip": "..."
}
```

**Indexes** — `users.clerk_user_id` unique, `users.email` unique, `profiles.user_id`
unique, `refresh_tokens.token_hash` unique, `(user_id, family_id)`, and a **TTL** on
`expires_at`.

The unique indexes do real work: they make first sign-in an upsert rather than a
read-then-write race, so two simultaneous first requests cannot create two accounts. The
TTL reaps only *expired* rows — revoked-but-unexpired tokens stay, which is exactly what
reuse detection needs in order to recognise a stolen token when it comes back.

**Why `users` and `profiles` are separate.** `profiles` is written by the person it
describes; `users` is written only by authorization decisions. Splitting them means a
profile update physically cannot reach `role` or `is_banned`. Two gates enforce it: the
Pydantic model in `schemas/auth.py` has no such fields, and `EDITABLE_FIELDS` in
`profile_repo.py` drops anything outside the allowlist.

### 12.6 Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /auth/session` | Clerk bearer | Exchange a Clerk token for cookies. Called once after sign-in. |
| `POST /auth/refresh` | Refresh cookie | Rotate. The only path that cookie is sent to. |
| `POST /auth/logout` | None | Revoke + clear. Unauthenticated on purpose — an expired token must still be able to log out. |
| `POST /auth/logout-all` | Session | End every session on every device. |
| `GET /auth/me` | Session | User + profile. |
| `PATCH /auth/me/profile` | Session | Update the editable fields. |
| `POST /webhooks/clerk` | Svix signature | Sync. Machine-to-machine, so no session. |

Handled webhook events: `user.created` / `user.updated` (upsert user + profile),
`user.deleted` (**ban rather than delete** — conversations still reference the id, and a
hard delete would orphan them), `session.revoked` (revoke tokens + bump `token_version`),
`user.banned` / `user.unbanned`. Anything else returns 204 so Clerk does not retry it
forever.

Signature verification runs against the **raw** request body, before parsing — re-serialised
JSON would not match the digest.

### 12.7 The vulnerability this replaced

Worth recording, because the shape of the bug is more instructive than the fix.

`user_id` used to be a Pydantic field on `ChatRequest` and `ConversationCreate`, defaulting
to `"default_user"`, and a query parameter on `GET /conversations`. Separately,
`get_conversation(conversation_id, user_id=None)` skipped the owner clause whenever
`user_id` was falsy — and every `/conversations/{id}` route called it without one.

Together that meant **any client could read, rename or delete any conversation by id**,
and could write messages as any user by editing one JSON field.

Both halves had to change:

- The `user_id` fields were **deleted** from the schemas. Making them dynamic would not
  have helped — as long as the client sends the value, the client picks who to be.
- `user_id` became a **required** argument on the store methods, so the filter cannot be
  omitted again by a future caller. The owner clause is also repeated inside the rename
  and delete updates rather than left to the route's earlier existence check.

Not-yours returns **404, not 403**, so the API never confirms that an id exists to someone
who cannot see it.

### 12.8 Configuration

```
CLERK_SECRET_KEY=sk_...         CLERK_PUBLISHABLE_KEY=pk_...
CLERK_WEBHOOK_SECRET=whsec_...  # the signing secret, not a URL
JWT_SECRET=<32 random bytes, hex>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_TTL_MIN=15         REFRESH_TOKEN_TTL_DAYS=7
COOKIE_SECURE=false             # true in production
COOKIE_SAMESITE=lax             # "none" if the frontend is on another domain
COOKIE_DOMAIN=
```

`JWT_SECRET` must be high-entropy. HS256 signed with a short human-typed string can be
brute-forced offline from any single issued token, which would let an attacker mint
admin tokens at will.

`COOKIE_SAMESITE=lax` works in development because `localhost:3000 → localhost:8000` is
same-site — a port is not part of the site. A production deployment on two different
domains needs `none`, which browsers only honour together with `Secure`, which requires
HTTPS on both ends.

Startup logs a warning when `CLERK_SECRET_KEY` or `JWT_SECRET` is missing, because the
alternative symptom is a 401 at sign-in with no explanation.

### 12.9 Frontend contract

- `proxy.ts` (Next 16 renamed `middleware.ts` → `proxy.ts`) gates every route except
  `/`, `/sign-in`, `/sign-up`. **With a `src/` directory it must live at `src/proxy.ts`** —
  at the project root it compiles to an empty manifest and protects nothing, silently.
- `axios` and the streaming `fetch` both set credentials (`withCredentials` /
  `credentials: "include"`), or the browser omits the cookie cross-origin.
- `SessionProvider` performs the `/auth/session` exchange and blocks the app until it
  resolves, so no child component fires a request that races the cookie.
- Logging out calls **both** `POST /auth/logout` and Clerk's `signOut()`. Either alone
  leaves a live session on the other side.
- `/` is public and interactive while signed out: submitting a prompt stashes it in
  `sessionStorage`, redirects to sign-in, and replays it once the *backend* session
  exists — waiting on Clerk alone would race the cookie.
