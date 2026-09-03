# LangGraph Agentic RAG & Multi-Modal AI System

A production-grade, asynchronous Agentic Retrieval-Augmented Generation (RAG) and multi-modal AI backend built with **FastAPI**, **LangChain**, **LangGraph**, and **MongoDB Atlas**.

This system features dynamic query routing via a stateful **LangGraph StateGraph** (with built-in `ToolNode`, conditional edges, LRU-cached graph compilation, and `MemorySaver` checkpointer), hybrid vector & keyword retrieval fused via **Reciprocal Rank Fusion (RRF)**, automatic fallback across dual LLM providers (**Mistral AI mistral-small-latest** + **Google Gemini 2.5 Flash**), tool calling with external webhooks, 3-stage LCEL document transformation chains, multi-modal capabilities (voice STT/TTS & image generation), and **Clerk-backed authentication with rotating server-issued sessions**.

---

## 📖 Architecture & Documentation Index

Comprehensive technical documentation is available in the [`docs/`](file:///c:/Users/haris/Documents/Projects/Langchain-RAG/docs) directory:

- 🏗️ **[System Architecture](docs/architecture.md)** — Complete component breakdown, LangGraph StateGraph topology, and system-wide Mermaid diagram. **§12 covers authentication**: the Clerk/session split, token rotation, and ownership model.
- 🎙️ **[Voice Mode](docs/voice-mode.md)** — The spoken path end to end: WebSocket + PCM transport, streaming TTS, latency optimizations, and free-tier limits.
- 🔄 **[RAG & App Workflows](docs/workflow.md)** — Detailed sequence flows for startup ingestion, hybrid retrieval, LangGraph node execution, tool loops, and token streaming.
- ✂️ **[Chunking Strategy](docs/chunking.md)** — Recursive text splitting, parameter choices, and incremental deduplication logic.
- 🚀 **[Advanced RAG Concepts](docs/advanced-rag.md)** — Hybrid search, query expansion, re-ranking, and context compression.
- 🤖 **[Agentic RAG Concepts](docs/agentic-rag.md)** — Planner loops, state graph routing, tool selection, reflection, and self-correction.
- 🕸️ **[Graph RAG Concepts](docs/graph-rag.md)** — Knowledge graphs, entity-relation extraction, and graph traversal.
- 🧩 **[Modular RAG Concepts](docs/modular-rag.md)** — Decoupled modules and flexible pipeline architectures.

---

## ⚡ Multi-Agent Supervisor Workflow

The core chat pipeline (`POST /chatbot`) is a **supervisor graph orchestrating specialist agents**, each a LangGraph subgraph with its own nodes. The entry point is [`app/ai/agents/graph.py`](file:///c:/Users/haris/Documents/Projects/Langgraph/Naive-RAG-LangChain/app/ai/agents/graph.py); the supervisor itself lives in [`supervisor/supervisor_graph.py`](file:///c:/Users/haris/Documents/Projects/Langgraph/Naive-RAG-LangChain/app/ai/agents/supervisor/supervisor_graph.py).

```
app/ai/agents/
├── state.py                  # ChatState (parent) + RagState + ToolState
├── router.py                 # QueryRouter
├── graph.py                  # public entry point
├── supervisor/               # orchestration + the DIRECT route
│   ├── supervisor_graph.py
│   └── nodes/{route_query,generate}.py
├── rag_agent/                # owns retrieval
│   ├── rag_graph.py
│   └── nodes/{retrieve,generate}.py
└── tool_agent/               # owns tools + the approval gate
    ├── tool_graph.py
    ├── toolbelt.py
    └── nodes/{call_llm_with_tools,approve_tools}.py
```

```mermaid
graph TD
    START["__start__"] --> route_query["route_query<br/>(QueryRouter classification)"]

    route_query --> pick_route{"pick_route"}

    pick_route -->|"RAG / BOTH"| rag_agent
    pick_route -->|"TOOL"| tool_agent
    pick_route -->|"DIRECT"| generate["generate<br/>(plain LLM completion)"]

    subgraph rag_agent["🔍 rag_agent (subgraph)"]
        rag_retrieve["retrieve<br/>(hybrid vector search)"] --> rag_answer{"route == BOTH?"}
        rag_answer -->|"no"| rag_generate["generate<br/>(RAG-grounded answer)"]
    end

    subgraph tool_agent["🔧 tool_agent (subgraph)"]
        call_llm["call_llm_with_tools"] --> should_continue{"tool_calls?"}
        should_continue -->|"yes"| approve["approve_tools<br/>(HITL interrupt gate)"]
        approve -->|"accept"| tools["tools<br/>(LangGraph ToolNode)"]
        tools --> call_llm
    end

    rag_agent -->|"route == BOTH"| tool_agent
    rag_agent -->|"route == RAG"| END["__end__"]
    tool_agent -->|"tool refused"| generate
    tool_agent -->|"answered"| END
    generate --> END
```

### Components & Design

1. **State Management (`agents/state.py`)**:
   - `ChatState` is the supervisor's state; `RagState` and `ToolState` are **subsets** of it, which is what lets each compiled subgraph be dropped into the parent as a plain node with no adapter code.
   - `messages`: annotated with the `add_messages` reducer in **every** schema — without it a subgraph would replace conversation history instead of appending to it.
   - `route`: active route (`"RAG"`, `"TOOL"`, `"BOTH"`, `"DIRECT"`); `search_query`: the router's standalone rewrite; `context`: retrieved chunks; `denied_tools`: tools the user refused this turn.
2. **Agents & Responsibility**:
   - **Supervisor** — routes the turn, owns the `DIRECT` answer, and sequences the two-agent `BOTH` route.
   - **`rag_agent`** — MongoDB Atlas hybrid search (Vector + BM25 + RRF) and RAG-grounded answers. On `BOTH` it retrieves and stops, handing the context to the tool agent.
   - **`tool_agent`** — the toolbelt, the `ToolNode`, and the `approve_tools` human-in-the-loop gate. On refusal it ends with `denied_tools` set, and the supervisor takes over (a subgraph cannot jump to a parent node).
3. **Conditional Edges**:
   - **`pick_route`**: inspects `state["route"]` to dispatch to `rag_agent`, `tool_agent`, or `generate`.
   - **`after_rag`** / **`after_tools`**: continue the `BOTH` chain, and catch the refused-tool fallback.
   - **`should_continue`** (inside `tool_agent`): inspects `AIMessage.tool_calls` to route into the approval gate or terminate.
4. **Adding an agent**: a new folder with its own compiled subgraph, one `add_node`, and one branch in `pick_route`.
5. **Memory & Checkpointing**:
   - Uses `MemorySaver` checkpointer compiled with thread isolation (`thread_id = conversation_id`).
   - Eliminates manual window management for text chat while MongoDB `conversation_store` maintains persistence for UI history.
5. **Latency & Graph Caching**:
   - Graph compilation is optimized with `@lru_cache(maxsize=4)` (`_compile_graph`).
   - Stateless LLM instances are cached via `@lru_cache(maxsize=4)` (`_build_models`).

---

## ✨ Key Features & Capabilities

### 1. Dynamic Agent Routing via the Supervisor
Before any retrieval or LLM generation, an ultra-fast classification node (`route_query`) categorizes incoming prompts into one of four execution routes, and the supervisor dispatches each to the agent that owns it:
- **`RAG`** → `rag_agent`: hybrid search over uploaded knowledge base documents in MongoDB Atlas, then answers from them (no tools).
- **`TOOL`** → `tool_agent`: executes external tools (e.g. `get_weather` webhook) without performing document retrieval.
- **`BOTH`** → `rag_agent` **then** `tool_agent`: retrieves context first to resolve entity/location details, then executes the tool call with that context in the prompt. The only route that spans two agents, sequenced by the supervisor.
- **`DIRECT`** → the supervisor's own `generate`: answers from the LLM's parametric knowledge for general conversation.

### 2. MongoDB Atlas Hybrid Search & RRF
- **Vector Search (`$vectorSearch`)**: Dense semantic search using Google's `models/gemini-embedding-001` (768 dimensions) with Cosine similarity.
- **Keyword Search (`$search`)**: Sparse text search leveraging Atlas Search (BM25 algorithm).
- **Reciprocal Rank Fusion (RRF)**: Combines ranked vector and keyword candidate lists into a single deduplicated, highly accurate context payload.
- **Automatic Index Management**: Automatically provisions `vector_index` and `keyword_index` on MongoDB Atlas via PyMongo `SearchIndexModel` on boot.

### 3. Dual LLM High Availability Engine
- **Primary LLM**: Mistral AI `mistral-small-latest` (fast, high intelligence).
- **Fallback LLM**: Google Gemini `gemini-2.5-flash` (high quality, high context).
- Integrated seamlessly using LangChain's `with_fallbacks()` runnable wrapper to protect against rate limits and outage events.

### 4. Incremental Ingestion & Zero-Cost Boot
- On server startup (`app/main.py` lifespan), the system scans the `uploads/` directory.
- It checks existing document chunk IDs in MongoDB Atlas before embedding.
- **Only new or updated chunks** are passed to the embedding API, guaranteeing zero redundant embedding cost and fast boot times.

### 5. Multi-Modal & Specialized AI Modules
- **3-Stage LCEL Chain (`app/ai/chain.py`)**: Concept extraction (JSON parser) → Concept enrichment → Markdown report generation.
- **Image Generation (`app/ai/image.py`)**: Generates images using LiteLLM and Gemini Imagen 3 (`gemini/Gemini 2.5 Flash Preview Image`), returning PIL image instances.
- **Voice Engine (`app/ai/voice.py`)**: Speech-to-Text via Groq Whisper Turbo (`whisper-large-v3-turbo`) and streaming Text-to-Speech via the ElevenLabs `stream-input` WebSocket (`eleven_flash_v2_5`, `pcm_24000`).

### 6. Authentication & Per-User Isolation
Clerk handles identity; this backend owns the session and every authorization decision — see **[docs/architecture.md](docs/architecture.md) §12**.
- **Split of responsibility**: Clerk runs sign-in/OAuth, the backend issues its own `HttpOnly` cookies so it can enforce `is_banned` with no third-party round trip on the streaming path — and so `/ws/voice` has a credential at all, since the browser `WebSocket` API cannot set headers.
- **Rotating refresh tokens**: single-use, stored only as a SHA-256 hash, grouped by `family_id`. Presenting a spent token revokes the whole family, on the assumption it may have been stolen.
- **Fast ban propagation**: a `token_version` claim is checked against the user record, so a ban invalidates live access tokens immediately rather than after their 15-minute TTL.
- **Clerk webhooks** (`POST /webhooks/clerk`, Svix-signed) keep the two systems in sync — without them a user revoked in Clerk would keep a working session here.
- **Ownership enforced in the data layer**: `user_id` is a required argument on every conversation query, and is resolved from the cookie — the request body carries no identity at all.

### 7. Real-Time Voice Mode
Push-to-talk speech in, synthesised speech out, over a single WebSocket at **`/ws/voice`** — see **[docs/voice-mode.md](docs/voice-mode.md)**.
- **Transport**: raw PCM16 both directions (16 kHz up, 24 kHz down). No WebRTC, no Opus, no ffmpeg — and **no new dependencies** on either side.
- **Streaming TTS**: LLM tokens are regrouped at sentence boundaries and pushed into ElevenLabs while the model is still writing, so audio begins after the *first* sentence rather than the last.
- **Shared brain**: the spoken path reuses `ChatService` — routing, retrieval, tools and memory all apply, with a voice-specific prompt for short spoken answers.
- **Measured**: ~0.2 s to on-screen transcript, ~1.3–1.8 s median to the first spoken word.

---

## 📁 Repository Structure

```
Langchain-RAG/
├── app/
│   ├── ai/
│   │   ├── chain.py          # 3-Stage LCEL Concept Extraction & Enrichment Pipeline
│   │   ├── chat.py           # LangGraph streaming wrapper (stream_chat) & legacy ChatService for voice
│   │   ├── graph.py          # LangGraph StateGraph (Nodes, Edges, ToolNode & MemorySaver)
│   │   ├── image.py          # Image generation via LiteLLM (Gemini Imagen 3)
│   │   ├── router.py         # Lightweight QueryRouter classification chain
│   │   └── voice.py          # STT (Groq Whisper), streaming TTS (ElevenLabs WS), sentence chunker
│   ├── api/
│   │   └── deps.py           # Auth dependencies: get_current_user, require_verified/admin, WS variant
│   ├── core/
│   │   ├── config.py         # Centralized Settings & environment variable configuration
│   │   ├── ids.py            # Prefixed UUID generation & validation (conv_/msg_/usr_/prf_)
│   │   └── security.py       # Access-token signing, refresh hashing, cookie policy
│   ├── repositories/         # Mongo-only data access for auth (no policy)
│   │   ├── user_repo.py      # users: Clerk upsert, ban, token_version
│   │   ├── profile_repo.py   # profiles: editable-field allowlist
│   │   └── session_repo.py   # refresh tokens: rotation, reuse detection, revocation
│   ├── services/             # Policy & orchestration
│   │   ├── auth_service.py   # establish / refresh / logout / me
│   │   └── clerk_service.py  # Clerk token verification, user lookup, OAuth token read
│   ├── db/
│   │   └── mongodb.py        # Motor async MongoDB client & connection lifecycle
│   ├── prompts/
│   │   ├── chain_prompts.py  # Prompts for 3-stage chain pipeline
│   │   ├── rag_prompt.py    # System prompts & context formatters for RAG
│   │   ├── router_prompt.py # System prompts for Router & route-specific models
│   │   └── voice_prompt.py  # Spoken-answer shaping: 2-3 sentences, no markdown
│   ├── rag/
│   │   ├── data_processor.py # File loader (.pdf, .txt, .md) & RecursiveCharacterTextSplitter
│   │   ├── embeddings.py     # Gemini GoogleGenerativeAIEmbeddings (768d)
│   │   ├── rag_pipeline.py   # Ingestion & retrieval orchestrator singleton
│   │   └── vector_store.py   # MongoDB Atlas Vector/Keyword search & RRF implementation
│   ├── memory/
│   │   ├── store.py          # MongoDB persistence for conversations & messages
│   │   └── window.py         # Window buffer: last k turns -> LangChain messages
│   ├── schemas/
│   │   ├── chat.py           # Pydantic request/response models (no user_id — see docs §12.7)
│   │   └── auth.py           # Profile update + user/profile output models
│   ├── routes/
│   │   ├── auth.py           # /auth/session, /refresh, /logout, /me, /me/profile
│   │   ├── webhooks.py       # POST /webhooks/clerk — Svix-signed, machine-to-machine
│   │   ├── chatbot.py        # FastAPI APIRouter streaming endpoint (/chatbot)
│   │   ├── voice.py          # WebSocket /ws/voice — push-to-talk turn loop
│   │   └── conversations.py  # Conversation CRUD (new chat, list, transcript, rename, delete)
│   └── main.py               # FastAPI app initialization, lifespan handler & health routes
├── docs/                     # Technical architecture, workflow, and RAG guides
├── uploads/                  # Input directory for knowledge base documents (.pdf, .txt, .md)
├── .env                      # Environment secrets (API keys & MongoDB connection URI)
├── readme.md                 # Project Overview & System Documentation
├── requirements.txt          # Python dependencies
└── server.py                 # Server startup script (Uvicorn runner)
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root with the following keys:

```env
# LLM & Embedding API Keys
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
FLUX_AI=optional_flux_key

# MongoDB Atlas Configuration
MONGO_URL=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
DB_NAME=rag_db

# External Tools & Webhooks
WEATHER_WEBHOOK_URL=https://your-webhook-endpoint.com

# Authentication — Clerk identity + backend-issued sessions (see docs/architecture.md §12)
CLERK_SECRET_KEY=sk_test_...
CLERK_PUBLISHABLE_KEY=pk_test_...
# The Svix SIGNING SECRET from the Clerk dashboard webhook page — not a URL.
CLERK_WEBHOOK_SECRET=whsec_...
# Must be high entropy: HS256 signed with a short human-typed string can be
# brute-forced offline from any single issued token.
#   python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET=<64 hex chars>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_TTL_MIN=15        # also the worst-case delay before a ban bites
REFRESH_TOKEN_TTL_DAYS=7
COOKIE_SECURE=false            # true in production (required for SameSite=none)
COOKIE_SAMESITE=lax            # "none" if frontend and backend are on different domains
COOKIE_DOMAIN=

# Voice mode (ElevenLabs). Only ELEVEN_API is required; the rest have defaults.
ELEVEN_API=your_elevenlabs_api_key
# Free accounts CANNOT use library voices over the API. Verified working on free:
#   Sarah EXAVITQu4vr4xnSDxMaL (default) | Adam pNInz6obpgDQGcFmaJgB
#   Antoni ErXwobaYiN019PkySvjV | Arnold VR6AewLTigWG4xSOukaG
#   George JBFqnCBsd6RMkjVDRZzb | Jessica cgSgspJ2msm6clMCkdW9
#   Daniel onwK4e9ZLuTAKqWW03F9
ELEVEN_VOICE_ID=EXAVITQu4vr4xnSDxMaL
ELEVEN_MODEL_ID=eleven_flash_v2_5   # 0.5 credits/char and ~75ms to first byte
ELEVEN_OUTPUT_FORMAT=pcm_24000      # pcm_44100 would require a Pro subscription
MIC_SAMPLE_RATE=16000               # what the browser worklet sends
TTS_SAMPLE_RATE=24000               # must match ELEVEN_OUTPUT_FORMAT
VOICE_MAX_TOKENS=120                # short spoken answers; also a credit control
TTS_CACHE_ENABLED=true              # replay identical phrases free while developing

# LangSmith Tracing (optional — tracing is off unless LANGSMITH_TRACING=true)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your_project_name
# Required when the API key is org-scoped rather than workspace-scoped
LANGSMITH_WORKSPACE_ID=your_workspace_id

# Conversation memory window (all optional — defaults shown)
WINDOW_K=4                    # turns replayed to the model
WINDOW_K_LARGE=5              # turns once a thread gets long
LARGE_HISTORY_THRESHOLD=100   # turn count at which the wider window kicks in
```

---

## 🚀 Running the Server

1. **Activate Virtual Environment**:
   ```bash
   venv\Scripts\activate   # Windows
   source venv/bin/activate # Linux/macOS
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server**:
   ```bash
   python server.py
   ```
   Or directly via Uvicorn:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Verify Health Check**:
   Open `http://127.0.0.1:8000/health` in your browser.

---

## 📡 API Endpoint Reference

> **Every endpoint below except `/health`, `/` and `/webhooks/clerk` requires a session.**
> Identity comes from the `access_token` cookie; a bearer header is also accepted so
> `curl` and `/docs` stay usable. Unauthenticated requests get `401`, banned users `403`.
> No request body carries a `user_id` — see [docs/architecture.md](docs/architecture.md) §12.7.

### Authentication

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/auth/session` | Clerk bearer token | Exchange a Clerk token for session cookies. Called once after sign-in. |
| `POST` | `/auth/refresh` | Refresh cookie | Rotate the pair. The only path the refresh cookie is scoped to. |
| `POST` | `/auth/logout` | none | Revoke + clear cookies. Unauthenticated so an expired session can still sign out. |
| `POST` | `/auth/logout-all` | session | End every session on every device. |
| `GET` | `/auth/me` | session | Current user + profile. |
| `PATCH` | `/auth/me/profile` | session | Update `username`, `avatar_url`, `mobile`, `address`, `bio`. |
| `POST` | `/webhooks/clerk` | Svix signature | Keeps Clerk and the local session store in sync. |

```bash
# Establish a session (the frontend does this automatically after Clerk sign-in)
curl -X POST "http://127.0.0.1:8000/auth/session" \
     -H "Authorization: Bearer <clerk_session_token>" -c cookies.txt

curl "http://127.0.0.1:8000/auth/me" -b cookies.txt
```

### `POST /chatbot`
Streams the answer back as plain text tokens. **Requires a session.**

- **JSON Body**:
  - `user_prompt` (`str`, required): The question or command for the assistant.
  - `conversation_id` (`str`, optional): Continue an existing thread. Omit it to start a new one.
  - There is deliberately **no `user_id`** — the owner is taken from the session cookie, and
    a `user_id` in the body is ignored. Continuing a thread you do not own returns `404`.

- **Response Header**:
  - `X-Conversation-Id`: The thread this message landed in — read it on the first
    message to learn the id of a newly created conversation.

- **Example Request**:
  ```bash
  # New chat (id comes back in the header)
  curl -N -D - -X POST "http://127.0.0.1:8000/chatbot" -b cookies.txt \
       -H "Content-Type: application/json" \
       -d '{"user_prompt": "Who is Harish?"}'

  # Continue that chat — follow-ups can rely on the previous turns
  curl -N -X POST "http://127.0.0.1:8000/chatbot" -b cookies.txt \
       -H "Content-Type: application/json" \
       -d '{"conversation_id": "conv_...", "user_prompt": "what are his skills?"}'
  ```

### `WS /ws/voice`
Push-to-talk voice turns. Binary frames are always audio, text frames are always JSON.
Full protocol table and rationale in **[docs/voice-mode.md](docs/voice-mode.md)** §3.

| Direction | Frame | Meaning |
|---|---|---|
| → server | binary | PCM16LE, 16 kHz, mono, ~40 ms per frame |
| → server | `{"type":"start","conversation_id":…}` | button pressed |
| → server | `{"type":"end"}` \| `{"type":"cancel"}` | run the turn \| discard it |
| ← client | `{"type":"transcript"\|"token"\|"done"\|"error", …}` | control + captions |
| ← client | binary | PCM16LE, 24 kHz, mono — synthesised speech |

> The `Origin` header is checked inside the endpoint against an allowlist, because
> `CORSMiddleware` does **not** apply to WebSocket handshakes. That proves where the page
> came from, not who is using it — authentication is a separate check against the session
> cookie, which the browser attaches to the handshake automatically. An unauthenticated
> socket is closed with code `1008`.

### Conversation management

All scoped to the caller. Another user's thread returns **`404`, not `403`** — the API
never confirms an id exists to someone who cannot see it.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/conversations` | Start a new chat explicitly |
| `GET` | `/conversations?limit=&skip=` | List **your** threads, most recently active first |
| `GET` | `/conversations/{id}?limit=&before_seq=` | Full transcript, oldest message first |
| `PATCH` | `/conversations/{id}` | Rename (`{"title": "..."}`) |
| `DELETE` | `/conversations/{id}` | Soft delete |

---

## 🧠 Conversation Memory

Chat history is persisted in MongoDB across two collections — `conversations`
(one per thread: title, owner, counters) and `messages` (one per turn, ordered
by `seq`).

On each request the last **k turns** are replayed to the model — a window
buffer, equivalent to the old `ConversationBufferWindowMemory(k=...)`, but
implemented natively over the async Motor client. (LangChain 1.0 removed
`langchain.memory` entirely, and its MongoDB backend was blocking pymongo,
which would stall the event loop in these async endpoints.) `k` defaults to 4
and widens to 5 once a thread passes 100 turns; all three values are tunable
via `WINDOW_K`, `WINDOW_K_LARGE` and `LARGE_HISTORY_THRESHOLD` in `.env`.

The query router is history-aware: it returns both the route **and** a
rewritten, self-contained version of the question. That rewrite is what gets
embedded for retrieval, so a follow-up like *"and where did he study?"* — which
on its own contains nothing searchable — still finds the right chunks.

Tool calls are recorded as message metadata but never replayed as
`AIMessage.tool_calls` / `ToolMessage` pairs; a tool call replayed without its
result is rejected outright by the Groq API.

---

## 🛡️ License & Contributing

Built with modern async Python standards and open for developer customization. Feel free to extend routers, add new custom tools, or swap vector backends!
