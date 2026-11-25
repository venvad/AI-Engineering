# Planned Improvements

## 1. Multi-Turn Conversation Memory for RAG

Level 1 (client-sent chat history passed to generation) is implemented. Levels 2 and 3 remain for follow-up retrieval quality and server-side persistence.

### Level 1: Pass Chat History to Generation (Implemented)
- **Schema (`apps/api/src/api/api/models.py`)**: Add `history: list[ChatMessage] = Field(default_factory=list)` to `RAGRequest`.
- **UI (`apps/chatbot_ui/src/chatbot_ui/app.py`)**: Send prior turns (`st.session_state.messages[1:-1]`, capped to the last ~6–10 turns) in the `/api/rag` payload.
- **Prompting (`apps/api/src/api/agents/retrieval_generation.py`)**: In `build_prompt`, splice `history` between `SYSTEM_PROMPT` and the current user turn (`<available_products>` + `<user_question>`).
- **Pros**: Zero extra LLM calls or latency; handles follow-ups about already-recommended items (*"Compare those two"*).
- **Limitation**: `retrieve_items(question)` still embeds only the raw follow-up text, so context-dependent searches (*"Does it come in black?"*, *"Any cheaper ones?"*) retrieve poor matches from Qdrant.

### Level 2: Query Rewriting + Chat History (Recommended Near-Term)
Add a lightweight contextualization step before vector search whenever `history` is non-empty:

1. **`rewrite_query(question: str, history: list[ChatMessage]) -> str`** (decorated with `@traceable(name="rewrite_query")`):
   - Skipped when `history` is empty (zero overhead on turn 1).
   - Uses a fast/cheap model to turn a follow-up + prior turns into a standalone search query (e.g., history about *"USB desk fans"* + *"Any cheaper ones?"* → *"affordable USB desk fan"*).
2. **Retrieval**: Pass the rewritten query to `retrieve_items(search_query, k=top_k)`.
3. **Generation**: Pass `history` + newly retrieved `<available_products>` + original `question` to `generate_answer(...)`.
4. **Observability**: Log `rewritten_query` on the `RAGResult` / LangSmith trace so retrieval quality can be evaluated independently from generation.

**Edge cases to handle**:
- Cap `history` length (e.g., last 6–10 messages) to bound token cost and latency.
- Enforce alternating `user` / `assistant` turns starting with `user` (required by Anthropic and Google Gemini).
- Keep plain-text `ChatMessage` history (strip any provider-specific artifacts if switching providers mid-chat).

### Level 3: Server-Side Session Memory (Long-Term)
- Replace client-sent `history` with a `session_id` / `thread_id` on `RAGRequest`.
- Persist conversation state server-side (e.g., Postgres or Redis, or a LangGraph checkpointer).
- Optionally add agentic routing so the model decides *whether* a follow-up needs a new Qdrant search or can be answered directly from prior context.


---

## 2. FastAPI Route & API Hardening

### URL Design & Versioning
- [x] **Version the API prefix**: Mount `api_router` at `/api/v1` in `apps/api/src/api/app.py` (`POST /api/v1/rag`) and update `API_URL` in `apps/chatbot_ui/src/chatbot_ui/core/config.py`.
- [ ] **Operational endpoints (root-level)**: Add `GET /health` (liveness) and `GET /ready` (verifies Qdrant connectivity) outside `/api/v1`.

### Router & Logging Structure
- [ ] **Remove duplicate `logging.basicConfig()`** from `apps/api/src/api/api/endpoints.py` — configure logging once in `app.py` and keep only `logger = logging.getLogger(__name__)` in submodules.
- [ ] **Split routers by domain** (once >1 feature exists): Move routes into `apps/api/src/api/api/routes/` (`rag.py`, `health.py`, etc.) aggregated by `api_router`.

### Input Validation (`apps/api/src/api/api/models.py`)
- [ ] **Reject blank queries**: Add `min_length=1` and a whitespace-stripping validator on `RAGRequest.query` so empty requests return `422` before calling OpenAI/Qdrant.
- [ ] **Optional bounded `top_k`**: Expose `top_k: int = Field(default=5, ge=1, le=50)` on `RAGRequest` if callers should control retrieval depth.
- [ ] **Remove or fix unused schemas**: Clean up legacy `ChatRequest` (including `models_name` typo) / `ChatResponse` if no direct `/chat` endpoint is kept.

### Error Handling & Status Codes
- [ ] **Map upstream errors accurately**: Return `504` on upstream timeouts (`APITimeoutError` / Qdrant timeout), `502` on upstream LLM/Qdrant errors, and `500` only for unexpected internal bugs.
- [ ] **Stop leaking raw exception strings**: Replace `detail=f"Request failed: {e}"` with a safe client message + `request_id`, while logging full tracebacks (`request.method` and `request.url.path`) server-side.
- [ ] **Centralize exception handlers**: Register `@app.exception_handler(...)` on `app` instead of duplicating `try/except` in every route (and always re-raise `HTTPException` before catching `Exception`).

### Observability, Docs & Security
- [ ] **Fix CORS spec violation** (`apps/api/src/api/app.py`): `allow_origins=["*"]` with `allow_credentials=True` is invalid per CORS — either set `allow_credentials=False` or list explicit origins (e.g., `["http://localhost:8501"]`).
- [ ] **Request ID / correlation middleware**: Generate a UUID per request, return it in an `X-Request-ID` response header, include it in logs/errors, and attach it to LangSmith run metadata.
- [ ] **Enrich OpenAPI `/docs` metadata**: Add `summary`, `description`, `status_code`, and `responses={502: ..., 504: ...}` to route decorators.
- [ ] **Async migration (future, if needed for concurrency)**: Only switch route handlers from `def` to `async def` alongside switching to `AsyncOpenAI`, `AsyncAnthropic`, and `AsyncQdrantClient`.

---

## 3. Async Migration Notes (`def` vs `async def`)

### Why `async def` with sync SDKs makes performance worse

FastAPI runs on a single-threaded `asyncio` event loop:

| Route type | Where FastAPI runs it | What happens during a 3s OpenAI call |
|---|---|---|
| `def chat(...)` *(current)* | Offloaded to a worker threadpool (default: 40 threads) | Other requests keep running on the remaining 39 threads |
| `async def chat(...)` + sync SDKs | Directly on the main event loop | Freezes the entire server for 3s — even `/health` waits |
| `async def chat(...)` + async SDKs | Directly on the main event loop, yielding on `await` | Hundreds/thousands of concurrent requests share 1 thread cleanly |

Changing `def` → `async def` without migrating the underlying SDKs is a classic FastAPI performance trap.

### When `async def` is worth doing

1. **Streaming answers**: `StreamingResponse` / SSE yielding tokens to Streamlit as they arrive.
2. **Parallel work inside one request**: e.g. `asyncio.gather(...)` to call multiple tools, query Qdrant + a SQL DB simultaneously, or run reranking in parallel.
3. **High concurrency**: LLM calls are slow (2–15s), so 40 threads saturate quickly under real traffic, whereas `asyncio` handles hundreds of waiting sockets effortlessly.

### What a full async migration touches

Every I/O layer in the chain must switch together:
- `OpenAI` → `AsyncOpenAI` (`await client.embeddings.create(...)`, `await client.chat.completions.create(...)`)
- `Anthropic` → `AsyncAnthropic` (`await client.messages.create(...)`)
- `genai.Client` → `await client.aio.models.generate_content(...)`
- `QdrantClient` → `AsyncQdrantClient` (`await client.query_points(...)`)
- `get_embedding`, `query_qdrant`, `retrieve_items`, `generate_answer`, `run_llm`, `rag_pipeline`, and `chat` all become `async def` with `await`.
- LangSmith (`@traceable` and `wrap_*`) supports async clients out of the box.
