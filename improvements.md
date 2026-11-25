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
- [x] **Remove duplicate `logging.basicConfig()`** from `apps/api/src/api/api/endpoints.py` — configure logging once in `app.py` and keep only `logger = logging.getLogger(__name__)` in submodules.
- [ ] **Split routers by domain** (once >1 feature exists): Move routes into `apps/api/src/api/api/routes/` (`rag.py`, `health.py`, etc.) aggregated by `api_router`.

### Input Validation (`apps/api/src/api/api/models.py`)
- [x] **Reject blank queries**: `RAGRequest` uses `ConfigDict(str_strip_whitespace=True)` + `min_length=1` on `query` and rejects `Role.SYSTEM` in `history`; `build_prompt` slices to the last `MAX_HISTORY_MESSAGES` (10).
- [ ] **Optional bounded `top_k`**: Expose `top_k: int = Field(default=5, ge=1, le=50)` on `RAGRequest` if callers should control retrieval depth.
- [x] **Remove unused schemas**: Removed legacy `ChatRequest` (with `models_name` typo) / `ChatResponse`.
- [ ] **Validate `model_name` against `provider`**: Mismatched pairs (e.g. `Anthropic` + `gpt-6-luna`) currently pass validation and fail upstream as a `500`. Add a per-provider allow-list in the backend, validate it with a `model_validator`, and expose it via `GET /api/v1/models` so the Streamlit UI stops hardcoding model lists.

### Error Handling & Status Codes
- [ ] **Map upstream errors accurately**: Return `504` on upstream timeouts (`APITimeoutError` / Qdrant timeout), `502` on upstream LLM/Qdrant errors, and `500` only for unexpected internal bugs.
- [ ] **Stop leaking raw exception strings**: Replace `detail=f"Request failed: {e}"` with a safe client message + `request_id`, while logging full tracebacks (`request.method` and `request.url.path`) server-side.
- [ ] **Centralize exception handlers**: Register `@app.exception_handler(...)` on `app` instead of duplicating `try/except` in every route (and always re-raise `HTTPException` before catching `Exception`).

### Observability, Docs & Security
- [x] **Fix CORS spec violation** (`apps/api/src/api/app.py`): Set `allow_credentials=False` (previously `allow_origins=["*"]` + `allow_credentials=True` made Starlette reflect any `Origin` with credentials).
- [ ] **Request ID / correlation middleware**: Generate a UUID per request, return it in an `X-Request-ID` response header, include it in logs/errors, and attach it to LangSmith run metadata.
- [ ] **Enrich OpenAPI `/docs` metadata**: Add `summary`, `description`, `status_code`, and `responses={502: ..., 504: ...}` to route decorators.
- [ ] **Async migration (future, if needed for concurrency)**: Only switch route handlers from `def` to `async def` alongside switching to `AsyncOpenAI`, `AsyncAnthropic`, and `AsyncQdrantClient`.

---

## 3. Code Quality & Architecture (from code review)

### Retrieval & Prompting (`apps/api/src/api/agents/retrieval_generation.py`)
- [ ] **Harden Qdrant payload parsing**: `retrieve_items` uses `p.payload["parent_asin"]` / `["preprocess_description"]`, so one malformed point raises `KeyError` and fails the request. Use `.get()` and skip + log incomplete points.
- [ ] **Escape prompt XML**: Product descriptions and the user question are interpolated raw into `<product>` / `<user_question>` tags; escape with `xml.sax.saxutils.escape` / `quoteattr` and render missing ratings as `"N/A"` instead of `"None"`.
- [ ] **Revisit history grounding rule**: `SYSTEM_PROMPT` now allows "products already referenced earlier in this conversation", so a hallucinated product from a prior turn can persist. Level 2 query rewriting (Section 1) mitigates this.

### Layering & Duplication
- [ ] **Move domain models to `api/core/schemas.py`**: `agents/` currently imports `ChatMessage`, `RetrievedItem`, `RAGResult` from the HTTP layer (`api/api/models.py`) — an inverted dependency one edit away from a circular import. Keep only request/response schemas in `api/api/models.py`.
- [ ] **Single OpenAI client**: `agents._openai()` and `retrieval_generation._openai_client()` build identical wrapped clients; expose one getter and reuse it for embeddings.
- [ ] **Provider-specific options**: `reasoning_effort="low"` is hardcoded in `_run_openai`; make it configurable per model so non-reasoning models don't fail.

### Configuration (`apps/api/src/api/core/config.py`)
- [ ] **Use `SecretStr` for API keys** (and `.get_secret_value()` at client construction) so keys never appear in reprs/logs.
- [ ] **Make provider keys optional**: Currently all three keys are required even if only OpenAI is used; fail fast with a clear error only when an unconfigured provider is selected.
- [ ] **Lazy config**: Replace import-time `config = Config()` with a cached `get_config()` so tests can import modules without real env vars.

### Testing
- [ ] **Add an automated test suite**: Add `pytest` + `httpx` to dev deps and cover `build_prompt` (history truncation / role filtering), `retrieve_items` (malformed payloads), and `POST /api/v1/rag` (200 / 422 / upstream errors) with Qdrant + LLM clients mocked.

### Streamlit UI (`apps/chatbot_ui/src/chatbot_ui/app.py`)
- [ ] **Loading feedback**: Wrap the API call in `st.spinner(...)` (calls take 3–15s).
- [ ] **Remove unreachable branch**: `else: raise ValueError(...)` after the provider `selectbox` can never execute.
- [ ] **User-friendly errors**: Don't render the backend's raw `detail` string to end users; show a generic message (pairs with the error-mapping item in Section 2).

---

## 4. Packaging, Docker & Tooling

### Packaging (`pyproject.toml` files)
- [ ] **Clean up workspace config**: Root project is named `"prerequisites"` and duplicates app dependencies; keep it for workspace + dev tooling only.
- [ ] **Remove dead config**: Both apps declare `[tool.uv.sources] shared = { workspace = true }` (no such package); `chatbot_ui` uses `uv_build` but still has `[tool.hatch.*]` sections pointing at `src/api`.
- [ ] **Declare direct dependencies**: `chatbot_ui` imports `requests` but only gets it transitively via `streamlit`; add it explicitly.

### Docker & Compose
- [ ] **Pin Qdrant image version** (`qdrant/qdrant:vX.Y.Z`) instead of implicit `latest`.
- [ ] **Healthchecks + startup ordering**: Once `/health` / `/ready` exist (Section 2), add Compose `healthcheck`s and `depends_on: condition: service_healthy` (including `streamlit-app` → `api`).
- [ ] **Split dev vs prod**: Move `uvicorn --reload` and source bind-mounts into a `docker-compose.override.yaml`; keep base images production-ready.
- [ ] **Add `.dockerignore`**: Exclude `.env`, `.venv`, `data/`, `qdrant_data/`, `notebooks/` as insurance against leaking secrets or bloating images.

### Tooling & Docs
- [ ] **Commit linter/type-checker config**: Add ruff config and un-ignore `pyrefly.toml` (currently in `.gitignore`).
- [ ] **Pre-commit + CI**: Run ruff, pyrefly, and pytest on every push.
- [ ] **Enforce clean notebooks**: Install `nbstripout` as a git filter (`01-llm-apis.ipynb` still has a committed output cell).
- [ ] **Docs**: Fill in the empty `README.md` files (root, `apps/api`, `apps/chatbot_ui`) and add a `.env.example` listing required keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `LANGSMITH_*`).

---

## 5. Async Migration Notes (`def` vs `async def`)

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
