# my_rag AGENTS.md

FastAPI chat microservice ("Chat AI Microservice") behind the `my_front` Next.js app. Multi-provider LLM streaming, text-based RAG over Pinecone, and per-provider model availability checks. Python 3.13, uv, LangChain, Pinecone, Pydantic v2.

## Project layout

The project is a uv package with `package = false` set in `[tool.uv]` — it is an intentionally flat application, not an installable library. There is no `[build-system]` or `[project.scripts]`; do not add them back, and do not restructure into `src/` layout. The app is `main.app`, so uvicorn is always started with `main:app` from this directory.

```
main.py                 FastAPI app factory; wires CORS and the four routers
api/                    HTTP layer only: request parsing, StreamingResponse, call one service function
services/               business logic; one file per provider plus rag_service
core/pinecone.py        Pinecone index and embedding client setup
models/request_model.py all Pydantic request bodies
models/response_model.py all Pydantic response bodies
Makefile                dev = uv run uvicorn main:app --reload
```

## Commands

```bash
uv sync
uv run uvicorn main:app --reload      # same as: make dev
uv add <package>                      # dependency add, updates pyproject and uv.lock together
make outdated
make audit
```

The service runs on port 8000 with automatic OpenAPI docs at `/docs`. It is not a package, so no build or test command exists here. When you introduce tests, run them with `uv run pytest tests` without adding a pyproject script.

## Environment

All provider credentials are read from `.env` via `load_dotenv()` at the top of `main.py`, before any other import. Required keys:

| Key | Purpose |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM inference |
| `OPENAI_API_KEY` | OpenAI inference |
| `GEMINI_API_KEY` | Google GenAI inference |
| `OPENROUTER_API_KEY` | OpenRouter inference and `/credits` wallet check |
| `OLLAMA_API_KEY` | Cloud Ollama inference |
| `PINECONE_API_KEY` | Text RAG vector store |
| `NEXT_PROD_URL` | CORS origin in production |
| `NEXT_DEV_URL` | CORS origin in development (default set in compose and .env) |

Services read base URLs and keys from these env vars by default, but a request may carry `base_url` in the body to override the env fallback for a specific provider.

## Routers

`main.py` mounts four routers. All paths are prefixed:

| Router | File | Prefix | Purpose |
|---|---|---|---|
| rag_router | `api/rag_api.py` | `/rag` | Pinecone text RAG: `POST /rag/add`, `DELETE /rag/delete`, `POST /rag/query`, `POST /rag/search` |
| llama_router | `api/local_lama_api.py` | `/llama` | `POST /llama/query` local llama streaming |
| llm_query_router | `api/llm_query_api.py` | `/llm` | `POST /llm/query` multi-provider streaming chat |
| models_router | `api/models_api.py` | `/models` | `GET /models/{provider_id}?base_url=...` model listing keyed by provider API key |

### Streaming contract

`api/llm_query_api.py` defines `_stream_with_error_boundary`, which wraps a provider service stream in a generator that converts exceptions into a final failure message text chunk so the browser stream never dies mid-flight. Unknown providers raise `HTTPException 400` with the allowed provider list before a stream starts; keep that pre-stream validation there. The response is `StreamingResponse(media_type="text/plain")` — plain text chunks, not SSE, matching `TextStreamChatTransport` on the frontend. The frontend Next.js proxy at `my_front/app/api/llm/query/route.ts` bodies `{ provider, model, question, base_url, history }` from the client JSON; `history` is a list of `{ role, content }` in chronological order, oldest first, excluding the current question. Any change to `models/request_model.py` `LLMQueryRequest` must be mirrored in `my_front/app/api/llm/query/route.ts` body composition.

### Provider services

`services/*.py` all follow the same two-function shape: `get_llm(model, base_url)` returning a configured LangChain chat model, and `query_stream(question, model, base_url, history)` yielding text chunks. `api/llm_query_api.py` imports them into a module-level `STREAMERS` dict keyed by provider id; that dict is the single registry the `/llm/query` route dispatches through, so a new provider must be added there to become reachable. Add a new provider by copying the existing service file that is closest in transport; never inline an SDK call inside an api handler. The seven streaming provider ids are `nvidia`, `lm-studio`, `local-ollama`, `cloud-ollama`, `openai`, `gemini`, `openrouter`.

### Model listing

`api/models_api.py` is the current contract for `GET /models/{provider_id}?base_url=...`, keep these facts in mind when editing it:

- `PROVIDER_API_KEYS` maps the five remote provider ids (`openrouter`, `cloud-ollama`, `openai`, `nvidia`, `gemini`) to their env key at import time. An unknown or keyless `provider_id` raises `HTTPException 400`; local providers (`lm-studio`, `local-ollama`) are therefore not listed here on purpose.
- The listing is a single `GET {base_url}/models` with a `Bearer` header and a 30 second `requests` timeout, then each item of `data[].id` becomes `{ id, name }` where `name` is the raw id. There is no availability verification, no caching, and no display-name prettification at the moment; if the frontend later needs verified or renamed models, that logic belongs here or in a service, added without changing the response shape.
- Error shape pitfall: the `except` path returns a bare `[]`, while the success path returns `{"models": [...]}`. The frontend does `data.models as Model[]` and would read `undefined` on the error branch — align the error return to `{"models": []}` when touching this handler.

Available provider ids for the `provider_id` URL segment are exactly the five keys of `PROVIDER_API_KEYS` plus `lm-studio` and `local-ollama` on the streaming side only. The `my_front` UI passes the provider label from Mongo through the `providerId` query param.

### RAG service

`services/rag_service.py` is the only RAG entrypoint: `add_text_to_db(index_id, raw_text)` chunk-embeds and upserts, `delete_text_from_db(index_id)` deletes all vectors whose metadata `index_id` matches, `similarity_search_chunks(query, k=5)` returns `(chunk, score)` pairs, and `query_db(question)` composes a single-string answer for `POST /rag/query`. Chunking lives in `process_text`. Pinecone index and embedder setup lives in `core/pinecone.py`; if you change the embedding model, wipe and rebuild the index because stored vectors become incompatible.

## Conventions

- Requests are validated by Pydantic models in `models/request_model.py` only; handlers must not parse dicts manually. Responses use the paired models in `models/response_model.py`, except streaming endpoints which return raw text.
- Any new endpoint must be added to an existing router and gets a prefix from the mount; never register routes on the app object itself.
- Service functions are sync generators yielding strings; never yield dicts, never yield JSON. Keep the failure-text-chunk contract of `_stream_with_error_boundary` intact: the frontend renders whatever this yields directly into the bubble.
- Failure surfaces as a final error text chunk, not an HTTP error status, once streaming has begun; unreadable keys or unknown providers should raise before streaming starts so the frontend gets a real status code.
- Typing: annotate every function return type. Use `str | None` style, not `Optional`.
- No commented-out code and no explanatory code comments; comments are not allowed in this codebase. Write self-documenting identifiers instead, and move what needs prose into this file or into the Pydantic model `description` fields.
- CORS: adding an origin means adding an env var plus a `main.py` origins list entry, never a hardcoded string.

## Working with the frontend

The frontend `my_front` is a separate git repository and reaches this service only through:
- `GET /models/{provider}?base_url=...` (from `my_front/app/api/llm/models/route.ts`)
- `POST /llm/query` (from `my_front/app/api/llm/query/route.ts`)

Never rename these paths or change their body or response shapes without updating both `my_front` route handlers at the same time. Extension ideas (`add`, `search`) exist in rag_api but are not consumed by the frontend yet.
