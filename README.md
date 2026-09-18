# Reset Fitness Agent — restructured

Same behavior as the original single-file `main.py`, split into a
standard, scalable FastAPI layout so the project can grow without
turning back into one giant file.

## Layout

```
app/
  main.py              # FastAPI app creation + router mounting only
  core/
    config.py          # every constant/knob (API URLs, models, prompts, rules)
    security.py        # ALL auth decisions live here (require_admin, require_member)
  db/
    admin_db.py         # agent config, knowledge base docs, usage logs (Postgres)
    checkpointer.py      # LangGraph conversation-state persistence (Postgres)
    vector_store.py       # pgvector storage for KB chunk embeddings
  agent/
    state.py             # AgentState (LangGraph state shape)
    prompts.py           # build_system_prompt() — RAG retrieval with full-dump fallback
    graph.py             # agent node, tool node, conditional edges, compiled graph
    tools/
      schedule_tools.py  # get_schedule, check_availability (real ProfitConnect API)
      booking_tools.py   # book_class (currently mock — see TODOs)
  rag/
    chunking.py           # splits KB doc text into overlapping chunks
    embeddings.py          # Voyage AI embed_documents() / embed_query()
    retrieval.py            # index_kb_doc(), retrieve_relevant_kb_text()
  routers/
    chat.py              # POST /chat  (public — web widget / WhatsApp / Messenger)
    admin.py              # /admin/*   (dashboard: prompt editor, KB, model, usage, test-chat)
  schemas/
    chat.py               # ChatRequest, TestChatRequest
    admin.py               # placeholder for typed admin request models
```

## Why split it this way

- **core/** — the knobs. Change a URL, a model name, or how auth works
  without touching business logic.
- **db/** — anything that opens a database connection. Both files now
  point at the same Postgres database (via `DATABASE_URL`), each owning
  its own tables — one connection string, one thing to back up.
- **agent/** — the actual "brain": state shape, prompt assembly, tools,
  and the LangGraph wiring. This is where a new tool or a new node in the
  graph gets added.
- **routers/** — thin HTTP layer. A route reads the request, calls into
  `agent/` or `db/`, and returns. No SQL, no prompt text, no LangGraph
  objects constructed inline.
- **schemas/** — the Pydantic request/response contracts, decoupled from
  the routes that use them.

## Run it

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
ANTHROPIC_API_KEY=your key
SCHEDULE_API_KEY=your key (if the ProfitConnect API needs one)
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
```

`DATABASE_URL` comes from your Postgres provider's dashboard (Neon, Railway,
Render Postgres, etc.). `admin_db.py`, `checkpointer.py`, and
`vector_store.py` all connect to this same database — `checkpointer.py`
runs `.setup()` at import time and `vector_store.py`'s `init_vector_store()`
runs at app startup, both creating their own tables automatically the
first time the app starts (vector_store also enables the `pgvector`
extension, so no manual `CREATE EXTENSION` step is needed).

```
uvicorn app.main:app --reload
```

## How KB retrieval (RAG) works now

Re-uploading a KB doc through `POST /admin/knowledge-base` now:
1. Saves the doc's full text as before (unchanged behavior)
2. Splits it into ~800-character overlapping chunks (`app/rag/chunking.py`)
3. Embeds each chunk with Voyage AI (`app/rag/embeddings.py`) and stores
   them in the `kb_chunks` table (`app/db/vector_store.py`)

On every chat turn, `build_system_prompt()` embeds the user's actual
question and pulls only the most relevant chunks (cosine similarity via
pgvector), instead of dumping every KB doc's full text into the prompt.

**Graceful fallback, deliberately:** if `VOYAGE_API_KEY` is missing/wrong,
Voyage is unreachable, or no doc has been indexed yet, `build_system_prompt()`
falls back to the old full-KB-dump behavior automatically — a RAG hiccup
degrades the answer quality, it never breaks the chat. Existing KB docs
uploaded before this feature was added won't be indexed until you
re-upload them through `/admin/knowledge-base`.

**No vector index yet, on purpose:** at Reset Fitness's KB size (a handful
of docs), a sequential scan over `kb_chunks.embedding` is exact and
effectively instant. An `ivfflat`/`hnsw` index only pays off past roughly
10k+ chunks — adding one earlier than that can actually *hurt* recall
(this was tested directly while building this: with a tiny row count, an
`ivfflat` index caused searches to silently return zero matches). Add an
index back once the KB is genuinely large.

## Known gaps / next steps (tracked here on purpose)

1. **Real booking API** — `app/agent/tools/booking_tools.py` still uses
   an in-memory mock. Fill in `BOOKING_CREATE_API_URL` /
   `BOOKING_CANCEL_API_URL` in `app/core/config.py` once the real
   endpoint details (URL, method, sample payload) are available, then
   replace the mock block in `book_class()` with a real HTTP call.
2. **Approval gate for booking** — before the real API goes live, add a
   step between "model wants to book" and "booking actually happens":
   model proposes → code validates → (once real auth exists) member
   confirms → tool executes. Natural place for this is either inside
   `book_class` or as an extra node in `app/agent/graph.py`.
3. **Real member auth** — `app/core/security.py` has a `require_member`
   placeholder. Wire in the existing member app's login/JWT there when
   ready; nothing else should need to change.
4. **Vector index at scale** — see "No vector index yet" above; revisit
   once the KB has grown well past what a sequential scan comfortably
   handles.
