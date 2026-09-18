"""
Admin database — agents (system prompt + model per agent), knowledge
base docs (scoped to an agent), usage logs (scoped to an agent).

Multi-agent support: what used to be a single global `agent_config` row
is now one row per agent in the `agents` table. Every KB doc belongs to
exactly one agent (`agent_id`), so each agent's knowledge base — and
RAG retrieval over it, see app/rag/retrieval.py — is fully isolated from
every other agent's.
"""

import psycopg
from psycopg.rows import dict_row

from app.core.config import DATABASE_URL, DEFAULT_AGENTS


def get_admin_db() -> psycopg.Connection:
    """Opens a new connection. Simple and safe at this scale — move to a
    connection pool (psycopg_pool) once request volume makes it worth it."""
    return psycopg.connect(DATABASE_URL, autocommit=False)


def init_admin_db() -> None:
    """Creates the multi-agent schema, and migrates a pre-multi-agent
    database in place rather than assuming a fresh database:
      - an old singleton `agent_config` row (if present) becomes the seed
        for the "tribe-app" agent, so a customized system prompt/model
        isn't lost
      - an old `knowledge_base_docs` table missing `agent_id` gets that
        column added and backfilled to the "tribe-app" agent, instead of
        being left broken
      - an old `usage_logs` table missing `agent_id` gets that column
        added as nullable (historical rows just have no agent attributed)
    Safe to call on every startup either way — every step below checks
    before acting."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id SERIAL PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL,
                active_model TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # knowledge_base_docs / usage_logs may already exist from before
        # multi-agent support — create them fresh if not, or migrate them
        # in place if so.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base_docs (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                uploaded_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id SERIAL PRIMARY KEY,
                thread_id TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("SELECT COUNT(*) FROM agents")
        agents_were_empty = cur.fetchone()[0] == 0

        if agents_were_empty:
            # Preserve a pre-existing customized prompt/model if this
            # database has the old singleton agent_config table.
            tribe_app_prompt = DEFAULT_AGENTS[0]["system_prompt"]
            tribe_app_model = DEFAULT_AGENTS[0]["active_model"]
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables WHERE table_name = 'agent_config'
                )
            """)
            if cur.fetchone()[0]:
                cur.execute("SELECT system_prompt, active_model FROM agent_config WHERE id = 1")
                old_config = cur.fetchone()
                if old_config:
                    tribe_app_prompt, tribe_app_model = old_config

            tribe_app_id = None
            for agent in DEFAULT_AGENTS:
                prompt = tribe_app_prompt if agent["slug"] == "tribe-app" else agent["system_prompt"]
                model = tribe_app_model if agent["slug"] == "tribe-app" else agent["active_model"]
                cur.execute(
                    "INSERT INTO agents (slug, name, description, system_prompt, active_model) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (agent["slug"], agent["name"], agent["description"], prompt, model)
                )
                new_id = cur.fetchone()[0]
                if agent["slug"] == "tribe-app":
                    tribe_app_id = new_id

            # Migrate knowledge_base_docs to be agent-scoped.
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'knowledge_base_docs' AND column_name = 'agent_id'
                )
            """)
            if not cur.fetchone()[0]:
                cur.execute("ALTER TABLE knowledge_base_docs ADD COLUMN agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE")
                # Any docs uploaded before multi-agent support existed
                # were for the one agent that existed then — attribute
                # them to tribe-app rather than deleting/orphaning them.
                cur.execute("UPDATE knowledge_base_docs SET agent_id = %s WHERE agent_id IS NULL", (tribe_app_id,))
                cur.execute("ALTER TABLE knowledge_base_docs ALTER COLUMN agent_id SET NOT NULL")

            # Migrate usage_logs to be agent-scoped (nullable — old rows
            # simply have no agent attributed, which is fine for a log).
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'usage_logs' AND column_name = 'agent_id'
                )
            """)
            if not cur.fetchone()[0]:
                cur.execute("ALTER TABLE usage_logs ADD COLUMN agent_id INTEGER REFERENCES agents(id) ON DELETE SET NULL")
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Agents
# ---------------------------------------------------------

def list_agents() -> list[dict]:
    conn = get_admin_db()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, slug, name, description, active_model, created_at FROM agents ORDER BY id")
        rows = cur.fetchall()
    conn.close()
    return rows


def get_agent_by_slug(slug: str) -> dict | None:
    conn = get_admin_db()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM agents WHERE slug = %s", (slug,))
        row = cur.fetchone()
    conn.close()
    return row


def get_agent_by_id(agent_id: int) -> dict | None:
    conn = get_admin_db()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM agents WHERE id = %s", (agent_id,))
        row = cur.fetchone()
    conn.close()
    return row


def create_agent(slug: str, name: str, description: str, system_prompt: str, active_model: str) -> int:
    """Returns the new agent's id."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agents (slug, name, description, system_prompt, active_model) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (slug, name, description, system_prompt, active_model)
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return new_id


def update_agent(
    agent_id: int,
    name: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    active_model: str | None = None,
) -> None:
    current = get_agent_by_id(agent_id)
    if not current:
        return
    new_name = name if name is not None else current["name"]
    new_description = description if description is not None else current["description"]
    new_prompt = system_prompt if system_prompt is not None else current["system_prompt"]
    new_model = active_model if active_model is not None else current["active_model"]

    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE agents SET name = %s, description = %s, system_prompt = %s, active_model = %s WHERE id = %s",
            (new_name, new_description, new_prompt, new_model, agent_id)
        )
    conn.commit()
    conn.close()


def delete_agent(agent_id: int) -> None:
    """Cascades to that agent's knowledge_base_docs (and their kb_chunks,
    via ON DELETE CASCADE on that table too)."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM agents WHERE id = %s", (agent_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Knowledge base docs (scoped to an agent)
# ---------------------------------------------------------

def get_active_kb_text(agent_id: int) -> str:
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM knowledge_base_docs WHERE agent_id = %s AND active = TRUE",
            (agent_id,)
        )
        rows = cur.fetchall()
    conn.close()
    return "\n\n---\n\n".join(r[0] for r in rows)


def list_kb_docs(agent_id: int) -> list[dict]:
    conn = get_admin_db()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, filename, active, uploaded_at FROM knowledge_base_docs "
            "WHERE agent_id = %s ORDER BY uploaded_at DESC",
            (agent_id,)
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def insert_kb_doc(agent_id: int, filename: str, content: str) -> int:
    """Returns the new doc's id, so the caller can index it into the
    vector store right after inserting it."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO knowledge_base_docs (agent_id, filename, content, active) "
            "VALUES (%s, %s, %s, TRUE) RETURNING id",
            (agent_id, filename, content)
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return new_id


def toggle_kb_doc(doc_id: int) -> bool | None:
    """Returns the new active state, or None if the doc wasn't found."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute("SELECT active FROM knowledge_base_docs WHERE id = %s", (doc_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        new_state = not row[0]
        cur.execute("UPDATE knowledge_base_docs SET active = %s WHERE id = %s", (new_state, doc_id))
    conn.commit()
    conn.close()
    return new_state


def delete_kb_doc(doc_id: int) -> None:
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM knowledge_base_docs WHERE id = %s", (doc_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Usage logs (scoped to an agent)
# ---------------------------------------------------------

def log_usage(agent_id: int | None, thread_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
    conn = get_admin_db()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO usage_logs (agent_id, thread_id, model, input_tokens, output_tokens) "
            "VALUES (%s, %s, %s, %s, %s)",
            (agent_id, thread_id, model, input_tokens, output_tokens)
        )
    conn.commit()
    conn.close()


def get_usage_summary(agent_id: int | None = None) -> list[tuple]:
    """If agent_id is given, restricts to that agent; otherwise summarizes
    across all agents (useful for a global usage view)."""
    conn = get_admin_db()
    with conn.cursor() as cur:
        if agent_id is not None:
            cur.execute("""
                SELECT model, COUNT(*) as requests, SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output
                FROM usage_logs WHERE agent_id = %s GROUP BY model
            """, (agent_id,))
        else:
            cur.execute("""
                SELECT model, COUNT(*) as requests, SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output
                FROM usage_logs GROUP BY model
            """)
        rows = cur.fetchall()
    conn.close()
    return rows
