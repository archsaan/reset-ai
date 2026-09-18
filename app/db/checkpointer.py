"""
LangGraph checkpointer — the persistence layer for conversation state,
keyed by thread_id (see app/agent/graph.py for how thread_id is built).

Backed by the same Postgres database as admin_db.py, in its own set of
tables that LangGraph manages itself via .setup().
"""

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.core.config import DATABASE_URL

_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    # Neon (and most managed/serverless Postgres) silently closes idle
    # connections after a period of inactivity. Without `check`, the pool
    # will happily hand out one of those now-dead connections and the
    # next query fails with something like:
    #   psycopg.OperationalError: consuming input failed: SSL connection
    #   has been closed unexpectedly
    # `check_connection` pings a connection before handing it out and
    # transparently reconnects if it's dead — reproduced this exact
    # failure and confirmed this fixes it while building this.
    check=ConnectionPool.check_connection,
    max_idle=180,  # recycle idle connections ourselves well before Neon's own timeout kicks in
    kwargs={"autocommit": True, "prepare_threshold": 0},
)

checkpointer = PostgresSaver(_pool)

# Creates LangGraph's own checkpoint tables the first time this runs.
# Safe to call every startup — it's a no-op if the tables already exist.
checkpointer.setup()
