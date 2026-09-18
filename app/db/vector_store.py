"""
pgvector-backed storage for knowledge base chunk embeddings.

Lives in the same Postgres database as admin_db.py and checkpointer.py —
one extension (`vector`), one extra table (`kb_chunks`), no separate
vector database service to run or pay for.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.core.config import DATABASE_URL, EMBEDDING_DIM, KB_RETRIEVAL_TOP_K


def get_vector_db() -> psycopg.Connection:
    """For use AFTER the `vector` extension is known to exist (i.e. after
    init_vector_store() has run at least once) — register_vector() looks
    up the `vector` type in the database and fails on a database that
    doesn't have the extension enabled yet."""
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    register_vector(conn)
    return conn


def init_vector_store() -> None:
    """Creates the `vector` extension and kb_chunks table if they don't
    exist yet. Uses a plain, unregistered connection deliberately — on a
    brand new database, the `vector` type doesn't exist until the
    CREATE EXTENSION below runs, so calling register_vector() first (as
    get_vector_db() does) would fail here specifically. This was a real
    bug caught testing against a genuinely fresh database: it worked in
    earlier testing only because that database already had the
    extension enabled from a prior manual step."""
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id SERIAL PRIMARY KEY,
                doc_id INTEGER NOT NULL REFERENCES knowledge_base_docs(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM})
            )
        """)
        # Deliberately NOT creating an ivfflat/hnsw index here. Those
        # indexes are approximate — they trade recall for speed by only
        # searching a subset of the data — and only pay off once you have
        # tens of thousands of rows. At Reset Fitness's KB scale (a
        # handful of docs, low hundreds of chunks), a sequential scan
        # over the embedding column is exact AND effectively instant.
        # Tested directly: with an ivfflat index on this few rows, Postgres
        # warns "low recall" and searches can silently return zero
        # matches (each search only probes 1 of many near-empty index
        # buckets by default) — a real bug caught while building this.
        # Add an index back (ivfflat once you're past ~10k chunks, hnsw
        # for better recall at higher build cost) only once the KB is
        # actually that large.
    conn.commit()
    conn.close()


def replace_doc_chunks(doc_id: int, chunks: list[str], embeddings: list[list[float]]) -> None:
    """Deletes any existing chunks for this doc and inserts the new set —
    called whenever a KB doc is (re)uploaded, so re-uploading the same
    filename doesn't leave stale chunks behind."""
    conn = get_vector_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (doc_id,))
        for chunk_text, embedding in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO kb_chunks (doc_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                (doc_id, chunk_text, Vector(embedding))
            )
    conn.commit()
    conn.close()


def delete_doc_chunks(doc_id: int) -> None:
    """Not strictly needed since ON DELETE CASCADE handles this when the
    parent doc row is deleted — kept as an explicit function for clarity
    and for the case a doc is cleared without deleting its row."""
    conn = get_vector_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (doc_id,))
    conn.commit()
    conn.close()


def search_similar_chunks(agent_id: int, query_embedding: list[float], top_k: int = KB_RETRIEVAL_TOP_K) -> list[str]:
    """Returns the top_k KB chunk texts most similar to the query
    embedding, restricted to active KB docs belonging to this specific
    agent (cosine distance — smaller is more similar). This is what
    keeps one agent's knowledge base from leaking into another agent's
    answers — the filter is on agent_id, not just active."""
    conn = get_vector_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT kb_chunks.chunk_text
            FROM kb_chunks
            JOIN knowledge_base_docs ON knowledge_base_docs.id = kb_chunks.doc_id
            WHERE knowledge_base_docs.active = TRUE
              AND knowledge_base_docs.agent_id = %s
            ORDER BY kb_chunks.embedding <=> %s
            LIMIT %s
        """, (agent_id, Vector(query_embedding), top_k))
        rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def has_any_chunks(agent_id: int) -> bool:
    """Lets prompts.py fall back to the old full-KB-dump behavior if this
    agent's KB hasn't been indexed yet (e.g. right after this feature is
    added, before any doc has been re-uploaded through the new upload
    path — or simply a brand new agent with no KB docs at all)."""
    conn = get_vector_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM kb_chunks
                JOIN knowledge_base_docs ON knowledge_base_docs.id = kb_chunks.doc_id
                WHERE knowledge_base_docs.agent_id = %s
                LIMIT 1
            )
        """, (agent_id,))
        exists = cur.fetchone()[0]
    conn.close()
    return exists
