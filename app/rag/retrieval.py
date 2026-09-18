"""
Ties chunking + embeddings + vector_store together into the two
operations the rest of the app actually needs: index a doc, retrieve
relevant chunks for a question — both scoped to a specific agent, so
each agent's knowledge base stays isolated from every other agent's.
"""

from app.db.vector_store import has_any_chunks, replace_doc_chunks, search_similar_chunks
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_documents, embed_query


def index_kb_doc(doc_id: int, content: str) -> int:
    """Chunks a KB doc's content, embeds each chunk, and stores them.
    Returns the number of chunks created. Call this right after a doc
    is uploaded (see app/routers/admin.py). Which agent this doc
    belongs to is already fixed by doc_id -> knowledge_base_docs.agent_id,
    so it doesn't need to be passed in here separately."""
    chunks = chunk_text(content)
    if not chunks:
        return 0
    embeddings = embed_documents(chunks)
    replace_doc_chunks(doc_id, chunks, embeddings)
    return len(chunks)


def retrieve_relevant_kb_text(agent_id: int, user_question: str) -> str | None:
    """Returns the joined text of the most relevant KB chunks for a
    question, restricted to the given agent's own knowledge base, or
    None if that agent's KB hasn't been indexed yet (caller should fall
    back to the full-KB-dump behavior in that case)."""
    if not has_any_chunks(agent_id):
        return None
    query_embedding = embed_query(user_question)
    chunks = search_similar_chunks(agent_id, query_embedding)
    if not chunks:
        return "(no matching knowledge base content found for this question)"
    return "\n\n---\n\n".join(chunks)
