"""
Embeddings via Voyage AI — Anthropic's recommended embedding provider
for RAG with Claude (Anthropic doesn't run its own embedding endpoint).
Has a free tier that comfortably covers a project this size.

voyage-3-lite is used here: cheap, fast, 512 dimensions — plenty for a
knowledge base of pricing/FAQ/policy documents. Swap EMBEDDING_MODEL in
config.py for voyage-3 (1024 dims, higher quality) if retrieval quality
ever becomes the bottleneck.

Voyage embeddings are asymmetric: a KB chunk should be embedded with
input_type="document", and a user's question with input_type="query" —
using the right one measurably improves retrieval.
"""

import voyageai

from app.core.config import EMBEDDING_MODEL, VOYAGE_API_KEY

_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of KB chunks for storage."""
    if not texts:
        return []
    result = _client.embed(texts, model=EMBEDDING_MODEL, input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single user question for retrieval."""
    result = _client.embed([text], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]
