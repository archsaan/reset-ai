"""
Splits a knowledge base document into overlapping text chunks small
enough to embed and retrieve individually.

Kept deliberately simple: a paragraph-aware sliding window, not a
tokenizer-exact splitter. Good enough for the size of Reset Fitness's
KB docs (pricing, FAQs, policies) — revisit if documents get much
longer or more structured (e.g. a real chunker like
langchain's RecursiveCharacterTextSplitter) once that's needed.
"""

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Splits on paragraph boundaries first (so a chunk doesn't cut a
    sentence in half where avoidable), then packs paragraphs into
    windows of roughly chunk_size characters with overlap between
    windows so a fact sitting right at a chunk boundary isn't lost.

    Any single paragraph longer than chunk_size on its own is hard-split
    (regardless of what's currently buffered in `current`) so no chunk
    ever grows unboundedly large."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap < len(current) else current

    for para in paragraphs:
        if len(para) > chunk_size:
            # Oversized paragraph: flush whatever's buffered first, then
            # hard-split this paragraph on its own into chunk_size pieces.
            flush()
            current = ""
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i:i + chunk_size]
                if piece:
                    chunks.append(piece)
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush()
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)

    return chunks
