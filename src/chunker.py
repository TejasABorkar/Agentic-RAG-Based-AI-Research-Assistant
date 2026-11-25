from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split pages into overlapping word chunks.

    Chunks may cross page boundaries. Each chunk stores the page where it
    starts ("page") and where it ends ("page_end").
    Returns (chunks, metadatas).
    """
    words = []
    for p in pages:
        for word in p["text"].split():
            words.append((word, p["page"]))

    chunks, metadatas = [], []
    if not words:
        return chunks, metadatas

    source = pages[0]["source"]
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append(" ".join(w for w, _ in window))
        metadatas.append({
            "page": window[0][1],
            "page_end": window[-1][1],
            "source": source,
        })
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks, metadatas