import chromadb
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, CHROMA_PATH

_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def add_chunks(chunks, metadatas, collection_name):
    """Create a fresh cosine-similarity collection and store the chunks."""
    client = _get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = _get_model().encode(
        chunks, batch_size=32, normalize_embeddings=True
    ).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    batch = 500
    for i in range(0, len(chunks), batch):
        collection.add(
            documents=chunks[i:i + batch],
            embeddings=embeddings[i:i + batch],
            metadatas=metadatas[i:i + batch],
            ids=ids[i:i + batch],
        )


def search_chunks(query, top_k, collection_name):
    """Return hits: [{"text", "page", "page_end", "source", "similarity"}]."""
    client = _get_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    query_embedding = _get_model().encode(
        [query], normalize_embeddings=True
    ).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
    )

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "page": meta["page"],
            "page_end": meta.get("page_end", meta["page"]),
            "source": meta.get("source", ""),
            "similarity": round(1 - distance, 3),  # cosine distance -> similarity
        })
    return hits