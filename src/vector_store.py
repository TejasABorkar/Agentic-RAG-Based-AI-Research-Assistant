import chromadb
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, CHROMA_PATH

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name="documents"):
    return client.get_or_create_collection(name=name)

def add_chunks(chunks, metadatas, collection_name="documents"):
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = get_collection(collection_name)
    
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def search_chunks(query, top_k=4, collection_name="documents"):
    collection = get_collection(collection_name)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results