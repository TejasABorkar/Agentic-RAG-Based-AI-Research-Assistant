from src.vector_store import search_chunks
from src.llm import ask_llm
from src.web_search import web_search
from src.config import SIMILARITY_THRESHOLD

def web_fallback_answer(question):
    web_results = web_search(question)
    if not web_results:
        return {"answer": "No information found.", "source": "Not found", "pages": []}
    
    context = "\n\n".join([
        f"[{i+1}] {r['title']}: {r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(web_results[:3])
    ])
    
    prompt = f"""Answer using ONLY these web results. Mention source number.
If results insufficient, say so.

Results:
{context}

Question: {question}
Answer:"""
    
    answer = ask_llm(prompt)
    return {
        "answer": answer,
        "source": "From web",
        "urls": [r['url'] for r in web_results[:3]],
        "pages": []
    }

def answer_question(question, top_k=4):
    results = search_chunks(question, top_k=top_k)
    
    if not results['documents'][0]:
        return web_fallback_answer(question)
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results else [0] * len(chunks)
    
    top_similarity = 1 - distances[0] if distances else 0
    
    context = "\n\n".join([
        f"[Page {m['page']}]: {c}" 
        for c, m in zip(chunks, metadatas)
    ])
    
    prompt = f"""Answer using ONLY the context below. Keep answer short.
If the answer is not in the context, reply exactly: NOT_FOUND

Context:
{context}

Question: {question}
Answer:"""
    
    answer = ask_llm(prompt)
    
    if "NOT_FOUND" in answer.upper() or top_similarity < SIMILARITY_THRESHOLD:
        return web_fallback_answer(question)
    
    pages = list(set([m['page'] for m in metadatas]))
    return {
        "answer": answer,
        "source": "From document",
        "pages": pages,
        "chunks_used": chunks[:2]
    }