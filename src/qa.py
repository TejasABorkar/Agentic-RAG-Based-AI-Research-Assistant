from src.vector_store import search_chunks
from src.llm import ask_llm
from src.web_search import web_search

def web_fallback_answer(question):
    """जेव्हा डॉक्युमेंटमध्ये उत्तर नसेल, तेव्हा वेब सर्च करण्यासाठीचं फंक्शन"""
    web_results = web_search(question)
    if not web_results:
        return {"answer": "No information found on the web.", "source": "Not found", "pages": []}
    
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
    """डॉक्युमेंटमधून उत्तर शोधण्यासाठीचं मुख्य फंक्शन"""
    results = search_chunks(question, top_k=top_k)
    
    if not results['documents'][0]:
        return web_fallback_answer(question)
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if 'distances' in results else [0] * len(chunks)
    
    # ChromaDB distance कमी म्हणजे चांगलं. 
    top_similarity = 1 - distances[0] if distances else 0
    
    context = "\n\n".join([
        f"[Page {m['page']}]: {c}" 
        for c, m in zip(chunks, metadatas)
    ])
    
    prompt = f"""Answer using ONLY the context below. Keep the answer short.
If the answer is clearly not in the context, reply exactly: NOT_FOUND

Context:
{context}

Question: {question}
Answer:"""
    
    answer = ask_llm(prompt)
    
    # 🔴 थ्रेशोल्ड ०.१ (जर डॉक्युमेंटमधलं उत्तर कमजोर असेल तरच वेबवर जाईल)
    if "NOT_FOUND" in answer.upper() or top_similarity < 0.1: 
        return web_fallback_answer(question)
    
    pages = list(set([m['page'] for m in metadatas]))
    return {
        "answer": answer,
        "source": "From document",
        "pages": pages,
        "chunks_used": chunks[:2]
    }