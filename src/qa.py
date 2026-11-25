from src.vector_store import search_chunks
from src.llm import ask_llm
from src.web_search import web_search

def web_fallback_answer(question):
    """जेव्हा डॉक्युमेंटमध्ये उत्तर नसेल, तेव्हा वेब सर्च करण्यासाठीचं फंक्शन"""
    web_results = web_search(question, max_results=10)
    if not web_results:
        return {"answer": "I could not find any relevant information on the web. Please try rephrasing your question.", "source": "Not found", "pages": []}
    
    results_text = "\n\n".join([
        f"[{i+1}] Title: {r['title']}\nSnippet: {r['snippet']}"
        for i, r in enumerate(web_results)
    ])
    
    # 🔴 प्रॉम्प्ट सुधारला: जर काहीच रेलेव्हंट नसेल तर 'None' म्हणायला सांगितलं आहे
    rank_prompt = f"""You are a strict search assistant. Given the user's question and 10 search results, identify the top 3 most relevant results that actually answer the question.
If none of the results are relevant, return exactly: None
Return ONLY the numbers of the top 3 results as a comma-separated list (e.g., 2, 5, 7). Do not include any other text.

Question: {question}

Search Results:
{results_text}

Top 3 result numbers (or 'None'):"""
    
    top_indices_str = ask_llm(rank_prompt).strip()
    
    if "None" in top_indices_str:
        return {"answer": "I could not find any relevant information on the web for this question.", "source": "Not found", "pages": []}

    try:
        top_indices = [int(i.strip()) - 1 for i in top_indices_str.split(",") if i.strip().isdigit()]
        top_indices = top_indices[:3]
        if not top_indices:
            top_indices = [0, 1, 2] 
    except:
        top_indices = [0, 1, 2]
        
    top_results = [web_results[i] for i in top_indices if i < len(web_results)]
    
    context = "\n\n".join([
        f"[{i+1}] {r['title']}: {r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(top_results)
    ])
    
    answer_prompt = f"""Answer using ONLY these web results. Mention the source number if possible.
If the results are insufficient, say so.

Results:
{context}

Question: {question}
Answer:"""
    
    answer = ask_llm(answer_prompt)
    return {
        "answer": answer,
        "source": "From web",
        "urls": [r['url'] for r in top_results],
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