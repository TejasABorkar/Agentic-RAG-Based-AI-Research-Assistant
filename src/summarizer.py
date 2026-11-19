from src.llm import ask_llm

def summarize_document(pages, max_pages=10):
    text = "\n".join([p["text"] for p in pages[:max_pages]])
    text = text[:10000]
    
    prompt = f"""Write a short summary (5-10 lines) of this document.
Cover: main purpose, key points, conclusion.

Document:
{text}

Summary:"""
    
    return ask_llm(prompt)
