from ddgs import DDGS

def web_search(query, max_results=10):
    try:
        with DDGS() as ddgs:
            # आपण query मध्ये 'India' सारखे शब्द जोडून सर्च अधिक अचूक करू शकतो
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                }
                for r in results
            ]
    except Exception as e:
        print(f"Web search error: {e}")
        return []