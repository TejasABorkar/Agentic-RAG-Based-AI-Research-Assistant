import fitz

def load_pdf(file_path):
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": file_path.split("\\")[-1]
            })
    doc.close()
    return pages