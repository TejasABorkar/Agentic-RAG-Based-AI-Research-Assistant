import os

import fitz  # PyMuPDF


def load_pdf(file_path, display_name=None):
    """Return a list of {"text", "page", "source"} for pages that contain text."""
    source = display_name or os.path.basename(file_path)
    pages = []
    with fitz.open(file_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "page": index + 1, "source": source})
    return pages