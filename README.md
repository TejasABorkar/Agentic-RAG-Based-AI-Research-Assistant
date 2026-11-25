# 📚 Agentic RAG-Based AI Research Assistant

An **agentic Retrieval-Augmented Generation (RAG)** system that lets you upload research papers or long PDFs, get a concise summary, and ask questions in a chat interface. The agent searches your document first. If the answer isn't there, it automatically falls back to a **web search** and returns an answer with cited sources.

---

## 📑 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Example](#-example)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

- **PDF ingestion and summarization**: extracts text page by page, builds semantic chunks, and generates a summary with an LLM.
- **Vector search (RAG)**: uses ChromaDB and Sentence-Transformers to retrieve the most relevant chunks for a question.
- **Agentic web fallback**: if the document doesn't contain the answer, the agent searches DuckDuckGo and answers from the results.
- **Source citations**: every answer shows its origin, either *Document* (with page numbers) or *Web* (with URLs).
- **Interactive chat UI**: built with Streamlit.
- **Multi-LLM support**: Groq (Llama-3.3-70b) as primary, Google Gemini as fallback.

---

## 🧠 How It Works

```
        ┌──────────────┐
        │  Upload PDF  │
        └──────┬───────┘
               ▼
   ┌───────────────────────┐
   │ Extract text (PyMuPDF)│
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐        ┌──────────────┐
   │ Chunk with overlap    │───────▶│ Summarizer   │──▶ Summary
   └──────────┬────────────┘        └──────────────┘
              ▼
   ┌───────────────────────┐
   │ Embed → ChromaDB      │
   └──────────┬────────────┘
              ▼
        User question
              ▼
   ┌───────────────────────┐
   │ Retrieve top chunks   │
   └──────────┬────────────┘
              ▼
      Answer in document?
        ┌─────┴─────┐
       Yes          No
        ▼            ▼
  Answer from   DuckDuckGo search
  document      → Answer from web
  (page nos.)   (URLs)
```

1. **Ingest**: the PDF is parsed, split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.
2. **Retrieve**: for each question, the most similar chunks are fetched.
3. **Decide**: the agent logic in `qa.py` checks whether the retrieved context is sufficient.
4. **Answer**: from the document if it is, otherwise from a live web search, always with citations.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| LLMs | Groq API (Llama-3.3-70b), Google Gemini API |
| Web search | DuckDuckGo (`ddgs`) |

---

## 📂 Project Structure

```
Agentic-RAG-Based-AI-Research-Assistant/
│
├── app/
│   └── main.py            # Streamlit frontend UI
│
├── src/
│   ├── config.py          # Configuration (API keys, model names)
│   ├── pdf_loader.py      # Extracts text from PDF page by page
│   ├── chunker.py         # Splits text into overlapping chunks
│   ├── vector_store.py    # ChromaDB embedding and retrieval
│   ├── llm.py             # LLM API wrapper (Groq / Gemini)
│   ├── summarizer.py      # Generates document summary
│   ├── web_search.py      # DuckDuckGo web search integration
│   └── qa.py              # Agentic decision logic (Document vs Web)
│
├── data/                  # Sample PDFs
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+ and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/TejasABorkar/Agentic-RAG-Based-AI-Research-Assistant.git
cd Agentic-RAG-Based-AI-Research-Assistant

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.env` file in the project root (copy it from `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

- Get a Groq key at [console.groq.com](https://console.groq.com)
- Get a Gemini key at [aistudio.google.com](https://aistudio.google.com)

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## ▶️ Usage

```bash
streamlit run app/main.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and:

1. **Upload** a PDF (research paper, report, book chapter).
2. Read the auto-generated **summary**.
3. **Ask questions** in the chat box.
4. Check the **source label** under each answer (Document pages or Web links).

---

## 💡 Example

| Question | Source used |
|---|---|
| "What dataset did the authors use?" | 📄 Document (pages 4, 5) |
| "What is the latest state-of-the-art on this task?" | 🌐 Web (with URLs) |

---

## ⚠️ Limitations

- Scanned or image-only PDFs need OCR, which is not included.
- Answer quality depends on the LLM and on chunk size and overlap settings.
- Web fallback depends on DuckDuckGo results and may not always be reliable.
- Free-tier API rate limits (Groq / Gemini) may apply.

---

## 🔮 Future Improvements

- Multi-document support and persistent chat history
- OCR for scanned PDFs
- Re-ranking of retrieved chunks
- Support for DOCX / TXT / URLs
- Docker support and cloud deployment

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

---

## 📄 License

Add a license of your choice (for example MIT) and mention it here.

---

## 👤 Author

**Tejas Borkar**: [GitHub](https://github.com/TejasABorkar)