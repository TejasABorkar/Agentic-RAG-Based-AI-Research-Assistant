\# 📚 Agentic RAG-Based AI Research Assistant



An advanced, fully offline/online hybrid \*\*Retrieval-Augmented Generation (RAG)\*\* system that allows users to upload research papers or long documents (PDFs), generate summaries, and ask questions. The system acts as an \*\*Agent\*\*—it searches the document first, but if the answer isn't there, it automatically falls back to a web search to provide accurate, cited answers.



\## 🚀 Features



\- \*\*PDF Ingestion \& Summarization:\*\* Extracts text from PDFs, creates semantic chunks, and generates a concise summary using LLMs.

\- \*\*Vector Search (RAG):\*\* Uses `ChromaDB` and `sentence-transformers` to find the most relevant chunks from the document.

\- \*\*Agentic Web Fallback:\*\* If the document lacks the answer, the agent automatically searches the web (DuckDuckGo) and generates an answer with source links.

\- \*\*Source Citations:\*\* Clearly shows whether the answer came from the "Document" (with page numbers) or "Web" (with URLs).

\- \*\*Interactive UI:\*\* Built with Streamlit for a seamless chat-based user experience.

\- \*\*Multi-LLM Support:\*\* Powered by Groq (Llama-3) with fallback to Google Gemini.



\## 🛠️ Technology Stack



\- \*\*Frontend:\*\* Streamlit

\- \*\*Backend:\*\* Python

\- \*\*PDF Parsing:\*\* PyMuPDF (`fitz`)

\- \*\*Embeddings:\*\* Sentence-Transformers (`all-MiniLM-L6-v2`)

\- \*\*Vector Database:\*\* ChromaDB

\- \*\*LLMs:\*\* Groq API (Llama-3.3-70b) / Google Gemini API

\- \*\*Web Search:\*\* DuckDuckGo Search (`ddgs`)



\## 📂 Project Structure



```text

Agentic\_RAG\_Project/

│

├── app/

│   └── main.py               # Streamlit Frontend UI

│

├── src/

│   ├── config.py             # Configuration (API Keys, Model names)

│   ├── pdf\_loader.py         # Extracts text from PDF page-by-page

│   ├── chunker.py            # Splits text into overlapping chunks

│   ├── vector\_store.py       # ChromaDB embedding and retrieval

│   ├── llm.py                # LLM API wrapper (Groq/Gemini)

│   ├── summarizer.py         # Generates document summary

│   ├── web\_search.py         # DuckDuckGo web search integration

│   └── qa.py                 # Agentic decision logic (Doc vs Web)

│

├── data/                     # Folder for sample PDFs

├── .env.example              # Environment variables template

├── .gitignore                # Git ignore file

├── requirements.txt          # Python dependencies

└── README.md                 # Project documentation

