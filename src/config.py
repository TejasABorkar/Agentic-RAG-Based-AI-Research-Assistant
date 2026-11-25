import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get(name, default=None):
    """Read from environment/.env first, then from Streamlit secrets (cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return default


# API keys (set them in .env locally, or in Streamlit Secrets when deployed)
GEMINI_API_KEY = _get("GEMINI_API_KEY")
GROQ_API_KEY = _get("GROQ_API_KEY")

# Main LLM: "gemini" or "groq". The other one is the automatic backup.
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# Embeddings and chunking (can be overridden by environment variables,
# evaluate.py uses this to compare embedding models and chunk sizes)
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(_get("CHUNK_SIZE", "180"))       # words
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "30"))  # words
TOP_K = 4
MIN_SIMILARITY = 0.25  # cosine similarity; tune between 0.2 and 0.4

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Summary settings
SUMMARY_BLOCK_CHARS = 6000
SUMMARY_MAX_BLOCKS = 6