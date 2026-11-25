import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API keys (set them in the .env file, never in the code)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Main LLM: "gemini" or "groq". The other one is the automatic backup.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# Embeddings and chunking (environment variables can override them;
# evaluate.py uses this to compare embedding models and chunk sizes)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "180"))       # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))  # words
TOP_K = 4
MIN_SIMILARITY = 0.25  # cosine similarity; tune between 0.2 and 0.4

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Summary settings
SUMMARY_BLOCK_CHARS = 6000
SUMMARY_MAX_BLOCKS = 6