import os
from dotenv import load_dotenv

# .env फाईल लोड करा
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
SIMILARITY_THRESHOLD = 0.3
CHROMA_PATH = "./chroma_db"