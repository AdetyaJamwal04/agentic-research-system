"""
Centralized configuration for the Agentic Research System.
All API keys, model settings, and paths in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---- API Keys ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---- Model Settings ----
DEFAULT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- Retrieval Settings ----
MAX_QUERIES_PER_TASK = 6
MAX_RETRIEVED_DOCS = 20
RERANK_TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- Critic Settings ----
CRITIC_THRESHOLD = 8          # Score >= this = sufficient
MAX_RETRY_ITERATIONS = 3

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "research.db")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHROMA_COLLECTION_NAME = "rag_documents"
