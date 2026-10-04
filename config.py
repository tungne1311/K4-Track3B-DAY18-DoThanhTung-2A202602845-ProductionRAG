"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OpenRouter (OpenAI-compatible): đặt OPENAI_BASE_URL=https://openrouter.ai/api/v1 trong .env
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

# Optional independent RAGAS judge model.
EVAL_MODEL = os.getenv("EVAL_MODEL", LLM_MODEL)
# Some OpenAI-compatible providers (e.g. Gemini) reject n>1; RAGAS then loops single calls.
EVAL_SUPPORTS_N = os.getenv("EVAL_SUPPORTS_N", "1") == "1"
# Client-side request caps for free tiers (0 = unlimited).
LLM_RPM = float(os.getenv("LLM_RPM", "0"))
EVAL_RPM = float(os.getenv("EVAL_RPM", "0"))

# Optional separate provider for M5 enrichment (defaults to the generation provider).
ENRICHMENT_API_KEY = os.getenv("ENRICHMENT_API_KEY", OPENAI_API_KEY)
ENRICHMENT_BASE_URL = os.getenv("ENRICHMENT_BASE_URL", OPENAI_BASE_URL)
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", LLM_MODEL)
