"""Project configuration. All paths and runtime settings live here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Project root resolution
# ============================================================
# This file lives at: <project_root>/src/doc_assistant/config.py
# parents[0] = doc_assistant/
# parents[1] = src/
# parents[2] = <project_root>/

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Runtime data paths
# ============================================================

DATA_PATH = PROJECT_ROOT / "data"
DOCS_PATH = DATA_PATH / "sources"
CACHE_PATH = DATA_PATH / "cache"
CHROMA_PATH = str(DATA_PATH / "chroma")  # Chroma wants a string, not Path
PC_CHROMA_PATH = str(DATA_PATH / "chroma_pc") # Parent - Child
SQLITE_PATH = str(DATA_PATH / "library.db")
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"


# ============================================================
# LLM configuration
# ============================================================

LLM_MODE = os.getenv("LLM_MODE", "api")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ============================================================
# Extraction configuration
# ============================================================

PDF_EXTRACTOR = os.getenv("PDF_EXTRACTOR", "pymupdf")


# ============================================================
# HuggingFace token (read, faster downloads)
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# Experimental flags (Default set to best results)
# ============================================================

# Use parent-child retrieval: small chunks for retrieval precision
USE_PARENT_CHILD = os.getenv("USE_PARENT_CHILD", "true").lower() == "true" # Enabled by default -> better perf in testing

# Multi-query expansion
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "false").lower() == "true"

# Number of chunks/parents passed to the LLM at query time
TOP_K = int(os.getenv("TOP_K", "10"))