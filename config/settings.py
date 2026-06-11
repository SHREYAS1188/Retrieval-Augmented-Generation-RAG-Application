"""
config/settings.py
==================
Centralised configuration for the Multilingual RAG pipeline.
All environment variables and tuneable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_PDFS_DIR = DATA_DIR / "sample_pdfs"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file or set it as an environment variable."
    )

# Set it for LangChain / Google SDK consumers
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
# BAAI/bge-m3 supports 100+ languages including Kannada, Hindi, Tamil, etc.
# Switch to "models/embedding-001" (Google) for single-language (English) use.
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "100"))
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
RETRIEVER_K: int = int(os.environ.get("RETRIEVER_K", "5"))

# ---------------------------------------------------------------------------
# Speech-to-Text (Whisper)
# ---------------------------------------------------------------------------
# Options: "tiny", "base", "small", "medium", "large"
# Larger models are more accurate but slower.
WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
GRADIO_SERVER_PORT: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE: bool = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
GRADIO_DEBUG: bool = os.environ.get("GRADIO_DEBUG", "false").lower() == "true"
