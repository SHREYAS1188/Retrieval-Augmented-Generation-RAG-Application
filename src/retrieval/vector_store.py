"""
src/retrieval/vector_store.py
==============================
Manages the ChromaDB vector store: building, persisting, loading, and
exposing a LangChain retriever.
"""

import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import CHROMA_DB_DIR, EMBEDDING_MODEL, RETRIEVER_K

logger = logging.getLogger(__name__)

# Module-level cache so the heavy embedding model is loaded only once per process
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return (or lazily initialise) the shared embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model


def build_vector_store(
    documents: list[Document],
    persist_directory: str | Path | None = CHROMA_DB_DIR,
    clear_existing: bool = True,
) -> Chroma:
    """
    Build a ChromaDB vector store from a list of Documents.

    Args:
        documents: LangChain Document objects to embed and store.
        persist_directory: Directory to persist the DB. Pass ``None`` for
            an in-memory (ephemeral) store — useful on Hugging Face Spaces.
        clear_existing: If True and a directory already exists, wipe it
            before building. Prevents stale data from mixing with new data.

    Returns:
        Initialised Chroma vector store.
    """
    embedding_model = get_embedding_model()

    if persist_directory is not None:
        persist_directory = Path(persist_directory)
        if clear_existing and persist_directory.exists():
            shutil.rmtree(persist_directory)
            logger.info("Cleared existing vector store at: %s", persist_directory)

    logger.info("Building vector store from %d documents …", len(documents))

    kwargs: dict = {
        "documents": documents,
        "embedding": embedding_model,
    }
    if persist_directory is not None:
        kwargs["persist_directory"] = str(persist_directory)

    vector_store = Chroma.from_documents(**kwargs)
    logger.info("Vector store built successfully.")
    return vector_store


def load_vector_store(
    persist_directory: str | Path = CHROMA_DB_DIR,
) -> Chroma | None:
    """
    Load an existing persisted ChromaDB vector store.

    Args:
        persist_directory: Directory where the DB was persisted.

    Returns:
        Loaded Chroma instance, or ``None`` if the directory does not exist.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        logger.warning("No vector store found at: %s", persist_directory)
        return None

    logger.info("Loading existing vector store from: %s", persist_directory)
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_model,
    )
    logger.info("Vector store loaded.")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """
    Create a retriever from a Chroma vector store.

    Args:
        vector_store: Initialised Chroma instance.
        k: Number of top documents to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
