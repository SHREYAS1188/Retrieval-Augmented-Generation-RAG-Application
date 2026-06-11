"""Retrieval package: vector store management and retriever creation."""

from .vector_store import (
    get_embedding_model,
    build_vector_store,
    load_vector_store,
    get_retriever,
)

__all__ = [
    "get_embedding_model",
    "build_vector_store",
    "load_vector_store",
    "get_retriever",
]
