"""
tests/test_retrieval.py
========================
Unit tests for the retrieval layer.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestGetRetriever:
    def test_returns_retriever(self):
        from src.retrieval.vector_store import get_retriever

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        retriever = get_retriever(mock_vs, k=3)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert retriever is mock_retriever


class TestLoadVectorStore:
    def test_returns_none_when_directory_missing(self, tmp_path):
        from src.retrieval.vector_store import load_vector_store
        result = load_vector_store(tmp_path / "nonexistent_db")
        assert result is None
