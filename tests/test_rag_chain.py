"""
tests/test_rag_chain.py
========================
Unit tests for the RAG chain and utility helpers.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.utils.text_utils import normalize_text, is_rag_answer, get_rag_status


class TestNormalizeText:
    def test_nfc_normalization(self):
        decomposed = "caf\u0065\u0301"  # e + combining acute
        assert normalize_text(decomposed) == "café"

    def test_no_op_on_ascii(self):
        text = "Hello world"
        assert normalize_text(text) == text


class TestIsRagAnswer:
    def test_positive_answer(self):
        assert is_rag_answer("The poem describes the beauty of nature.") is True

    def test_negative_answer(self):
        assert is_rag_answer("I do not have enough information.") is False

    def test_partial_match(self):
        assert is_rag_answer("This is not mentioned in the context provided.") is False


class TestGetRagStatus:
    def test_grounded_answer(self):
        status = get_rag_status("The document is about Kannada poetry.")
        assert "✅" in status

    def test_no_info_answer(self):
        status = get_rag_status("I do not have enough information.")
        assert "⚠️" in status


class TestBuildRagChain:
    @patch("src.llm.rag_chain.ChatGoogleGenerativeAI")
    def test_chain_built(self, mock_llm_cls):
        from src.llm.rag_chain import build_rag_chain

        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        chain = build_rag_chain(mock_retriever)
        assert chain is not None
