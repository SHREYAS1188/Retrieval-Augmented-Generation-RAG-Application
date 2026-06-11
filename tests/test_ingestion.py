"""
tests/test_ingestion.py
========================
Unit tests for the ingestion pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.ingestion.chunker import split_text, chunks_to_documents


class TestSplitText:
    def test_basic_split(self):
        """Long text should be split into multiple chunks."""
        text = "Hello world. " * 200  # ~2600 chars
        chunks = split_text(text)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        """Short text should remain a single chunk."""
        text = "Short text."
        chunks = split_text(text)
        assert len(chunks) == 1

    def test_empty_text(self):
        """Empty text should return an empty list."""
        chunks = split_text("")
        assert chunks == []

    def test_chunk_max_size(self):
        """No chunk should exceed CHUNK_SIZE + CHUNK_OVERLAP characters."""
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        text = "A" * (CHUNK_SIZE * 5)
        chunks = split_text(text)
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE + CHUNK_OVERLAP


class TestChunksToDocuments:
    def test_returns_documents(self):
        from langchain_core.documents import Document
        chunks = ["Hello, this is an English sentence.", "ಕನ್ನಡ ಭಾಷೆ"]
        docs = chunks_to_documents(chunks)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_has_language(self):
        chunks = ["This is an English sentence for language detection."]
        docs = chunks_to_documents(chunks)
        assert "language" in docs[0].metadata

    def test_nfc_normalisation(self):
        # NFC-decomposed 'é' (e + combining acute) → NFC 'é'
        text = "caf\u0065\u0301"  # decomposed
        docs = chunks_to_documents([text])
        assert docs[0].page_content == "café"


class TestPdfLoader:
    def test_missing_file_raises(self, tmp_path):
        from src.ingestion.pdf_loader import pdf_to_markdown
        with pytest.raises(FileNotFoundError):
            pdf_to_markdown(tmp_path / "nonexistent.pdf")

    def test_wrong_extension_raises(self, tmp_path):
        from src.ingestion.pdf_loader import pdf_to_markdown
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError):
            pdf_to_markdown(txt_file)
