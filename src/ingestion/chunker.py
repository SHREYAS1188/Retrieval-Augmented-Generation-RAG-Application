"""
src/ingestion/chunker.py
=========================
Splits Markdown text into overlapping chunks and detects the language
of each chunk for metadata tagging.
"""

import logging
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect, DetectorFactory

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

# Make langdetect deterministic
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)


def split_text(text: str) -> list[str]:
    """
    Split raw text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: The full document text (typically Markdown).

    Returns:
        List of text chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_text(text)
    logger.info("Text split into %d chunks (size=%d, overlap=%d)",
                len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def _detect_language(text: str) -> str:
    """Detect ISO 639-1 language code. Returns 'unknown' on failure."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def chunks_to_documents(chunks: list[str]) -> list[Document]:
    """
    Convert raw text chunks into LangChain Documents with language metadata.

    Args:
        chunks: List of text chunk strings.

    Returns:
        List of LangChain Document objects with `language` metadata.
    """
    documents: list[Document] = []
    for chunk in chunks:
        normalised = unicodedata.normalize("NFC", chunk)
        lang = _detect_language(normalised)
        documents.append(
            Document(
                page_content=normalised,
                metadata={"language": lang},
            )
        )

    lang_counts: dict[str, int] = {}
    for doc in documents:
        lang = doc.metadata["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    logger.info("Created %d documents. Language distribution: %s",
                len(documents), lang_counts)
    return documents
