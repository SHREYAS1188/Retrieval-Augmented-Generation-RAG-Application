"""
src/utils/text_utils.py
========================
Text normalization and language utilities.
"""

import unicodedata


def normalize_text(text: str) -> str:
    """
    Apply NFC Unicode normalization to a string.

    NFC normalization ensures consistent representation of Unicode characters,
    which is especially important for Indic scripts (Kannada, Hindi, Tamil, etc.)
    where the same glyph can be encoded multiple ways.

    Args:
        text: Input string.

    Returns:
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


def is_rag_answer(response: str) -> bool:
    """
    Heuristic check: did the LLM answer from context, or say it didn't know?

    Args:
        response: The LLM's response string.

    Returns:
        True if the answer appears to be grounded in the retrieved context.
    """
    no_info_phrases = [
        "do not have enough information",
        "not present in the context",
        "cannot find",
        "not mentioned",
    ]
    lower = response.lower()
    return not any(phrase in lower for phrase in no_info_phrases)


def get_rag_status(response: str) -> str:
    """
    Return a human-readable RAG status string based on the response.

    Args:
        response: The LLM's response string.

    Returns:
        Status label string.
    """
    if is_rag_answer(response):
        return "✅ Answer based on the uploaded document."
    return "⚠️ No direct answer found in the document."
