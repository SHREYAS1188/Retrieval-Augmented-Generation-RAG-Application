"""
src/ingestion/pdf_loader.py
============================
Handles PDF → Markdown conversion using pymupdf4llm.

Why Markdown?
-------------
Converting PDFs to Markdown preserves semantic structure (headers, tables,
lists) that raw text extraction loses. This results in better chunking,
higher retrieval accuracy, and lower token usage in the LLM.
"""

import logging
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    Convert a PDF file to a Markdown string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Markdown representation of the PDF content.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If conversion fails.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    logger.info("Converting PDF to Markdown: %s", pdf_path.name)

    try:
        md_text: str = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to convert '{pdf_path}' to Markdown: {exc}") from exc

    logger.info(
        "PDF converted successfully. Markdown length: %d characters", len(md_text)
    )
    return md_text


def save_markdown(md_text: str, output_path: str | Path) -> Path:
    """
    Save a Markdown string to a file.

    Args:
        md_text: Markdown content.
        output_path: Destination file path.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Markdown saved to: %s", output_path)
    return output_path
