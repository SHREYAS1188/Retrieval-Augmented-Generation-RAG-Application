"""
scripts/ingest_pdf.py
======================
CLI utility to pre-ingest a PDF into the persistent ChromaDB vector store.

Usage:
    python scripts/ingest_pdf.py --pdf path/to/document.pdf

This is useful for pre-building the knowledge base before deploying the app,
or for batch ingesting multiple documents.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the multilingual RAG vector store."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the PDF file to ingest.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Directory to persist the ChromaDB vector store. "
             "Defaults to the value in config/settings.py.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear an existing vector store before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.ingestion import pdf_to_markdown, split_text, chunks_to_documents
    from src.retrieval import build_vector_store
    from config.settings import CHROMA_DB_DIR

    persist_dir = args.db_dir or CHROMA_DB_DIR

    logger.info("=== Multilingual RAG — PDF Ingestion ===")
    logger.info("PDF      : %s", args.pdf)
    logger.info("DB dir   : %s", persist_dir)

    # Step 1: PDF → Markdown
    md_text = pdf_to_markdown(args.pdf)

    # Step 2: Markdown → Chunks
    chunks = split_text(md_text)

    # Step 3: Chunks → LangChain Documents (with language metadata)
    documents = chunks_to_documents(chunks)

    # Step 4: Documents → ChromaDB
    build_vector_store(
        documents=documents,
        persist_directory=persist_dir,
        clear_existing=not args.no_clear,
    )

    logger.info("✅ Ingestion complete. Vector store saved to: %s", persist_dir)


if __name__ == "__main__":
    main()
