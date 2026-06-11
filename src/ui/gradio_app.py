"""
src/ui/gradio_app.py
=====================
Gradio interface definition for the Universal Multilingual RAG app.

Inputs:
  - PDF file (optional — upload to rebuild knowledge base)
  - Audio file (optional — voice query via Whisper)
  - Text input (optional — typed query)

Outputs:
  - Transcribed / normalised query string
  - RAG answer
  - RAG status label
"""

import logging

import gradio as gr

from src.ingestion import pdf_to_markdown, split_text, chunks_to_documents
from src.retrieval import build_vector_store, load_vector_store, get_retriever
from src.llm import build_rag_chain, invoke_chain
from src.utils import transcribe_audio, normalize_text, get_rag_status
from config.settings import GRADIO_SERVER_PORT, GRADIO_SHARE, GRADIO_DEBUG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — module-level so Gradio function closures share it
# ---------------------------------------------------------------------------
_state: dict = {
    "vector_store": None,
    "rag_chain": None,
}


def _rebuild_pipeline(vector_store) -> None:
    """Rebuild the retriever and RAG chain after a new vector store is created."""
    retriever = get_retriever(vector_store)
    _state["vector_store"] = vector_store
    _state["rag_chain"] = build_rag_chain(retriever)
    logger.info("Pipeline rebuilt with new vector store.")


def _try_load_existing_db() -> str:
    """Attempt to load a persisted vector store on startup."""
    vector_store = load_vector_store()
    if vector_store is not None:
        _rebuild_pipeline(vector_store)
        return "Existing knowledge base loaded. You can start asking questions."
    return "No knowledge base found. Please upload a PDF document to begin."


def process_query(
    pdf_file,
    audio_path: str | None,
    text_query: str,
) -> tuple[str, str, str]:
    """
    Main Gradio handler function.

    Args:
        pdf_file: Gradio file object (or None).
        audio_path: Path to uploaded audio file (or None).
        text_query: Typed query string (or empty).

    Returns:
        Tuple of (query_display, answer, status).
    """
    # ------------------------------------------------------------------
    # Step 1 — Process uploaded PDF if provided
    # ------------------------------------------------------------------
    if pdf_file is not None:
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
        logger.info("Processing uploaded PDF: %s", pdf_path)
        try:
            md_text = pdf_to_markdown(pdf_path)
            chunks = split_text(md_text)
            documents = chunks_to_documents(chunks)
            # In-memory store — no persist_directory — safe for cloud deployments
            vector_store = build_vector_store(documents, persist_directory=None)
            _rebuild_pipeline(vector_store)
            logger.info("New knowledge base built from uploaded PDF.")
        except Exception as exc:
            logger.exception("PDF processing error")
            return "", "", f"❌ Error processing PDF: {exc}"

    # ------------------------------------------------------------------
    # Step 2 — Ensure a knowledge base is available
    # ------------------------------------------------------------------
    if _state["rag_chain"] is None:
        return "", "", "⚠️ Please upload a PDF document first to build the knowledge base."

    # ------------------------------------------------------------------
    # Step 3 — Resolve the user query (audio > text)
    # ------------------------------------------------------------------
    user_query = ""

    if audio_path:
        try:
            user_query = transcribe_audio(audio_path)
            logger.info("Audio transcribed: %.80s …", user_query)
        except Exception as exc:
            logger.exception("Audio transcription error")
            return "", "", f"❌ Audio transcription failed: {exc}"

    if not user_query and text_query.strip():
        user_query = text_query.strip()

    if not user_query:
        return "", "", "⚠️ Please provide a question via voice or text."

    user_query = normalize_text(user_query)

    # ------------------------------------------------------------------
    # Step 4 — Run the RAG chain
    # ------------------------------------------------------------------
    try:
        answer = invoke_chain(_state["rag_chain"], user_query)
    except Exception as exc:
        logger.exception("RAG chain error")
        return user_query, "", f"❌ Error generating answer: {exc}"

    status = get_rag_status(answer)
    final_answer = f"{answer}\n\n---\n*Powered by Gemini 2.5 Flash*"
    return user_query, final_answer, status


def build_interface() -> gr.Interface:
    """Create and return the Gradio Interface object."""
    return gr.Interface(
        fn=process_query,
        inputs=[
            gr.File(
                label="📄 Upload PDF Document",
                file_types=[".pdf"],
            ),
            gr.Audio(
                type="filepath",
                label="🎙️ Speak your question (optional)",
            ),
            gr.Textbox(
                label="✏️ Or type your question (optional)",
                placeholder="e.g., What is the main theme of this document?",
                lines=2,
            ),
        ],
        outputs=[
            gr.Textbox(label="Your Query"),
            gr.Textbox(label="RAG Answer", lines=8),
            gr.Textbox(label="Status"),
        ],
        title="🌐 Universal RAG with Voice/Text Input",
        description=(
            "Upload a PDF to build its knowledge base. "
            "Then ask questions in **any language** using voice or text. "
            "Answers are grounded in your document and powered by **Gemini 2.5 Flash**."
        ),
        allow_flagging="never",
    )


def launch() -> None:
    """
    Entry point: try to load an existing DB, then launch the Gradio UI.
    """
    startup_status = _try_load_existing_db()
    logger.info("Startup: %s", startup_status)

    iface = build_interface()
    iface.launch(
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        debug=GRADIO_DEBUG,
    )
