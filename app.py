"""
app.py
=======
Entry point for the Multilingual RAG application.

Usage:
    python app.py

For Hugging Face Spaces, this file must be named `app.py` and live at the
root of the repository. Spaces will automatically run it on startup.
"""

import logging
import sys

# ---------------------------------------------------------------------------
# Logging — set up before any imports that might log at module level
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Multilingual RAG application …")

    # Import here so logging is configured before any module-level side effects
    from src.ui.gradio_app import launch  # noqa: E402

    launch()
