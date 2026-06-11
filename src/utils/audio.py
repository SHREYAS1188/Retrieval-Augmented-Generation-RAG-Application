"""
src/utils/audio.py
===================
Audio transcription using OpenAI Whisper.
"""

import logging
from pathlib import Path

import whisper

from config.settings import WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)

# Module-level cache — Whisper model is heavy; load it once per process
_whisper_model: whisper.Whisper | None = None


def get_whisper_model() -> whisper.Whisper:
    """Return (or lazily load) the shared Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str | Path) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, etc.).

    Returns:
        Transcribed text string.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If transcription fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model()
    logger.info("Transcribing audio: %s", audio_path.name)

    try:
        result = model.transcribe(str(audio_path))
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    text: str = result.get("text", "").strip()
    logger.info("Transcription complete. Text: %.80s …", text)
    return text
