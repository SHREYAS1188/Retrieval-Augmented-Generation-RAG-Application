[![Hugging Face Space](https://shields.io)](https://huggingface.co/spaces/shreyasp11111111111111/rag2)



# 🌐 Multilingual RAG with Voice Input

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline with multilingual support and voice input, powered by Google Gemini 2.5, LangChain, ChromaDB, and OpenAI Whisper.

---

## ✨ Features

- 📄 **PDF Ingestion** — Convert PDFs to Markdown and chunk them intelligently
- 🌍 **Multilingual Support** — Responds in the same language as the user's question (uses `BAAI/bge-m3` multilingual embeddings)
- 🎙️ **Voice Input** — Ask questions via audio using OpenAI Whisper transcription
- 💬 **Text Input** — Standard text query support
- 🔍 **Semantic Search** — ChromaDB vector store with cosine similarity retrieval
- 🤖 **Gemini 2.5 Flash LLM** — Grounded, context-only answers with anti-hallucination prompts
- 🖥️ **Gradio UI** — Clean web interface for end users
- 🚀 **Hugging Face Spaces Ready** — Deploy in minutes

---

## 📁 Project Structure

```
multilingual-rag/
├── src/
│   ├── ingestion/          # PDF loading, chunking, vector DB creation
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   └── chunker.py
│   ├── retrieval/          # Vector store management and retriever
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── llm/                # LLM setup, prompts, and RAG chain
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── rag_chain.py
│   ├── ui/                 # Gradio UI definition
│   │   ├── __init__.py
│   │   └── gradio_app.py
│   └── utils/              # Shared helpers (audio, text normalization)
│       ├── __init__.py
│       ├── audio.py
│       └── text_utils.py
├── config/
│   └── settings.py         # Centralised config (model names, chunk sizes, etc.)
├── data/
│   └── sample_pdfs/        # Drop sample PDFs here for testing
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_rag_chain.py
├── scripts/
│   └── ingest_pdf.py       # CLI script: pre-ingest a PDF into vector DB
├── docs/
│   └── architecture.md     # System design notes
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── app.py                  # Entry point — launches Gradio UI
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/multilingual-rag.git
cd multilingual-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `openai-whisper` requires `ffmpeg`. Install it with:
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - macOS: `brew install ffmpeg`
> - Windows: Download from [ffmpeg.org](https://ffmpeg.org/)

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 5. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## ⚙️ Configuration

All tuneable parameters live in `config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature (lower = more grounded) |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Top-K documents retrieved |
| `WHISPER_MODEL` | `base` | Whisper model size |

---

## 🌍 Multilingual Support

This pipeline uses `BAAI/bge-m3`, a multilingual embedding model that supports 100+ languages including **Kannada**, Hindi, Tamil, English, and more.

The RAG prompt instructs the LLM to:
- Answer **only** from the retrieved context
- Respond in the **same language** as the user's question

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) with **Gradio** SDK
2. Upload `app.py`, `requirements.txt`, and the `src/` + `config/` folders
3. Add `GOOGLE_API_KEY` as a **Repository Secret** in Space Settings
4. Your app will auto-build and go live!

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 📖 Architecture

See [`docs/architecture.md`](docs/architecture.md) for a detailed system design walkthrough.

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| PDF Parsing | `pymupdf4llm` |
| Text Splitting | `langchain_text_splitters` |
| Embeddings | `BAAI/bge-m3` (HuggingFace) |
| Vector Store | `ChromaDB` |
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangChain LCEL |
| Speech-to-Text | OpenAI Whisper |
| UI | Gradio |

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first for major changes.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
