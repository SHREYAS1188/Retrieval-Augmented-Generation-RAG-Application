# Architecture — Multilingual RAG Pipeline

## Overview

This application implements a **Retrieval-Augmented Generation (RAG)** pipeline
that accepts PDF documents as a knowledge base and answers user questions using
semantic search and a Gemini LLM — with full multilingual and voice support.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       Gradio UI (app.py)                         │
│                                                                  │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────────────┐  │
│  │ PDF File │   │ Audio (.wav)  │   │ Text Query             │  │
│  └────┬─────┘   └───────┬───────┘   └────────────┬───────────┘  │
└───────┼─────────────────┼────────────────────────┼──────────────┘
        │                 │                         │
        ▼                 ▼                         │
┌───────────────┐  ┌──────────────┐                 │
│  PDF Loader   │  │   Whisper    │                 │
│ pymupdf4llm   │  │  (STT)       │                 │
└───────┬───────┘  └──────┬───────┘                 │
        │                 │                         │
        ▼                 └─────────────────────────┘
┌───────────────┐                         │
│   Chunker     │              ┌──────────▼──────────┐
│ RecursiveText │              │   Query Normaliser  │
│   Splitter    │              │  (NFC Unicode)      │
└───────┬───────┘              └──────────┬──────────┘
        │                                 │
        ▼                                 ▼
┌───────────────┐         ┌───────────────────────────┐
│  Lang Detect  │         │        Retriever          │
│  + NFC norm   │         │   ChromaDB similarity_    │
└───────┬───────┘         │   search (k=5)            │
        │                 └──────────┬────────────────┘
        ▼                            │
┌───────────────┐                    │
│  HuggingFace  │◄───────────────────┘
│  Embeddings   │
│  BAAI/bge-m3  │
└───────┬───────┘
        │
        ▼
┌───────────────┐         ┌────────────────────────────┐
│  ChromaDB     │         │     RAG Prompt             │
│  Vector Store │         │  (multilingual, grounded)  │
└───────────────┘         └──────────┬─────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │  Gemini 2.5 Flash (LLM)    │
                          └──────────┬─────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────────┐
                          │       Answer (str)         │
                          └────────────────────────────┘
```

---

## Component Breakdown

### 1. Ingestion (`src/ingestion/`)

| File | Purpose |
|---|---|
| `pdf_loader.py` | PDF → Markdown via `pymupdf4llm`. Markdown preserves headers, tables, and lists — critical for accurate chunking. |
| `chunker.py` | Splits Markdown into 1000-char overlapping chunks using `RecursiveCharacterTextSplitter`. Each chunk gets NFC-normalized and language-tagged via `langdetect`. |

### 2. Retrieval (`src/retrieval/`)

| File | Purpose |
|---|---|
| `vector_store.py` | Builds, persists, loads, and exposes a ChromaDB vector store. Uses `BAAI/bge-m3` — a multilingual embedding model supporting 100+ languages. |

### 3. LLM (`src/llm/`)

| File | Purpose |
|---|---|
| `prompts.py` | Defines the multilingual RAG system prompt. Key constraint: "Answer ONLY from context" + "Respond in the user's language". |
| `rag_chain.py` | Assembles the LCEL chain: retriever → format_docs → prompt → LLM → StrOutputParser. |

### 4. Utils (`src/utils/`)

| File | Purpose |
|---|---|
| `audio.py` | Whisper speech-to-text. Model is cached in memory across requests. |
| `text_utils.py` | NFC normalization, RAG answer detection heuristic, status label generator. |

### 5. UI (`src/ui/`)

| File | Purpose |
|---|---|
| `gradio_app.py` | Gradio interface. Module-level `_state` dict holds the vector store and chain across requests (Gradio doesn't use class state). |

---

## Multilingual Design Decisions

- **`BAAI/bge-m3`** embeds text in 100+ languages into the same vector space, so a Kannada query can match an English chunk (and vice versa).
- **NFC normalization** ensures consistent Unicode representation across Indic scripts.
- The **RAG prompt** explicitly instructs the LLM to mirror the user's language, enabling seamless code-switching.

---

## Deployment Modes

### Local
```bash
python app.py  # ChromaDB persisted to ./chroma_db
```

### Hugging Face Spaces
- Set `GOOGLE_API_KEY` as a Repository Secret
- ChromaDB runs **in-memory** (no `persist_directory`) to avoid filesystem issues
- `GRADIO_SHARE=false` (Spaces provides its own public URL)

---

## Extension Points

- **Swap embeddings**: Change `EMBEDDING_MODEL` in `config/settings.py` to try different HuggingFace models.
- **Swap LLM**: Change `LLM_MODEL` to any Gemini variant or adapt `rag_chain.py` for OpenAI / Anthropic.
- **Add BM25 hybrid retrieval**: Replace the ChromaDB retriever with an `EnsembleRetriever`.
- **Multi-PDF support**: Extend `build_vector_store` to accept a list of PDFs and merge documents.
