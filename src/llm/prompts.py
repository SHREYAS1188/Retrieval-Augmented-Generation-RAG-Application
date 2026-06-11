"""
src/llm/prompts.py
===================
Prompt templates for the RAG pipeline.

Design principles:
- Answer ONLY from the provided context (anti-hallucination).
- Mirror the language of the user's question (multilingual support).
- Clear instruction hierarchy: system role → context → human question.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Multilingual RAG Prompt (default)
# ---------------------------------------------------------------------------
MULTILINGUAL_RAG_SYSTEM = (
    "You are a multilingual assistant.\n\n"
    "Answer ONLY using the provided context.\n\n"
    "Always answer in the SAME language as the user's question.\n\n"
    "If the answer is not present in the context, "
    "say you do not have enough information.\n\n"
    "Context:\n{context}"
)

MULTILINGUAL_RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", MULTILINGUAL_RAG_SYSTEM),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# English-only RAG Prompt (single-language fallback)
# ---------------------------------------------------------------------------
ENGLISH_RAG_SYSTEM = (
    "You are a helpful assistant answering questions based strictly "
    "on the provided context.\n"
    "If you do not know the answer or if it's not explicitly found in the "
    "context, state that you do not have enough information. "
    "Do not attempt to guess.\n\n"
    "--- Context ---\n{context}"
)

ENGLISH_RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", ENGLISH_RAG_SYSTEM),
        ("human", "{question}"),
    ]
)
