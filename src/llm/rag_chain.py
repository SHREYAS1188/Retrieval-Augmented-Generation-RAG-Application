"""
src/llm/rag_chain.py
=====================
Builds and runs the LangChain LCEL RAG chain.

Chain flow:
    user question
        ↓
    retriever  →  format_docs  →  context string
        ↓                               ↓
    RunnablePassthrough  →  question ───┘
        ↓
    ChatPromptTemplate
        ↓
    ChatGoogleGenerativeAI (Gemini)
        ↓
    StrOutputParser
        ↓
    answer string
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import LLM_MODEL, LLM_TEMPERATURE
from src.llm.prompts import MULTILINGUAL_RAG_PROMPT

logger = logging.getLogger(__name__)


def _format_docs(docs) -> str:
    """Join document page content with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever: VectorStoreRetriever) -> Runnable:
    """
    Assemble the LCEL RAG chain.

    Args:
        retriever: A LangChain VectorStoreRetriever to fetch context documents.

    Returns:
        A compiled LCEL Runnable that accepts a question string and returns
        an answer string.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    chain: Runnable = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | MULTILINGUAL_RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain built with model=%s, temperature=%s", LLM_MODEL, LLM_TEMPERATURE)
    return chain


def invoke_chain(chain: Runnable, question: str) -> str:
    """
    Invoke the RAG chain with a user question.

    Args:
        chain: A compiled LCEL Runnable (from ``build_rag_chain``).
        question: The user's question (any language).

    Returns:
        The LLM's grounded answer string.

    Raises:
        RuntimeError: On chain invocation failure.
    """
    logger.info("Invoking RAG chain. Question: %.80s …", question)
    try:
        answer: str = chain.invoke(question)
    except Exception as exc:
        raise RuntimeError(f"RAG chain invocation failed: {exc}") from exc

    logger.info("Chain returned answer of length %d.", len(answer))
    return answer
