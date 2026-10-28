"""Context-aware RAG question answering over the repository vector store (Gemini).

Pipeline per question (and why each step exists):
    1. ``retrieve``        - MMR search returns relevant *and* diverse chunks, so answers
                             aren't dominated by five near-duplicate snippets.
    2. ``_format_context`` - label every chunk with its file path so the model can cite it.
    3. ``stream_answer``   - fill the prompt template and stream tokens back for instant UX.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Gemma 4 served through the Gemini API (same API key, same langchain-google-genai client).
# Options: "gemma-4-26b-a4b-it" (fast, ~4B active params) or "gemma-4-31b-it" (higher quality).
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")

SYSTEM_PROMPT = """You are RepoMind AI, a principal software engineer who explains codebases.

Rules:
- Answer ONLY from the repository overview and retrieved code context below.
- Cite the files you rely on using backticks, e.g. `src/app.py`.
- If the context does not contain the answer, say so plainly and suggest which files or
  folders are most likely to hold it. Never invent files, functions or behaviour.
- Be concise and concrete. Use markdown; put code in fenced blocks with a language tag.
- For architecture questions, explain responsibilities and how modules interact.

=== REPOSITORY OVERVIEW ===
{overview}"""

HUMAN_PROMPT = """Conversation so far:
{history}

=== RETRIEVED CONTEXT ===
{context}

=== QUESTION ===
{question}"""

# Instructions and context go in ONE user message: Gemma endpoints on the Gemini API have
# historically rejected separate system/developer instructions, and this works everywhere.
PROMPT = ChatPromptTemplate.from_messages([("human", SYSTEM_PROMPT + "\n\n" + HUMAN_PROMPT)])


def _chunk_text(content) -> str:
    """Normalise a streamed chunk's content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _friendly_error(exc: Exception) -> str:
    """Convert raw API exceptions into a short message shown inside the chat."""
    text = str(exc).lower()
    if "api key not valid" in text or "api_key_invalid" in text:
        return "The Gemini API key was rejected. Please check it in the sidebar."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "Gemma rate limit reached (free-tier caps are low). Wait a few seconds and ask again."
    if "not found" in text and "model" in text:
        return f"Model '{DEFAULT_MODEL}' is unavailable for this key. Set GEMINI_MODEL to another Gemini model."
    return f"Something went wrong while contacting Gemini: {str(exc)[:250]}"


class RepoAgent:
    """RAG agent bound to one repository's vector store."""

    def __init__(self, vector_store, api_key: str, overview: str, model: str = DEFAULT_MODEL, k: int = 6):
        """Create the agent.

        Raises:
            ValueError: if ``api_key`` is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("A Gemini API key is required.")
        self.vector_store = vector_store
        self.overview = overview
        self.k = k
        self.last_sources: list[str] = []
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key.strip(),
            temperature=0.2,  # low: we want faithful answers, not creative ones
        )

    def retrieve(self, question: str, history: list[dict] | None = None) -> list[Document]:
        """Fetch relevant chunks. Short follow-ups ("why?") borrow the previous question."""
        query = question
        if history and len(question.split()) < 6:
            previous = [m["content"] for m in history if m["role"] == "user"]
            if previous:
                query = f"{previous[-1]} {question}"
        return self.vector_store.max_marginal_relevance_search(query, k=self.k, fetch_k=self.k * 4)

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        if not docs:
            return "(no relevant chunks were retrieved)"
        return "\n\n---\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)

    @staticmethod
    def _format_history(history: list[dict] | None, max_turns: int = 6) -> str:
        if not history:
            return "(none)"
        recent = history[-max_turns:]
        return "\n".join(f"{m['role'].capitalize()}: {m['content'][:600]}" for m in recent)

    def stream_answer(self, question: str, history: list[dict] | None = None) -> Iterator[str]:
        """Yield the answer token-by-token; errors are yielded as a readable message."""
        self.last_sources = []
        if not question or not question.strip():
            yield "Please ask a question about the repository."
            return
        try:
            docs = self.retrieve(question, history)
            self.last_sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))
            messages = PROMPT.format_messages(
                overview=self.overview,
                history=self._format_history(history),
                context=self._format_context(docs),
                question=question,
            )
            emitted = False
            for chunk in self.llm.stream(messages):
                text = _chunk_text(chunk.content)
                if text:
                    emitted = True
                    yield text
            if not emitted:
                yield "The model returned an empty response. Try rephrasing your question."
        except Exception as exc:
            logger.exception("RAG answer failed")
            yield f"⚠️ {_friendly_error(exc)}"
