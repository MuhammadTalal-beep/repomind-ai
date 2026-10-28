"""Chunk repository files and embed them into an in-memory ChromaDB vector store.

Pipeline (and why each step exists):
    1. ``chunk_files``           - language-aware splitting keeps functions/classes
                                   together, which makes retrieved context coherent.
    2. ``build_vector_store``    - embeds chunks in small batches with retry/backoff so
                                   free-tier Gemini rate limits don't kill a live demo.
    3. ``summarize_repository``  - a compact overview injected into every prompt so the
                                   model can answer "what is this project?" questions.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter
from typing import Callable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.ingester import RepoFile

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CHUNKS = int(os.getenv("REPOMIND_MAX_CHUNKS", "400"))   # caps embedding time/quota
MAX_CHUNKS_PER_FILE = 8                                      # keeps coverage broad
BATCH_SIZE = 50
MAX_RETRIES = 4


class IndexingError(Exception):
    """Raised for user-facing failures while embedding or storing chunks."""


ProgressCallback = Callable[[int, int], None]


def _splitter_for(language: str, cache: dict[str, RecursiveCharacterTextSplitter]):
    """Return (and cache) a splitter tuned to ``language``, falling back to generic."""
    if language not in cache:
        try:
            cache[language] = RecursiveCharacterTextSplitter.from_language(
                Language(language), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
        except ValueError:  # "text" and anything not in langchain's Language enum
            cache[language] = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
    return cache[language]


def chunk_files(files: list[RepoFile], max_chunks: int = MAX_CHUNKS) -> list[Document]:
    """Split files into LangChain ``Document`` chunks.

    Each chunk is prefixed with its file path: embeddings then capture *where* the
    code lives, and the LLM can cite the file without extra metadata plumbing.
    Files arrive priority-sorted, so truncating at ``max_chunks`` drops the least
    important content first.
    """
    cache: dict[str, RecursiveCharacterTextSplitter] = {}
    documents: list[Document] = []
    for file in files:
        if len(documents) >= max_chunks:
            break
        try:
            pieces = _splitter_for(file.language, cache).split_text(file.content)
        except Exception:  # unparseable/odd file: skip it, never crash the run
            logger.warning("Skipping unsplittable file %s", file.path, exc_info=True)
            continue
        for idx, piece in enumerate(pieces[:MAX_CHUNKS_PER_FILE]):
            documents.append(
                Document(
                    page_content=f"File: {file.path}\n\n{piece}",
                    metadata={"source": file.path, "language": file.language, "chunk": idx},
                )
            )
    return documents[:max_chunks]


def _friendly_error(exc: Exception) -> str:
    """Translate raw Google API errors into short, actionable messages."""
    text = str(exc).lower()
    if "api key not valid" in text or "api_key_invalid" in text or "permission_denied" in text:
        return "The Gemini API key was rejected. Please check it and try again."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "Gemini rate limit/quota hit. Wait a minute or use a key with more quota."
    return f"Embedding failed: {str(exc)[:300]}"


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("429", "quota", "resource_exhausted", "503", "unavailable", "timeout"))


def build_vector_store(
    files: list[RepoFile],
    api_key: str,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Chroma, int]:
    """Embed repository chunks into a fresh in-memory Chroma collection.

    Returns:
        ``(vector_store, number_of_chunks)``.

    Raises:
        IndexingError: on missing key, empty input, or unrecoverable embedding errors.
    """
    if not api_key or not api_key.strip():
        raise IndexingError("A Gemini API key is required.")
    documents = chunk_files(files)
    if not documents:
        raise IndexingError("Nothing to index: no chunks were produced.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key.strip())
    # No persist_directory => ephemeral, in-memory client. A unique collection name
    # avoids collisions when the user analyses several repos in one session.
    store = Chroma(
        collection_name=f"repomind_{uuid.uuid4().hex[:12]}",
        embedding_function=embeddings,
    )

    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                store.add_documents(batch)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES - 1 or not _is_retryable(exc):
                    raise IndexingError(_friendly_error(exc)) from exc
                time.sleep(2 ** (attempt + 1))  # exponential backoff: 2s, 4s, 8s
        if progress_cb:
            progress_cb(min(start + BATCH_SIZE, total), total)
    return store, total


def summarize_repository(files: list[RepoFile], repo_name: str) -> str:
    """Build a compact text overview (languages, top-level layout, key files)."""
    langs = Counter(f.language for f in files if f.language not in ("text", "markdown"))
    lang_line = ", ".join(f"{lang} ({n})" for lang, n in langs.most_common(5)) or "n/a"
    top_level = Counter(f.path.split("/")[0] if "/" in f.path else "(root files)" for f in files)
    layout = ", ".join(f"{name} [{n}]" for name, n in top_level.most_common(12))
    key_files = [
        f.path for f in files
        if f.path.lower().rsplit("/", 1)[-1] in (
            "readme.md", "package.json", "pyproject.toml", "requirements.txt",
            "main.py", "app.py", "index.js", "index.ts", "dockerfile",
        )
    ][:8]
    return (
        f"Repository: {repo_name}\n"
        f"Files indexed: {len(files)}\n"
        f"Languages: {lang_line}\n"
        f"Top-level layout: {layout}\n"
        f"Key files: {', '.join(key_files) or 'n/a'}"
    )
