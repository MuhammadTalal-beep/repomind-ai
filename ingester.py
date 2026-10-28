"""Repository ingestion: validate a GitHub URL, shallow-clone it, extract source files.

Pipeline (and why each step exists):
    1. ``validate_github_url``  - fail fast on bad input before touching the network.
    2. ``clone_repository``     - a *shallow* (depth=1) clone keeps the demo fast.
    3. ``extract_files``        - walk the tree, pruning junk directories *in place*
                                  so we never descend into node_modules/.git/etc.
    4. ``ingest_repository``    - orchestrates 1-3 and always deletes the temp dir.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, Repo

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

IGNORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", "dist", "build",
        "out", "target", "vendor", "venv", ".venv", "env", ".env", ".idea",
        ".vscode", ".next", ".nuxt", ".cache", ".pytest_cache", ".mypy_cache",
        ".tox", ".gradle", "coverage", "htmlcov", "site-packages", "bower_components",
        "Pods", "DerivedData", "bin", "obj", "migrations", "__snapshots__",
    }
)

IGNORED_FILES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
        "Pipfile.lock", "Cargo.lock", "composer.lock", "go.sum", "uv.lock",
    }
)

# extension -> language key (keys match langchain's ``Language`` enum where possible)
LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".vue": "js", ".svelte": "js", ".ts": "ts", ".tsx": "ts", ".java": "java",
    ".kt": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".swift": "swift", ".scala": "scala", ".md": "markdown",
    ".html": "html", ".sh": "text", ".sql": "text", ".yaml": "text",
    ".yml": "text", ".toml": "text", ".json": "text",
}
SPECIAL_FILENAMES = frozenset({"Dockerfile", "Makefile", "requirements.txt"})

MAX_FILE_BYTES = 120_000      # skip huge generated/vendored files
MAX_TEXT_BYTES = 30_000       # config/docs are only useful when small
MAX_FILES = int(os.getenv("REPOMIND_MAX_FILES", "500"))

_GITHUB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?/?$"
)


class IngestionError(Exception):
    """Raised for any user-facing failure while fetching or reading a repository."""


@dataclass(frozen=True)
class RepoFile:
    """A single source file extracted from the repository."""

    path: str       # POSIX-style path relative to the repo root
    content: str
    language: str   # language key, e.g. "python", "js", "text"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def validate_github_url(url: str) -> str:
    """Validate a GitHub URL and return a canonical ``https://github.com/o/r.git``.

    Why: users paste URLs like ``.../repo/tree/main`` or with a trailing slash;
    normalising avoids confusing git errors later.

    Raises:
        IngestionError: if the URL is empty or not a public GitHub repo URL.
    """
    if not url or not url.strip():
        raise IngestionError("Please enter a GitHub repository URL.")
    match = _GITHUB_RE.match(url.strip())
    if not match:
        raise IngestionError(
            "That doesn't look like a GitHub repo URL. "
            "Expected something like https://github.com/owner/repo"
        )
    owner, repo = match.groups()
    return f"https://github.com/{owner}/{repo}.git"


def _force_remove(func, path, _exc_info):
    """``shutil.rmtree`` error hook: git marks pack files read-only (breaks Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        logger.warning("Could not remove %s", path)


def cleanup_directory(path: str | Path) -> None:
    """Delete a directory tree, ignoring errors (best-effort temp cleanup)."""
    shutil.rmtree(path, onerror=_force_remove)


def clone_repository(clean_url: str, destination: str | Path) -> None:
    """Shallow-clone ``clean_url`` into ``destination``.

    Why depth=1: we only need the latest snapshot, so history is wasted bandwidth.
    ``GIT_TERMINAL_PROMPT=0`` makes private/missing repos fail instead of hanging
    on a credentials prompt.

    Raises:
        IngestionError: with a human-readable message on any git failure.
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        Repo.clone_from(clean_url, str(destination), depth=1, single_branch=True, env=env)
    except GitCommandError as exc:
        text = f"{exc.stderr or ''} {exc}".lower()
        if any(s in text for s in ("not found", "could not read username", "authentication")):
            raise IngestionError(
                "Repository not found, or it is private. Only public repos are supported."
            ) from exc
        raise IngestionError(f"Git clone failed: {(exc.stderr or str(exc)).strip()[:300]}") from exc
    except Exception as exc:  # e.g. git binary missing
        raise IngestionError(f"Could not run git: {exc}") from exc


def _detect_language(path: Path) -> str | None:
    """Return the language key for a path, or ``None`` if the file should be skipped."""
    name = path.name
    if name in IGNORED_FILES or name.endswith((".min.js", ".min.css", ".map")):
        return None
    if name in SPECIAL_FILENAMES:
        return "text"
    return LANGUAGE_BY_EXT.get(path.suffix.lower())


def _read_text(path: Path, language: str) -> str | None:
    """Read a file as text; return ``None`` for binary, oversized, empty or minified files."""
    try:
        size = path.stat().st_size
        limit = MAX_TEXT_BYTES if language == "text" else MAX_FILE_BYTES
        if size == 0 or size > limit:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:1024]:  # NUL byte => binary
        return None
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    # Minified bundles have gigantic average line length and are useless for RAG.
    if len(text) / (text.count("\n") + 1) > 400:
        return None
    return text


def _priority(file: RepoFile) -> tuple[int, int, str]:
    """Sort key: README first, then code, then docs/config; shallow paths before deep."""
    is_readme = file.path.lower().rsplit("/", 1)[-1].startswith("readme")
    rank = 0 if is_readme else (2 if file.language in ("text", "markdown") else 1)
    return rank, file.path.count("/"), file.path


def extract_files(root: str | Path, max_files: int = MAX_FILES) -> list[RepoFile]:
    """Walk ``root`` and return relevant text files, most important first.

    Why prune ``dirnames`` in place: ``os.walk`` then never descends into ignored
    directories, which is dramatically faster than filtering afterwards.
    """
    root_path = Path(root)
    collected: list[RepoFile] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            language = _detect_language(file_path)
            if language is None:
                continue
            content = _read_text(file_path, language)
            if content is None:
                continue
            rel = file_path.relative_to(root_path).as_posix()
            collected.append(RepoFile(path=rel, content=content, language=language))
    collected.sort(key=_priority)
    return collected[:max_files]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def ingest_repository(url: str) -> tuple[list[RepoFile], str]:
    """Clone a public GitHub repo and return ``(files, "owner/repo")``.

    The temporary clone is always removed, even on failure.

    Raises:
        IngestionError: for invalid URLs, unreachable repos, or repos with no readable code.
    """
    clean_url = validate_github_url(url)
    owner, repo = _GITHUB_RE.match(url.strip()).groups()  # type: ignore[union-attr]
    tmp_dir = tempfile.mkdtemp(prefix="repomind_")
    try:
        clone_repository(clean_url, tmp_dir)
        files = extract_files(tmp_dir)
    finally:
        cleanup_directory(tmp_dir)
    if not files:
        raise IngestionError("No readable source files were found in this repository.")
    return files, f"{owner}/{repo.removesuffix('.git')}"
