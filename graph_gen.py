"""Build a Mermaid ``graph TD`` of file-to-file dependencies from import statements.

Pipeline (and why each step exists):
    1. ``collect_dependencies`` - parse imports per language (Python via ``ast``,
                                  JS/TS via regex, Java/Kotlin via regex) and keep only
                                  imports that resolve to *files inside the repo*;
                                  external packages would just add noise.
    2. ``build_mermaid_graph``  - keep the most-connected files (diagram stays readable),
                                  group them into folder subgraphs, and highlight entry
                                  points and hubs so the architecture is obvious at a glance.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from collections import Counter, defaultdict

from src.ingester import RepoFile

logger = logging.getLogger(__name__)

MAX_NODES = 60
JS_EXTS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte")
JS_RESOLVE_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".json")
JVM_EXTS = (".java", ".kt")

PY_FALLBACK_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)
JS_IMPORT_RE = re.compile(
    r"""(?:import|export)\s+(?:[\w*\s{},$]+?\s+from\s+)?['"]([^'"\n]+)['"]"""
)
JS_REQUIRE_RE = re.compile(r"""require\(\s*['"]([^'"\n]+)['"]\s*\)""")
JS_DYNAMIC_RE = re.compile(r"""import\(\s*['"]([^'"\n]+)['"]\s*\)""")
JVM_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)", re.M)


# --------------------------------------------------------------------------- #
# Shared lookup helpers
# --------------------------------------------------------------------------- #

def _build_suffix_index(paths: set[str], exts: tuple[str, ...]) -> dict[str, set[str]]:
    """Map every dotted path suffix to matching files.

    ``src/pkg/mod.py`` is registered as ``src.pkg.mod``, ``pkg.mod`` and ``mod``.
    Why: repos use different import roots (``src/`` layouts, monorepos); suffix
    matching resolves them without needing to know the project's configuration.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        stem, ext = posixpath.splitext(path)
        if ext not in exts:
            continue
        parts = stem.split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            index[".".join(parts[i:])].add(path)
    return index


def _lookup(index: dict[str, set[str]], dotted: str) -> str | None:
    """Return the best match for a dotted name (shallowest path wins on ties)."""
    candidates = index.get(dotted)
    return min(candidates, key=lambda p: (len(p), p)) if candidates else None


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

def _resolve_py_module(importer, level, module, names, paths, index) -> set[str]:
    """Resolve one Python import statement to repo files (handles relative imports)."""
    found: set[str] = set()
    module_parts = module.split(".") if module else []

    if level > 0:
        package = importer.split("/")[:-1]
        if level - 1 > len(package):
            return found
        base = package[: len(package) - (level - 1)] + module_parts

        def lookup(parts: list[str]) -> str | None:
            joined = "/".join(parts)
            for candidate in (f"{joined}.py", f"{joined}/__init__.py"):
                if candidate in paths:
                    return candidate
            return None
    else:
        base = module_parts

        def lookup(parts: list[str]) -> str | None:
            return _lookup(index, ".".join(parts))

    for name in names:  # `from pkg import submodule` -> edge to the submodule file
        if name != "*":
            hit = lookup(base + [name])
            if hit:
                found.add(hit)
    if base and not found:  # otherwise fall back to the package/module itself
        hit = lookup(base)
        if hit:
            found.add(hit)
    return found


def _python_imports(file: RepoFile, paths: set[str], index: dict[str, set[str]]) -> set[str]:
    """Extract internal imports from a Python file using ``ast`` (regex if it won't parse)."""
    targets: set[str] = set()
    try:
        tree = ast.parse(file.content)
    except (SyntaxError, ValueError):
        for match in PY_FALLBACK_RE.finditer(file.content):
            raw = match.group(1) or match.group(2)
            level = len(raw) - len(raw.lstrip("."))
            targets |= _resolve_py_module(file.path, level, raw.lstrip("."), [], paths, index)
        return targets

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets |= _resolve_py_module(file.path, 0, alias.name, [], paths, index)
        elif isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            targets |= _resolve_py_module(file.path, node.level, node.module or "", names, paths, index)
    return targets


# --------------------------------------------------------------------------- #
# JavaScript / TypeScript
# --------------------------------------------------------------------------- #

def _resolve_js_spec(spec: str, importer: str, paths: set[str]) -> str | None:
    """Resolve a JS/TS module specifier to a repo file (relative paths and ``@/`` aliases)."""
    if spec.startswith("."):
        bases = [posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))]
    elif spec.startswith(("@/", "~/")):
        bases = [f"src/{spec[2:]}", spec[2:]]
    else:
        return None  # bare specifier => external package

    for base in bases:
        if base.startswith(".."):
            continue
        stem, ext = posixpath.splitext(base)
        # TS projects import "./foo.js" while the file on disk is foo.ts
        roots = [base] + ([stem] if ext in (".js", ".jsx", ".mjs", ".cjs") else [])
        for root in roots:
            if root in paths:
                return root
            for e in JS_RESOLVE_EXTS:
                if f"{root}{e}" in paths:
                    return f"{root}{e}"
                if f"{root}/index{e}" in paths:
                    return f"{root}/index{e}"
    return None


def _js_imports(file: RepoFile, paths: set[str]) -> set[str]:
    """Extract internal imports from JS/TS (``import``, ``export ... from``, ``require``, ``import()``)."""
    specs: set[str] = set()
    for regex in (JS_IMPORT_RE, JS_REQUIRE_RE, JS_DYNAMIC_RE):
        specs.update(m.group(1) for m in regex.finditer(file.content))
    resolved = (_resolve_js_spec(s, file.path, paths) for s in specs)
    return {r for r in resolved if r}


# --------------------------------------------------------------------------- #
# Java / Kotlin
# --------------------------------------------------------------------------- #

def _jvm_imports(file: RepoFile, index: dict[str, set[str]]) -> set[str]:
    """Extract internal imports from Java/Kotlin by matching ``com.foo.Bar`` to file paths."""
    targets: set[str] = set()
    for match in JVM_IMPORT_RE.finditer(file.content):
        parts = match.group(1).split(".")
        while len(parts) >= 2:  # strip trailing member/inner-class names until a file matches
            hit = _lookup(index, ".".join(parts))
            if hit:
                targets.add(hit)
                break
            parts.pop()
    return targets


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def collect_dependencies(files: list[RepoFile]) -> tuple[dict[str, set[str]], int, int]:
    """Return ``(dependencies, files_parsed, files_failed)``; ``dependencies[src] = {targets}``."""
    paths = {f.path for f in files}
    py_index = _build_suffix_index(paths, (".py",))
    jvm_index = _build_suffix_index(paths, JVM_EXTS)

    deps: dict[str, set[str]] = {}
    parsed = failed = 0
    for file in files:
        ext = posixpath.splitext(file.path)[1].lower()
        try:
            if ext == ".py":
                targets = _python_imports(file, paths, py_index)
            elif ext in JS_EXTS:
                targets = _js_imports(file, paths)
            elif ext in JVM_EXTS:
                targets = _jvm_imports(file, jvm_index)
            else:
                continue
            parsed += 1
        except Exception:  # one weird file must never break the whole diagram
            logger.warning("Could not parse imports in %s", file.path, exc_info=True)
            failed += 1
            continue
        targets.discard(file.path)
        if targets:
            deps[file.path] = targets
    return deps, parsed, failed


def _labels(nodes: list[str]) -> dict[str, str]:
    """Short node labels: basename, or ``parent/basename`` when basenames collide."""
    counts = Counter(posixpath.basename(p) for p in nodes)
    labels = {}
    for path in nodes:
        base = posixpath.basename(path)
        label = base if counts[base] == 1 else "/".join(path.split("/")[-2:])
        labels[path] = label.replace('"', "'")
    return labels


def build_mermaid_graph(files: list[RepoFile]) -> tuple[str, dict]:
    """Generate a Mermaid ``graph TD`` string and stats for the repository.

    Returns:
        ``(mermaid_code, stats)`` where stats has keys
        ``nodes, edges, files_parsed, files_failed, truncated``.
    """
    deps, parsed, failed = collect_dependencies(files)
    edges = [(s, t) for s, targets in deps.items() for t in sorted(targets)]
    total_edges = len(edges)

    degree: Counter[str] = Counter()
    for s, t in edges:
        degree[s] += 1
        degree[t] += 1

    truncated = False
    if edges:
        ranked = [p for p, _ in degree.most_common(MAX_NODES)]
        truncated = len(degree) > MAX_NODES
        node_set = set(ranked)
        edges = [(s, t) for s, t in edges if s in node_set and t in node_set]
        nodes = sorted(node_set)
    else:  # no internal imports found: still show something useful
        nodes = [f.path for f in files if posixpath.splitext(f.path)[1].lower() in JS_EXTS + JVM_EXTS + (".py",)]
        nodes = sorted(nodes[:MAX_NODES])

    stats = {
        "nodes": len(nodes), "edges": len(edges), "files_parsed": parsed,
        "files_failed": failed, "truncated": truncated, "total_edges": total_edges,
    }
    if not nodes:
        return 'graph TD\n    empty["No source files with imports found"]', stats

    ids = {path: f"n{i}" for i, path in enumerate(nodes)}
    labels = _labels(nodes)
    indeg = Counter(t for _, t in edges)
    outdeg = Counter(s for s, _ in edges)

    groups: dict[str, list[str]] = defaultdict(list)
    for path in nodes:
        groups[path.split("/")[0] if "/" in path else "(root)"].append(path)

    lines = ["graph TD"]
    for g_idx, (group, members) in enumerate(sorted(groups.items())):
        lines.append(f'    subgraph g{g_idx}["{group.replace(chr(34), chr(39))}"]')
        lines.extend(f'        {ids[p]}["{labels[p]}"]' for p in members)
        lines.append("    end")
    lines.extend(f"    {ids[s]} --> {ids[t]}" for s, t in edges)

    hubs = [p for p, _ in indeg.most_common(3)]
    entries = [p for p in nodes if indeg[p] == 0 and outdeg[p] > 0 and p not in hubs]
    lines.append("    classDef entry fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b")
    lines.append("    classDef hub fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d")
    if entries:
        lines.append(f"    class {','.join(ids[p] for p in entries)} entry")
    if hubs:
        lines.append(f"    class {','.join(ids[p] for p in hubs)} hub")
    return "\n".join(lines), stats
