"""RepoMind AI - Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.agent import RepoAgent
from src.graph_gen import build_mermaid_graph
from src.indexer import IndexingError, build_vector_store, summarize_repository
from src.ingester import IngestionError, ingest_repository, validate_github_url

load_dotenv()
st.set_page_config(page_title="RepoMind AI", page_icon="🧠", layout="wide")

EXAMPLE_QUESTIONS = [
    "What does this project do?",
    "Explain the overall architecture",
    "Where is the entry point?",
]

# Mermaid is rendered inside an iframe component. svg-pan-zoom gives drag + wheel zoom.
MERMAID_HTML = """
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font-family: system-ui, sans-serif; }
  #bar { display: flex; gap: 6px; padding: 6px 0; }
  #bar button { border: 1px solid #9ca3af; background: #fff; color: #111; border-radius: 6px;
                padding: 4px 10px; cursor: pointer; font-size: 13px; }
  #stage { height: calc(__HEIGHT__px - 44px); border: 1px solid #d1d5db; border-radius: 8px;
           background: #fff; overflow: hidden; }
  pre { color: #b91c1c; padding: 12px; white-space: pre-wrap; }
</style>
<div id="bar">
  <button id="zin">＋ Zoom in</button><button id="zout">－ Zoom out</button>
  <button id="reset">⟲ Reset</button>
  <span style="font-size:12px;opacity:.7;align-self:center">Drag to pan · scroll to zoom</span>
</div>
<div id="stage">Rendering…</div>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
  const CODE = __CODE__;
  const stage = document.getElementById("stage");
  mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose",
    maxTextSize: 900000, maxEdges: 2000, flowchart: { useMaxWidth: false, curve: "basis" } });
  try {
    const { svg } = await mermaid.render("repomind-graph", CODE);
    stage.innerHTML = svg;
    const el = stage.querySelector("svg");
    el.setAttribute("width", "100%"); el.setAttribute("height", "100%");
    el.style.maxWidth = "none";
    const pz = svgPanZoom(el, { zoomEnabled: true, controlIconsEnabled: false, fit: true,
                                center: true, minZoom: 0.2, maxZoom: 20 });
    document.getElementById("zin").onclick = () => pz.zoomIn();
    document.getElementById("zout").onclick = () => pz.zoomOut();
    document.getElementById("reset").onclick = () => { pz.resetZoom(); pz.fit(); pz.center(); };
  } catch (err) {
    stage.innerHTML = "<pre>Could not render diagram:\\n" + String(err) + "</pre>";
  }
</script>
"""


def render_mermaid(code: str, height: int = 700) -> None:
    """Embed an interactive Mermaid diagram (pan/zoom) in the page."""
    safe_code = json.dumps(code).replace("</", "<\\/")  # prevent breaking out of <script>
    html = MERMAID_HTML.replace("__CODE__", safe_code).replace("__HEIGHT__", str(height))
    components.html(html, height=height, scrolling=False)


def init_state() -> None:
    """Create session-state keys once so reruns never hit a missing attribute."""
    defaults = {"agent": None, "mermaid": "", "stats": {}, "repo_name": "", "messages": [], "n_files": 0, "n_chunks": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def run_analysis(api_key: str, url: str) -> None:
    """Clone -> graph -> embed -> build agent, with live progress and friendly errors."""
    with st.status("Analyzing repository…", expanded=True) as status:
        try:
            st.write("📥 Cloning repository (shallow) and extracting code files…")
            files, repo_name = ingest_repository(url)
            st.write(f"✅ Extracted {len(files)} files from **{repo_name}**")

            st.write("🕸️ Building architecture graph from import statements…")
            mermaid, stats = build_mermaid_graph(files)

            st.write("🧬 Embedding code chunks with Gemini…")
            progress = st.progress(0.0)
            store, n_chunks = build_vector_store(
                files, api_key, progress_cb=lambda done, total: progress.progress(done / total)
            )
            agent = RepoAgent(store, api_key, summarize_repository(files, repo_name))
        except (IngestionError, IndexingError, ValueError) as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # last-resort guard so the app never shows a raw traceback
            status.update(label="Analysis failed", state="error")
            st.error(f"Unexpected error: {exc}")
            return
        status.update(label=f"Ready: {repo_name}", state="complete", expanded=False)

    st.session_state.update(
        agent=agent, mermaid=mermaid, stats=stats, repo_name=repo_name,
        messages=[], n_files=len(files), n_chunks=n_chunks,
    )


def render_sources(sources: list[str]) -> None:
    """Show retrieved files under an assistant message."""
    if sources:
        with st.expander("📎 Sources"):
            for src in sources:
                st.markdown(f"- `{src}`")


def render_chat() -> None:
    """Chat tab: history, example prompts, and streamed RAG answers."""
    agent: RepoAgent | None = st.session_state.agent
    if agent is None:
        st.info("Analyze a repository first, then ask questions about it here.")
        return

    messages = st.session_state.messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

    if not messages:
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, question in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(question, use_container_width=True):
                st.session_state.pending_prompt = question

    prompt = st.chat_input("Ask anything about this codebase…") or st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    history = list(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = st.write_stream(agent.stream_answer(prompt, history))
        sources = list(agent.last_sources)
        render_sources(sources)
    messages.append({"role": "assistant", "content": answer, "sources": sources})


def main() -> None:
    """Page layout and event handling."""
    init_state()

    with st.sidebar:
        st.title("🧠 RepoMind AI")
        st.caption("Open-source codebase intelligence & architecture visualizer")
        api_key = st.text_input(
            "Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""),
            help="Get a free key at https://aistudio.google.com/apikey. Never stored on disk.",
        )
        repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
        analyze = st.button("🔍 Analyze Repository", type="primary", use_container_width=True)
        if st.session_state.repo_name:
            st.success(f"Loaded: {st.session_state.repo_name}")

    if analyze:
        if not api_key.strip():
            st.sidebar.error("Please enter your Gemini API key.")
        else:
            try:
                validate_github_url(repo_url)
            except IngestionError as exc:
                st.sidebar.error(str(exc))
            else:
                run_analysis(api_key, repo_url)

    st.title("RepoMind AI")
    if not st.session_state.repo_name:
        st.markdown("Paste a **public GitHub URL** in the sidebar and click **Analyze Repository**.")

    tab_graph, tab_chat = st.tabs(["🕸️ Architecture Map", "💬 Chat with the Repo"])

    with tab_graph:
        if not st.session_state.mermaid:
            st.info("The architecture map will appear here after analysis.")
        else:
            stats = st.session_state.stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Files indexed", st.session_state.n_files)
            c2.metric("Chunks embedded", st.session_state.n_chunks)
            c3.metric("Modules in graph", stats.get("nodes", 0))
            c4.metric("Dependencies", stats.get("edges", 0))
            if stats.get("edges", 0) == 0:
                st.warning("No internal import relationships were detected (supported: Python, JS/TS, Java/Kotlin).")
            elif stats.get("truncated"):
                st.caption(f"Showing the {stats['nodes']} most connected modules. 🟢 entry points · 🔴 most-imported hubs")
            else:
                st.caption("🟢 entry points · 🔴 most-imported hubs")
            render_mermaid(st.session_state.mermaid)
            with st.expander("Mermaid source"):
                st.code(st.session_state.mermaid, language="text")
            st.download_button("⬇️ Download .mmd", st.session_state.mermaid, file_name="architecture.mmd")

    with tab_chat:
        render_chat()


if __name__ == "__main__":
    main()
