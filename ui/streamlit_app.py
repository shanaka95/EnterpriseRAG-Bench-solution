"""Streamlit UI for the RAG agent.

Run with::

    cd /home/shanaka/Desktop/projects/rag
    ./scripts/run_ui.sh
"""
from __future__ import annotations
import html
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

BACKEND = "/data/projects/rag/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Defaults — read MINIMAX_API_KEY from the environment (or .env via
# scripts/run_ui.sh). The agent will refuse to start without it.
os.environ.setdefault("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
os.environ.setdefault("MINIMAX_MODEL", "MiniMax-M2.7")

from agent import run_agent  # noqa: E402
from agent.llm import get_llm  # noqa: E402

QUESTIONS_PATH = "/data/projects/rag/data/questions.jsonl"
CORPUS_DIR = "/data/projects/rag/data/all_documents"


# ---------- data loaders ----------

@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    out = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def read_doc(doc_id: str, max_chars: int = 4000) -> str:
    fp = os.path.join(CORPUS_DIR, doc_id)
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            txt = f.read(max_chars)
        if len(txt) == max_chars:
            txt += "…[truncated]"
        return txt
    except FileNotFoundError:
        return "(file not found on disk)"


def doc_short_label(doc_id: str) -> str:
    return doc_id.split("/")[-1] if "/" in doc_id else doc_id


def source_of(doc_id: str) -> str:
    return doc_id.split("/")[0] if "/" in doc_id else "?"


SRC_COLOR = {
    "github":        ("#ddf4ff", "#0969da", "🐙"),
    "slack":         ("#fde7f3", "#ad239a", "💬"),
    "gmail":         ("#fee2e2", "#b91c1c", "📧"),
    "jira":          ("#e0f2fe", "#0369a1", "🎫"),
    "linear":        ("#ede9fe", "#5b21b6", "📐"),
    "fireflies":     ("#fef3c7", "#b45309", "🦉"),
    "hubspot":       ("#ffe4e6", "#be123c", "🟠"),
    "confluence":    ("#dbeafe", "#1d4ed8", "📚"),
    "google_drive":  ("#dcfce7", "#15803d", "📁"),
}
DEFAULT_SRC = ("#f1f5f9", "#475569", "📄")


def src_style(src: str) -> tuple[str, str, str]:
    return SRC_COLOR.get(src, DEFAULT_SRC)


# ---------- session state ----------

def init_state():
    ss = st.session_state
    ss.setdefault("selected_qid", None)
    ss.setdefault("last_run", None)
    ss.setdefault("running", False)
    # LLM connection (UI overrides env vars when any field is set)
    ss.setdefault("llm_api_key", "")
    ss.setdefault("llm_base_url", "")
    ss.setdefault("llm_model", "")
    ss.setdefault("llm_protocol", "openai")
    ss.setdefault("tracing_enabled", True)
    # Per-UI-session Langfuse session id. Generated once per
    # Streamlit session (browser tab) so all "Run" clicks within the
    # same tab share one Langfuse session and can be replayed as a
    # single conversation. Format: ui-YYYYMMDD-HHMMSS-<short uuid>.
    if "langfuse_session_id" not in ss:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        ss.langfuse_session_id = f"ui-{stamp}-{uuid.uuid4().hex[:8]}"


def build_llm_from_ui() -> tuple[object, str, str, str]:
    """Build the LLM from sidebar inputs (UI values override env vars).

    Caches the result on ``st.session_state.llm_cache`` keyed by the
    current (key, url, model, protocol) tuple. Reruns that don't touch
    the sidebar inputs return the cached LLM without re-instantiating.

    Returns (llm, source_label, protocol, model) where ``source_label``
    is "from UI" or "from env" — used in the status line.
    """
    ss = st.session_state
    ui_key = ss.llm_api_key.strip()
    ui_url = ss.llm_base_url.strip()
    ui_model = ss.llm_model.strip()
    use_ui = bool(ui_key or ui_url or ui_model)
    source = "from UI" if use_ui else "from env"
    cache_key = (ui_key, ui_url, ui_model, ss.llm_protocol)
    cached = ss.get("_llm_cache")
    if cached and cached[0] == cache_key:
        llm = cached[1]
    else:
        llm = get_llm(
            protocol=ss.llm_protocol,
            api_key=ui_key or None,
            base_url=ui_url or None,
            model=ui_model or None,
        )
        ss._llm_cache = (cache_key, llm)
    return llm, source, ss.llm_protocol, (ui_model or os.environ.get("MINIMAX_MODEL", "?"))


# ---------- UI ----------

def main():
    st.set_page_config(
        page_title="RAG Agent Tester",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": "Reactive RAG agent — BM25 + jina-v3 + RRF + LLM (batches of 5)"},
    )
    init_state()

    # ---- custom CSS — clean light theme, high contrast ----
    st.markdown("""
    <style>
      /* Global typography */
      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
      }
      .stApp { background: #ffffff; }

      /* Make Streamlit headers & body text dark by default */
      h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
      p, li, label, span, div { color: #0f172a; }
      .stMarkdown, .stText, .stCaption { color: #0f172a !important; }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #f8fafc; border-right: 1px solid #e2e8f0;
      }
      section[data-testid="stSidebar"] * { color: #0f172a !important; }
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2 { color: #0f172a !important; }
      section[data-testid="stSidebar"] .stCaption,
      section[data-testid="stSidebar"] small { color: #475569 !important; }

      /* Question header card */
      .q-header {
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1; border-left: 4px solid #2563eb;
        border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
      }
      .q-header h2 { margin: 0 0 0.4rem 0; color: #0f172a !important;
                      font-size: 1.3rem; line-height: 1.4; }
      .q-meta { color: #475569; font-size: 0.85rem; }
      .q-meta code { background: #e2e8f0; padding: 1px 5px; border-radius: 3px;
                      font-size: 0.8rem; }

      /* Doc card */
      .doc-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
        padding: 0.55rem 0.75rem; margin: 0.3rem 0;
        font-size: 0.88rem;
      }
      .doc-card.gold { border: 2px solid #f59e0b; background: #fffbeb; }
      .doc-card.hit  { border: 2px solid #10b981; background: #ecfdf5; }
      .doc-card .doc-id { color: #0f172a; font-family: ui-monospace, SFMono-Regular,
                            Menlo, monospace; font-size: 0.82rem; }
      .doc-card .doc-label { color: #64748b; font-size: 0.78rem; }

      /* Source badge */
      .src-badge {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 600; margin-right: 6px;
        border: 1px solid transparent;
      }

      /* Trace blocks */
      .step {
        background: #f1f5f9; border-left: 3px solid #3b82f6;
        padding: 0.55rem 0.85rem; margin: 0.35rem 0; border-radius: 0 6px 6px 0;
        font-size: 0.9rem; color: #0f172a;
      }
      .step .step-label { color: #1d4ed8; font-weight: 700; font-size: 0.78rem;
                            text-transform: uppercase; letter-spacing: 0.04em; }
      .step small { color: #475569; }

      .toolmsg {
        background: #ecfdf5; border-left: 3px solid #10b981;
        padding: 0.55rem 0.85rem; margin: 0.35rem 0; border-radius: 0 6px 6px 0;
        font-size: 0.9rem; color: #064e3b;
      }
      .toolmsg .step-label { color: #047857; }

      .final-card {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        border: 2px solid #8b5cf6; border-radius: 8px;
        padding: 1rem 1.25rem; margin: 0.6rem 0; color: #1e1b4b;
      }
      .final-card.hit {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-color: #059669; color: #064e3b;
      }

      /* Metric cards */
      [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700; }
      [data-testid="stMetricLabel"] { color: #475569 !important; }
      div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.5rem 0.8rem;
      }

      /* Buttons */
      .stButton > button {
        font-weight: 600; border-radius: 6px;
      }

      /* Hide the Streamlit toolbar (Stop / Deploy / etc) — they aren't useful
         for a tester and just clutter the page */
      header[data-testid="stHeader"] { background: #ffffff !important; }
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }

      /* Expander headers */
      details summary { color: #0f172a !important; font-weight: 600; }
      details[open] summary { color: #1d4ed8 !important; }

      /* Code blocks inside expanders */
      pre, code { color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

    # ---- sidebar: LLM connection + question picker ----
    with st.sidebar:
        st.markdown("## 🔎 RAG Agent Tester")
        st.caption("500 enterprise questions")
        st.markdown("---")

        # ---- LLM connection (overrides env vars if any field is set) ----
        st.markdown("### 🤖 LLM connection")
        st.text_input(
            "API key", key="llm_api_key", type="password",
            placeholder=("(env: " + (os.environ.get("MINIMAX_API_KEY", "")[:6] + "…") if os.environ.get("MINIMAX_API_KEY") else "sk-…"),
            help="Leave empty to use MINIMAX_API_KEY env var.",
        )
        st.text_input(
            "Base URL", key="llm_base_url",
            placeholder=os.environ.get("MINIMAX_BASE_URL", "http://167.233.22.91:19950/"),
            help="Leave empty to use MINIMAX_BASE_URL env var. The OpenAI client strips a trailing /v1 automatically.",
        )
        st.text_input(
            "Model", key="llm_model",
            placeholder="openai 5.4",
            help="Leave empty to use MINIMAX_MODEL env var.",
        )
        st.radio(
            "Protocol", options=("openai", "anthropic"),
            key="llm_protocol", horizontal=True,
            help="OpenAI-compatible: most providers. Anthropic-compatible: claude-* models.",
        )
        # Build the LLM once per rerun to validate the inputs (raises on
        # bad protocol, etc.) and show the resolved status line.
        try:
            _, llm_source, llm_proto, llm_model_name = build_llm_from_ui()
            env_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
            if llm_source == "from UI":
                st.success(
                    f"✅ Using UI values: `{llm_proto}` / `{llm_model_name}` @ "
                    f"`{st.session_state.llm_base_url or env_url}`"
                )
            else:
                st.caption(
                    f"From env: `{llm_proto}` / `{llm_model_name}` @ `{env_url}`"
                )
        except Exception as e:
            st.error(f"LLM config error: {e}")
            st.stop()

        st.markdown("---")

        questions = load_questions()
        st.caption(f"Loaded **{len(questions)}** questions")

        qid = st.session_state.selected_qid
        if qid is None:
            qid = questions[0]["question_id"]
            st.session_state.selected_qid = qid

        q_search = st.text_input("🔍 Filter by text", "")
        if q_search:
            qs = [q for q in questions if q_search.lower() in q["question"].lower()]
        else:
            qs = questions

        labels = [f"{q['question_id']} · {q['question'][:55]}{'…' if len(q['question'])>55 else ''}"
                  for q in qs]
        try:
            cur_idx = next(i for i, q in enumerate(qs) if q["question_id"] == st.session_state.selected_qid)
        except StopIteration:
            cur_idx = 0
        sel = st.selectbox("Pick a question", range(len(qs)),
                           index=cur_idx, format_func=lambda i: labels[i],
                           label_visibility="collapsed")
        st.session_state.selected_qid = qs[sel]["question_id"]

        st.markdown("---")
        st.markdown("**Pipeline**")
        st.markdown("""
1. BM25@2k (sparse)
2. jina-v3@2k (dense)
3. RRF (k0=60) → top-100
4. LLM agent reads in batches of 5
        """)
        st.caption("First run ~4 min (BM25 build); subsequent ~30-90s")

        st.markdown("---")
        st.markdown("### 🔭 Tracing (Langfuse)")
        st.checkbox(
            "Send runs to Langfuse",
            key="tracing_enabled",
            help="When on, every run is traced to the Langfuse project "
                 "configured via LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / "
                 "LANGFUSE_BASE_URL. Disable for local-only runs.",
        )
        if st.session_state.tracing_enabled:
            lf_url = os.environ.get("LANGFUSE_BASE_URL", "")
            lf_pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            if lf_pk and lf_url:
                st.caption(f"✅ Tracing → `{lf_url}`")
                # Show the session id so the user can find this UI
                # session in the Langfuse Sessions view.
                st.caption(f"🆔 Session: `{st.session_state.langfuse_session_id}`")
            else:
                st.caption("⚠️ Langfuse env vars not set — tracing will no-op")

    # ---- main: question card ----
    questions = load_questions()
    q = next(q for q in questions if q["question_id"] == st.session_state.selected_qid)

    src_badges = " ".join(
        f'<span class="src-badge" style="background:{src_style(s)[0]};'
        f'color:{src_style(s)[1]};border-color:{src_style(s)[1]}33;">'
        f'{src_style(s)[2]} {s}</span>'
        for s in q["source_types"]
    )
    st.markdown(f"""
    <div class="q-header">
      <h2>❓ {q['question']}</h2>
      <div class="q-meta">
        <code>id: {q['question_id']}</code> &nbsp;·&nbsp;
        <code>type: {q['question_type']}</code> &nbsp;·&nbsp;
        {src_badges}
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Expected answer & gold docs", expanded=True):
        st.markdown("**Gold answer:**")
        st.info(q.get("gold_answer") or "(none)")
        st.markdown(f"**Gold doc_id(s) — {len(q.get('expected_doc_ids', []))}:**")
        for did in q.get("expected_doc_ids", []):
            bg, fg, icon = src_style(source_of(did))
            st.markdown(
                f'<div class="doc-card gold">'
                f'<span class="src-badge" style="background:{bg};color:{fg};'
                f'border-color:{fg}33;">{icon} {source_of(did)}</span>'
                f'<span class="doc-id">{did}</span><br>'
                f'<span class="doc-label">{doc_short_label(did)}</span>'
                f'</div>', unsafe_allow_html=True)
            with st.expander(f"📄 Peek: {doc_short_label(did)}"):
                st.text(read_doc(did, max_chars=2500))

    # ---- run button ----
    col_run, col_clear, col_spacer = st.columns([1, 1, 4])
    with col_run:
        run = st.button("▶ Run agent", type="primary",
                        use_container_width=True,
                        disabled=st.session_state.running)
    with col_clear:
        if st.session_state.last_run and st.button("🗑 Clear", use_container_width=True):
            st.session_state.last_run = None
            st.rerun()

    if run:
        st.session_state.running = True
        with st.spinner("⏳ Running BM25 + jina-v3 + RRF + agent…"):
            t0 = time.time()
            try:
                llm, _, _, _ = build_llm_from_ui()
                final = run_agent(
                    q["question"],
                    question_id=q["question_id"],
                    llm=llm,
                    trace=st.session_state.tracing_enabled,
                    session_id=st.session_state.langfuse_session_id,
                )
                # Flush Langfuse events so the trace appears in the
                # dashboard before the spinner goes away
                if st.session_state.tracing_enabled:
                    try:
                        from agent.tracing import flush
                        flush()
                    except Exception:
                        pass
                final["_wallclock_s"] = time.time() - t0
                final["_llm_source"] = st.session_state.llm_api_key or st.session_state.llm_base_url or st.session_state.llm_model
                final["_traced"] = st.session_state.tracing_enabled
                st.session_state.last_run = final
            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.session_state.last_run = None
            finally:
                st.session_state.running = False
        st.rerun()

    # ---- results ----
    final = st.session_state.last_run
    if final is None:
        st.markdown("---")
        st.info("👆 Click **Run agent** to start. The pipeline runs BM25 + jina-v3 + RRF (top-100) and then the LLM agent reads the docs in batches of 5.")
        return

    render_results(final, q)


# ---------- result renderer ----------

def render_results(final: dict, q: dict):
    expected = q.get("expected_doc_ids", []) or []
    supporting = final.get("supporting_doc_ids") or []
    hit = any(any(e in d for d in supporting) for e in expected)

    st.markdown("---")
    st.markdown("## 📊 Run summary")
    bc = st.columns(5)
    bc[0].metric("⏱ Wall clock", f"{final.get('_wallclock_s', 0):.1f}s")
    refined = final.get("refined_doc_ids") or []
    bc[1].metric("📚 Refined", len(refined))
    bc[2].metric("📖 Docs read", final.get("current_idx", 0))
    n_ai = sum(1 for m in final.get("messages", []) if m.__class__.__name__ == "AIMessage")
    bc[3].metric("🤖 LLM turns", n_ai)
    bc[4].metric("🎯 Hit", "✅ YES" if hit else "❌ NO")
    llm_used = "UI values" if final.get("_llm_source") else "env vars"
    st.caption(f"🤖 LLM: {llm_used} · {st.session_state.get('llm_protocol', 'openai')} · "
               f"model = `{st.session_state.get('llm_model') or os.environ.get('MINIMAX_MODEL', '?')}`")

    # ---- retrieval stages ----
    with st.expander("⚙️ Retrieval stages (BM25, jina-v3, RRF)", expanded=False):
        for t in final.get("node_trace", []):
            if t.get("node") in ("bm25_retrieve", "jina_dense_retrieve", "rrf_fuse"):
                n_ret = t.get("n_returned", t.get("n_unique", "?"))
                st.markdown(
                    f'<div class="step"><span class="step-label">{t["node"]}</span> '
                    f'&nbsp;·&nbsp; {t.get("elapsed_s", 0):.3f}s '
                    f'&nbsp;·&nbsp; n_returned = <b>{n_ret}</b></div>',
                    unsafe_allow_html=True)

    # ---- refined 100 docs ----
    with st.expander(f"📚 Refined top-{len(refined)} (RRF, click to peek)", expanded=False):
        st.caption("Click any doc to peek at its text. ★ = gold doc.")
        for i, did in enumerate(refined, 1):
            gold = any(e in did for e in expected)
            mark = "★" if gold else "·"
            bg, fg, icon = src_style(source_of(did))
            label = f"{mark} {i:>3}. {icon} {source_of(did)} · {doc_short_label(did)}"
            with st.expander(label):
                st.markdown(
                    f'<div class="doc-card{" gold" if gold else ""}">'
                    f'<span class="src-badge" style="background:{bg};color:{fg};'
                    f'border-color:{fg}33;">{icon} {source_of(did)}</span>'
                    f'<span class="doc-id">{did}</span><br>'
                    f'<span class="doc-label">RRF rank #{i}</span>'
                    f'</div>', unsafe_allow_html=True)
                st.text(read_doc(did, max_chars=2000))

    # ---- agent trace ----
    st.markdown("## 🤖 Agent trace (LLM ↔ tool)")
    msgs = final.get("messages", [])
    trace_container = st.container()
    started = False
    for m in msgs:
        cname = m.__class__.__name__
        if not started and cname == "SystemMessage":
            continue
        if not started and cname == "HumanMessage":
            started = True
            trace_container.markdown(
                f'<div class="toolmsg"><span class="step-label">USER</span><br>'
                f'{str(m.content)[:400]}</div>', unsafe_allow_html=True)
            continue
        if cname == "AIMessage":
            tcs = m.tool_calls or []
            if tcs:
                tool_names = ", ".join(tc.get("name", "?") for tc in tcs)
                # Show the full args the agent passed to the tool — no
                # truncation, no matter how long the response is. For
                # `submit_answer` the `response` field is the final answer
                # the user wants to see in full; for `get_next_batch` the
                # args are tiny, so there's no cost to showing them whole.
                args_preview = "<br>".join(
                    f'<code>{tc.get("name")}('
                    f'{html.escape(json.dumps(tc.get("args", {}), ensure_ascii=False))}'
                    f')</code>'
                    for tc in tcs)
                trace_container.markdown(
                    f'<div class="step">'
                    f'<span class="step-label">AGENT → tool</span>: {tool_names}'
                    f'<br>{args_preview}</div>',
                    unsafe_allow_html=True)
            else:
                text = (str(m.content) or "")[:600]
                trace_container.markdown(
                    f'<div class="step"><span class="step-label">AGENT (final)</span><br>'
                    f'{text}</div>', unsafe_allow_html=True)
        elif cname == "ToolMessage":
            content = m.content
            if isinstance(content, str):
                try:
                    obj = json.loads(content)
                except Exception:
                    obj = {"raw": content[:300]}
            else:
                obj = content
            if isinstance(obj, dict) and "batch" in obj:
                batch = obj["batch"]
                rng = obj.get("index_range", [0, len(batch)])
                trace_container.markdown(
                    f'<div class="toolmsg"><span class="step-label">TOOL → docs '
                    f'{rng[0]}-{rng[1]} of {obj.get("total_refined", "?")}</span> '
                    f'({len(batch)} docs)</div>', unsafe_allow_html=True)
                for j, d in enumerate(batch, 1):
                    did = d.get("doc_id", "?")
                    gold = any(e in did for e in expected)
                    preview = (d.get("content") or "")[:200].replace("\n", " ")
                    bg, fg, icon = src_style(source_of(did))
                    trace_container.markdown(
                        f'<div class="doc-card{" gold" if gold else ""}">'
                        f'<b>{rng[0] + j - 1}.</b> '
                        f'<span class="src-badge" style="background:{bg};color:{fg};'
                        f'border-color:{fg}33;">{icon} {source_of(did)}</span>'
                        f'<span class="doc-id">{doc_short_label(did)}</span> '
                        f'{"<b style=\"color:#b45309;\">★ EXPECTED</b>" if gold else ""}'
                        f'<br><span class="doc-label">{preview}…</span></div>',
                        unsafe_allow_html=True)
            elif isinstance(obj, dict) and obj.get("exhausted"):
                trace_container.markdown(
                    f'<div class="toolmsg"><span class="step-label">TOOL → exhausted</span>'
                    f' &nbsp; No more docs.</div>', unsafe_allow_html=True)
            else:
                trace_container.markdown(
                    f'<div class="toolmsg"><span class="step-label">TOOL</span> '
                    f'{str(obj)[:300]}</div>', unsafe_allow_html=True)

    # ---- final answer ----
    st.markdown("---")
    st.markdown("## 🎯 Final answer")
    # final_answer is the `response` field of the submit_answer tool call.
    # The tool guarantees it's a clean string (validated JSON via Pydantic),
    # never polluted with thinking traces. An empty answer from the model
    # is replaced with the explicit "Question cannot be answered..." text.
    raw_ans = final.get("final_answer") or ""
    if not raw_ans.strip():
        ans = "Question cannot be answered with the available documents."
        # Style this case distinctly so the user sees at a glance
        # that the model could not find the answer.
        st.markdown(
            f'<div class="final-card" style="border-color:#94a3b8;'
            f'background:linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 100%);">'
            f'<b style="font-size:1.05rem;color:#475569;">❓ No answer found</b>'
            f'<hr style="border-color:#94a3b8;opacity:0.3;">'
            f'<span style="color:#0f172a;">{ans}</span></div>',
            unsafe_allow_html=True)
    else:
        ans = raw_ans
        if final.get("finished_via_tool"):
            st.caption("✅ Extracted from `submit_answer` tool call (structured JSON)")
        if hit:
            st.markdown(
                f'<div class="final-card hit">'
                f'<b style="font-size:1.1rem;">✅ HIT — gold doc was retrieved and cited</b>'
                f'<hr style="border-color:#10b981;opacity:0.3;">{ans}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="final-card">{ans}</div>', unsafe_allow_html=True)

    if supporting:
        st.markdown("### 📎 Supporting docs cited by the agent")
        for did in supporting:
            gold = any(e in did for e in expected)
            bg, fg, icon = src_style(source_of(did))
            st.markdown(
                f'<div class="doc-card{" hit" if gold else ""}">'
                f'<span class="src-badge" style="background:{bg};color:{fg};'
                f'border-color:{fg}33;">{icon} {source_of(did)}</span>'
                f'<span class="doc-id">{did}</span>'
                f' {"<b style=\"color:#b45309;\">★ EXPECTED</b>" if gold else ""}'
                f'</div>', unsafe_allow_html=True)
            with st.expander(f"📄 Peek: {doc_short_label(did)}"):
                st.text(read_doc(did, max_chars=3000))


if __name__ == "__main__":
    main()
