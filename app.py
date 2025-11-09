# app.py
import re
import streamlit as st
from PIL import Image

from main import (
    load_policy_data,
    build_clients,
    extract_text_from_image,
    extract_text_from_pdf,
    chat_with_history,
)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Medical Claim Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Pro UI CSS ----------------
st.markdown(
    """
    <style>
      :root{
        --bg:#f6f9fc;
        --text:#0f172a;
        --muted:#64748b;
        --primary:#2563eb;
        --border:#e6eaf0;
        --card:#ffffff;
        --ai:#f8fafc;
        --user:#e8f3ff;
        --shadow: 0 8px 24px rgba(15, 23, 42, .06);
      }
      html, body, [data-testid="stApp"] { background: var(--bg); color: var(--text); }

      /* page title */
      .titlebar { display:flex; align-items:center; gap:.6rem; }
      .tag { font-size:.8rem; color:var(--muted); border:1px solid var(--border); padding:.15rem .6rem; border-radius:999px; background:#fff; }

      /* cards */
      .panel { background: var(--card); border:1px solid var(--border); border-radius:18px; padding:18px; box-shadow: var(--shadow); }
      .metric { display:flex; align-items:center; justify-content:space-between; border:1px solid var(--border);
                border-radius:12px; padding:.55rem .8rem; margin:.35rem 0; background:#fff; }
      .metric b { color: var(--primary); }

      /* chat area */
      .chat-wrap { max-height: 60vh; overflow:auto; padding-right:8px; margin-top:.25rem; }
      .msg { display:flex; gap:.65rem; margin:14px 0; }
      .avatar { width:34px; height:34px; border-radius:50%; background:#eef2ff; display:flex; align-items:center; justify-content:center; font-size:18px; }
      .bubble { flex:1; border:1px solid var(--border); padding:12px 14px; border-radius:14px; background:var(--ai); box-shadow: var(--shadow); }
      .bubble.user { background: var(--user); border-top-right-radius:6px; }
      .bubble.ai   { background: var(--ai); border-top-left-radius:6px; }
      .who { font-size:.82rem; color:var(--muted); margin-bottom:.25rem; }

      .chat-empty { color: var(--muted); border:1px dashed var(--border); border-radius:12px; padding:16px; text-align:center; background:#fff; }

      /* composer */
      .composer { position: sticky; bottom: 0; background: linear-gradient(180deg, rgba(246,249,252,0), var(--bg) 45%); padding-top:8px; }
      textarea[aria-label="chat-input"] { background: var(--card)!important; color: var(--text)!important; border:1px solid var(--border)!important; border-radius:12px!important; }

      [data-testid="stFileUploader"] > div { padding: 0.25rem 0; }
      .btn-primary button { background:#ff5555 !important; border-color:#ff5555 !important; }
      .btn-ghost button { background:#fff !important; color:var(--text)!important; border:1px solid var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Data / Clients ----------------
policies = load_policy_data()

if "llm_chat" not in st.session_state:
    st.session_state.llm_chat, st.session_state.llm_vision = build_clients()

st.session_state.setdefault("bill_text", "")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("__clear_composer", False)  # flag to clear input safely next rerun

# ---------------- Helpers: format assistant nicely ----------------
def fmt_money_blocks(txt: str) -> str:
    """
    Make the assistant output look professional:
    - Ensure each '1. ... 2. ... 3. ...' point is on its own line
    - After '3. Breakdown:' convert ' - ' into proper bullets
    - Compress extra spaces/newlines
    - Render as Markdown (numbers + bullets look perfect)
    """
    s = txt.strip()

    # 1) Put each numbered item on its own line
    # Insert newline before a number+dot if it's not at the start already.
    s = re.sub(r'\s*(?<!^)(?=(\d+\.\s))', '\n', s)

    # 2) Normalize multiple spaces after periods
    s = re.sub(r'\.\s+', '. ', s)

    # 3) Make sure "3. Breakdown:" is its own line
    s = s.replace('3. Breakdown:', '3. Breakdown:\n')

    # 4) For bullet items (often written inline like " - X - Y"), break them into lines.
    #    Only add bullets for occurrences after "Breakdown:" to avoid touching normal hyphens.
    if 'Breakdown:' in s:
        head, tail = s.split('Breakdown:', 1)
        # ensure bullets: replace ' - ' with newline + bullet
        tail = tail.replace(' - ', '\n   - ')
        s = head + 'Breakdown:' + tail

    # 5) Remove duplicate blank lines
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s

# ---------------- Sidebar: Policy ----------------
st.sidebar.markdown('<div class="titlebar"><h2>🩺 Policy</h2><span class="tag">context</span></div>', unsafe_allow_html=True)

insurer = st.sidebar.selectbox("Insurer", list(policies.keys()))
plan_key = st.sidebar.selectbox("Plan", list(policies[insurer].keys()))
policy = policies[insurer][plan_key]

with st.sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"**{policy['insurer']} – {policy['plan']}**")
    st.markdown(
        f"""
        <div class="metric"><span>Sum Insured</span><b>₹{policy['sum_insured']:,}</b></div>
        <div class="metric"><span>Room Type</span><b>{policy.get('room',{}).get('type','-')}</b></div>
        <div class="metric"><span>Room Cap / day</span><b>{'No cap' if policy.get('room',{}).get('cap_per_day') in (None,'null') else '₹'+str(policy['room']['cap_per_day'])}</b></div>
        <div class="metric"><span>Proportionate Deduction</span><b>{str(policy.get('room',{}).get('proportionate_deduction', False))}</b></div>
        <div class="metric"><span>Co-pay</span><b>{policy.get('copay',{}).get('percentage',0)}%</b></div>
        """,
        unsafe_allow_html=True,
    )
    np_txt = ", ".join(policy.get("non_payables", [])) or "—"
    st.caption(f"**Non-payables keywords:** {np_txt}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Header ----------------
left, right = st.columns([0.6, 0.4], gap="large")
with left:
    st.markdown('<div class="titlebar"><h1>💬 Chat</h1><span class="tag">medical claim assistant</span></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="titlebar"><h1>📄 Bill</h1><span class="tag">upload & OCR</span></div>', unsafe_allow_html=True)

# ---------------- Layout: Chat | Bill ----------------
chat_col, bill_col = st.columns([0.6, 0.4], gap="large")

# --- Bill Panel ---
with bill_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    up = st.file_uploader("Upload bill (PDF / JPEG / PNG)", type=["pdf", "jpg", "jpeg", "png"])

    col_b1, col_b2 = st.columns(2)
    extract_clicked = col_b1.button("🔍 Extract Text", use_container_width=True)
    clear_clicked = col_b2.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.bill_text = ""
        st.rerun()

    if extract_clicked:
        if not up:
            st.warning("Please upload a bill file first.")
        else:
            try:
                if up.type.startswith("image/"):
                    img = Image.open(up).convert("RGB")
                    st.image(img, caption="Uploaded bill", use_column_width=True)
                    st.session_state.bill_text = extract_text_from_image(img, st.session_state.llm_vision)
                else:
                    st.session_state.bill_text = extract_text_from_pdf(up.read(), st.session_state.llm_vision)
                st.success("Text extracted from bill.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    # Extracted text HIDDEN by default
    with st.expander("Show extracted text (optional)", expanded=False):
        st.text_area("bill-text", st.session_state.bill_text, height=260, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Panel ---
with chat_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # history view
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown('<div class="chat-empty">No messages yet. Upload a bill (optional) and ask a question.</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(
                    f'''
                    <div class="msg">
                      <div class="avatar">🙋</div>
                      <div class="bubble user">
                        <div class="who">You</div>
                        {content}
                      </div>
                    </div>
                    ''', unsafe_allow_html=True
                )
            elif role == "assistant":
                clean = fmt_money_blocks(content)  # <- beautify
                st.markdown(
                    f'''
                    <div class="msg">
                      <div class="avatar">🤖</div>
                      <div class="bubble ai">
                        <div class="who">Assistant</div>
                        </div>
                    ''', unsafe_allow_html=True
                )
                # Render nicely as markdown inside the bubble
                st.markdown(clean)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- safely clear composer BEFORE rendering the widget if flagged ---
    if st.session_state.get("__clear_composer", False):
        st.session_state["composer"] = ""
        st.session_state["__clear_composer"] = False

    # composer
    st.markdown('<div class="composer">', unsafe_allow_html=True)
    st.text_area(
        "chat-input",
        key="composer",                         # widget state key only
        label_visibility="collapsed",
        placeholder="Ask about co-pay, coverage, non-payables, room cap…",
        height=100,
    )
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        send = st.button("Send", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        reset = st.button("Reset Chat", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if reset:
        st.session_state.chat_history = []
        st.session_state["__clear_composer"] = True  # clear input on next rerun
        st.rerun()

    if send:
        text = (st.session_state.get("composer") or "").strip()
        if text:
            try:
                _ = chat_with_history(
                    st.session_state.llm_chat,
                    st.session_state.chat_history,
                    policy,
                    st.session_state.bill_text,
                    text,
                )
                st.session_state["__clear_composer"] = True  # clear input next rerun
                st.rerun()
            except Exception as e:
                st.error(f"Chat failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
