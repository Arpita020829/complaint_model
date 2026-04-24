import streamlit as st
import numpy as np
import pickle
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Hostel Complaint Classifier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_LEN = 60

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0d0f1a;
    --surface: #13162a;
    --surface2: #1c2038;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --accent3: #43e97b;
    --text: #e8eaf6;
    --text-muted: #7986cb;
    --border: rgba(108,99,255,0.25);
    --glow: 0 0 30px rgba(108,99,255,0.35);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #12102b 50%, #0d1a1a 100%) !important;
    background-attachment: fixed !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1240 0%, #0f2027 60%, #1a0530 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(108,99,255,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,101,132,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff 0%, #a5b4fc 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(108,99,255,0.55);
    box-shadow: 0 0 20px rgba(108,99,255,0.2);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,99,255,0.6) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── TextArea ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.25) !important;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, rgba(67,233,123,0.08) 0%, rgba(56,249,215,0.04) 100%);
    border: 1px solid rgba(67,233,123,0.3);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.result-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #43e97b;
    margin-bottom: 0.5rem;
}
.result-category {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}
.result-conf {
    font-size: 1.4rem;
    font-weight: 600;
    color: #43e97b;
}

/* ── Progress Bars (top-k) ── */
.topk-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
    gap: 0.8rem;
}
.topk-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text);
    min-width: 140px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.topk-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--surface2);
    border-radius: 99px;
    overflow: hidden;
}
.topk-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #6c63ff, #a855f7);
    transition: width 0.6s ease;
}
.topk-pct {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-muted);
    min-width: 42px;
    text-align: right;
}

/* ── History Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a5b4fc !important;
    margin-bottom: 0.5rem;
}

/* ── Category chip ── */
.chip {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.35);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 3px 3px 3px 0;
}

/* ── Metric tiles ── */
.metric-tile {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #a5b4fc;
}
.metric-desc {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 2px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL & ARTIFACTS  (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model     = load_model("model/model.h5")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def predict(text, model, tokenizer, encoder, top_k=5):
    seq  = tokenizer.texts_to_sequences([text.lower().strip()])
    pad  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(pad, verbose=0)[0]
    top_indices = np.argsort(prob)[::-1][:top_k]
    return [
        {"category": encoder.classes_[i], "confidence": float(prob[i]) * 100}
        for i in top_indices
    ]


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")

    top_k = st.slider("Top-K Predictions", min_value=1, max_value=8, value=4,
                      help="How many category predictions to display")
    st.markdown("---")

    st.markdown('<div class="sidebar-title">📊 Session Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-num">{st.session_state.total_predictions}</div>
            <div class="metric-desc">Predictions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        hist_cats = len(set(h["Category"] for h in st.session_state.history)) if st.session_state.history else 0
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-num">{hist_cats}</div>
            <div class="metric-desc">Unique Cats</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 Sample Complaints</div>', unsafe_allow_html=True)
    samples = [
        "The WiFi is very slow and disconnects constantly",
        "My room has cockroaches and pests everywhere",
        "The water heater in my bathroom is not working",
        "Mess food quality is very bad and unhygienic",
        "The corridor lights have been broken for 3 days",
        "There is no hot water available in the morning",
    ]
    selected_sample = st.selectbox("Pick a sample →", ["(none)"] + samples, label_visibility="collapsed")

    st.markdown("---")
    if st.session_state.history and st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.total_predictions = 0
        st.rerun()


# ─────────────────────────────────────────
# LOAD MODEL (with status)
# ─────────────────────────────────────────
with st.spinner("Loading model…"):
    try:
        model, tokenizer, encoder = load_artifacts()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error = str(e)

# ─────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 Powered by Bi-LSTM NLP</div>
    <div class="hero-title">Hostel Complaint<br>Classifier</div>
    <p class="hero-sub">Instantly categorize student hostel complaints using deep learning — faster, smarter, fairer.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Failed to load model: `{load_error}`\n\nMake sure `model/` folder contains `model.h5`, `tokenizer.pkl`, `encoder.pkl`.")
    st.stop()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Classify Complaint", "📜  History"])

# ══════════════════════════════════
# TAB 1 — CLASSIFIER
# ══════════════════════════════════
with tab1:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">✍️ Complaint Input</div>', unsafe_allow_html=True)

        # Pre-fill from sidebar sample
        default_text = selected_sample if selected_sample != "(none)" else ""
        complaint_text = st.text_area(
            label="complaint_input",
            value=default_text,
            placeholder="e.g.  The washroom drain is completely blocked and there's water everywhere…",
            height=150,
            label_visibility="collapsed",
        )

        char_count = len(complaint_text)
        word_count = len(complaint_text.split()) if complaint_text.strip() else 0
        st.caption(f"Characters: **{char_count}** · Words: **{word_count}**")

        predict_btn = st.button("🚀  Classify Complaint", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── About card ──
        st.markdown("""
        <div class="card">
            <div class="card-title">ℹ️ About the Model</div>
            <p style="font-size:0.88rem;color:#9ca3af;line-height:1.7;margin:0">
            Bidirectional LSTM trained on a balanced hostel complaints dataset.
            Outputs calibrated probability scores across all complaint categories,
            letting warden staff triage and route issues instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        if predict_btn:
            if not complaint_text.strip():
                st.warning("⚠️  Please enter a complaint before classifying.")
            else:
                with st.spinner("Analysing complaint…"):
                    time.sleep(0.4)   # brief dramatic pause
                    results = predict(complaint_text, model, tokenizer, encoder, top_k=top_k)

                top   = results[0]
                conf  = top["confidence"]

                # ── Result box ──
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">✅ Top Prediction</div>
                    <div class="result-category">{top["category"]}</div>
                    <div class="result-conf">{conf:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Top-K bars ──
                st.markdown('<div class="card"><div class="card-title">📊 All Predictions</div>', unsafe_allow_html=True)
                for r in results:
                    pct = r["confidence"]
                    bar_w = max(pct, 1)
                    color = "#6c63ff" if r == results[0] else "#3b4fcf"
                    st.markdown(f"""
                    <div class="topk-row">
                        <div class="topk-label">{r["category"]}</div>
                        <div class="topk-bar-bg">
                            <div class="topk-bar-fill" style="width:{bar_w}%;background:{'linear-gradient(90deg,#43e97b,#38f9d7)' if r==results[0] else 'linear-gradient(90deg,#6c63ff,#a855f7)'};"></div>
                        </div>
                        <div class="topk-pct">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Save to history ──
                st.session_state.history.append({
                    "Complaint": complaint_text[:80] + ("…" if len(complaint_text) > 80 else ""),
                    "Category": top["category"],
                    "Confidence": f"{conf:.1f}%",
                })
                st.session_state.total_predictions += 1

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 2rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🏠</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#a5b4fc;margin-bottom:0.5rem;">
                    Ready to Classify
                </div>
                <div style="font-size:0.88rem;color:#6b7280;line-height:1.7;">
                    Type or paste a hostel complaint on the left,<br>then click <b>Classify Complaint</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show categories
            if model_loaded:
                st.markdown('<div class="card"><div class="card-title">🏷️ Supported Categories</div>', unsafe_allow_html=True)
                cats_html = "".join(f'<span class="chip">{c}</span>' for c in encoder.classes_)
                st.markdown(cats_html + "</div>", unsafe_allow_html=True)


# ══════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════
with tab2:
    if not st.session_state.history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">📭</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#a5b4fc;">
                No predictions yet
            </div>
            <div style="font-size:0.88rem;color:#6b7280;margin-top:0.4rem;">
                Head to the Classify tab and run your first prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state.history[::-1])
        st.markdown(f"**{len(df_hist)} predictions** this session")
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Complaint":   st.column_config.TextColumn("Complaint", width="large"),
                "Category":    st.column_config.TextColumn("Category",  width="medium"),
                "Confidence":  st.column_config.TextColumn("Confidence", width="small"),
            }
        )

        # Download button
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download as CSV",
            data=csv,
            file_name="complaint_predictions.csv",
            mime="text/csv",
        )