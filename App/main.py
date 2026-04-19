import sys
import os

# Add the PROJECT ROOT (parent of App) to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Now this import works
from src import (
    EMOTION_PALETTE,
    MODEL_META,
    load_all_models,
    predict_emotion,
    extract_texts_from_file,
    render_probability_bars,
    render_result_card,
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="EmotionDetector AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME ----------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ---------- CSS ----------
def get_css(theme):
    if theme == "dark":
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        :root {
            --bg:#0b0f19; --s1:#131a28; --s2:#1a2332; --border:#253040;
            --accent:#5b6ef5; --text:#eef2fb; --muted:#7a8aa3;
        }
        .stApp {background:var(--bg); color:var(--text); font-family:'Inter',sans-serif;}
        section[data-testid="stSidebar"] {background:var(--s1); border-right:1px solid var(--border);}
        .block-container {padding:2rem 2rem 4rem;}
        .hero {background:linear-gradient(145deg, #111b29, #0b101a); border-radius:20px; padding:2rem 2.5rem; margin-bottom:2rem; border:1px solid #1e2a3a;}
        .hero h1 {font-size:2.5rem; font-weight:600; margin:0 0 0.3rem; color:white;}
        .hero-sub {color:var(--muted); font-size:1rem;}
        .stat-badge {background:#1a2332; border-radius:12px; padding:0.6rem 1.2rem; border:1px solid #253040;}
        .stButton button {background:var(--accent)!important; border:none!important; border-radius:10px!important; font-weight:500!important; padding:0.6rem 1.5rem!important; width:100%;}
        .stTextArea textarea {background:#131a28!important; border:1px solid #253040!important; border-radius:14px!important; color:var(--text)!important;}
        .result-box {border-radius:20px; padding:2rem; text-align:center; background:#131a28; border:1px solid #253040;}
        .prob-row {display:flex; align-items:center; gap:10px; margin-bottom:10px;}
        .prob-emo {width:90px; text-align:right;}
        .prob-bg {flex:1; background:#1e2a3a; height:8px; border-radius:6px;}
        .prob-fill {height:100%; border-radius:6px;}
        .prob-val {width:50px; text-align:right; color:var(--muted); font-size:0.8rem;}
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        :root {
            --bg:#f5f7fc; --s1:#ffffff; --s2:#f0f3f9; --border:#d9e1ec;
            --accent:#5b6ef5; --text:#1a2639; --muted:#64748b;
        }
        .stApp {background:var(--bg); color:var(--text); font-family:'Inter',sans-serif;}
        section[data-testid="stSidebar"] {background:var(--s1); border-right:1px solid var(--border);}
        .block-container {padding:2rem 2rem 4rem;}
        .hero {background:white; border-radius:20px; padding:2rem 2.5rem; margin-bottom:2rem; border:1px solid #d9e1ec; box-shadow:0 6px 14px rgba(0,0,0,0.02);}
        .hero h1 {font-size:2.5rem; font-weight:600; margin:0 0 0.3rem; color:#0f172a;}
        .hero-sub {color:var(--muted); font-size:1rem;}
        .stat-badge {background:#f0f3f9; border-radius:12px; padding:0.6rem 1.2rem; border:1px solid #d9e1ec;}
        .stButton button {background:var(--accent)!important; border:none!important; border-radius:10px!important; font-weight:500!important; padding:0.6rem 1.5rem!important; width:100%;}
        .stTextArea textarea {background:white!important; border:1px solid #d9e1ec!important; border-radius:14px!important; color:var(--text)!important;}
        .result-box {border-radius:20px; padding:2rem; text-align:center; background:white; border:1px solid #d9e1ec;}
        .prob-row {display:flex; align-items:center; gap:10px; margin-bottom:10px;}
        .prob-emo {width:90px; text-align:right;}
        .prob-bg {flex:1; background:#e2e8f0; height:8px; border-radius:6px;}
        .prob-fill {height:100%; border-radius:6px;}
        .prob-val {width:50px; text-align:right; color:var(--muted); font-size:0.8rem;}
        </style>
        """

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_resources():
    tfidf, encoder, models = load_all_models()
    classes = encoder.classes_.tolist()
    return tfidf, encoder, models, classes

try:
    TFIDF, ENCODER, MODELS, CLASSES = load_resources()
    models_ok = True
except Exception as e:
    models_ok = False
    st.error(f"Failed to load models: {e}")
    st.stop()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🧠 EmotionDetector")
    st.caption("ML · NLP · v1.0")

    # Theme toggle
    st.write("**Appearance**")
    theme_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        toggle_theme()
        st.rerun()

    st.divider()
    st.write("**Choose Model**")
    model_choice = st.radio(
        "model",
        options=list(MODELS.keys()),
        label_visibility="collapsed"
    )

    # Model info — BUG 4 FIX: use concrete colors per theme instead of CSS vars
    # (CSS vars defined in .stApp scope are unreliable inside st.markdown html strings)
    if model_choice in MODEL_META:
        t, v, s, d = MODEL_META[model_choice]
        is_dark = st.session_state.theme == "dark"
        card_bg   = "#1a2332" if is_dark else "#f0f3f9"
        muted_col = "#7a8aa3" if is_dark else "#64748b"
        border_col= "#253040" if is_dark else "#d9e1ec"
        text_col  = "#eef2fb" if is_dark else "#1a2639"
        st.markdown(f"""
        <div style="background:{card_bg}; border-radius:12px; padding:1rem; margin-top:1rem; border:1px solid {border_col};">
            <p style="margin:0 0 8px; color:{text_col};"><span style="color:{muted_col};">Type &nbsp;: </span>{t}</p>
            <p style="margin:0 0 8px; color:{text_col};"><span style="color:{muted_col};">Input : </span>{v}</p>
            <p style="margin:0; color:{text_col};"><span style="color:{muted_col};">Speed : </span><span style="color:#f5c542;">{s}</span></p>
            <hr style="margin:12px 0; border-color:{border_col};">
            <p style="margin:0; font-size:0.85rem; color:{muted_col};">{d}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.write("**Emotion Classes**")
    for emo, cfg in EMOTION_PALETTE.items():
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
            f'<span>{cfg["emoji"]}</span>'
            f'<span style="text-transform:capitalize;">{emo}</span>'
            f'<span style="margin-left:auto; width:10px; height:10px; border-radius:50%; background:{cfg["color"]};"></span>'
            f'</div>',
            unsafe_allow_html=True
        )

# ---------- MAIN ----------
st.markdown("""
<div class="hero">
    <h1>EmotionDetector AI</h1>
    <p class="hero-sub">Analyze emotional tone with trained ML models — single text or batch files.</p>
    <div style="display:flex; gap:20px; margin-top:24px;">
        <div class="stat-badge">🧪 {} models loaded</div>
        <div class="stat-badge">🎭 {} emotions</div>
        <div class="stat-badge">📄 CSV · PDF · TXT · DOCX</div>
    </div>
</div>
""".format(len(MODELS), len(CLASSES)), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Single Input", "📂 Batch Processing", "💡 Examples"])

# ---------- TAB 1: Single Input ----------
with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Enter text")
        # BUG 1 & 3 FIX: read prefill from session_state so "Try →" in Examples tab works
        prefill = st.session_state.pop("ex_text", "")
        user_text = st.text_area(
            "input_text",
            value=prefill,
            placeholder="Type or paste a sentence, tweet, or paragraph...",
            height=200,
            label_visibility="collapsed"
        )
        word_count = len(user_text.split()) if user_text.strip() else 0
        st.caption(f"{word_count} words · {len(user_text)} characters")
        analyze_btn = st.button("Analyze Emotion →", key="single", use_container_width=True)

    with col2:
        st.subheader("Result")
        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing..."):
                emo, conf, probs = predict_emotion(user_text, MODELS[model_choice], TFIDF, ENCODER)
            # BUG 2 FIX: pass current theme so card gradient uses correct end color
            st.markdown(render_result_card(emo, conf, theme=st.session_state.theme), unsafe_allow_html=True)
            st.markdown("#### Probability Distribution")
            st.markdown(render_probability_bars(probs, emo), unsafe_allow_html=True)
        elif analyze_btn:
            st.warning("Please enter some text.")
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:260px; color:var(--muted);">
                <div style="font-size:3rem; opacity:0.5;">🎯</div>
                <p>Enter text on the left and click <strong>Analyze Emotion</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ---------- TAB 2: Batch Processing ----------
with tab2:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Upload file or paste")
        uploaded_file = st.file_uploader("Choose file", type=["csv", "txt", "pdf", "docx"])
        st.markdown("<p style='text-align:center; color:var(--muted);'>— or —</p>", unsafe_allow_html=True)
        pasted_text = st.text_area(
            "batch_paste",
            placeholder="One sentence per line...",
            height=150,
            label_visibility="collapsed"
        )
        run_batch = st.button("Run Batch Analysis →", key="batch", use_container_width=True)

    with col2:
        st.subheader("Results")
        if run_batch:
            texts = []
            if uploaded_file:
                try:
                    texts.extend(extract_texts_from_file(uploaded_file))
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            if pasted_text.strip():
                texts.extend([line.strip() for line in pasted_text.split("\n") if len(line.strip()) > 8])

            texts = list(dict.fromkeys(texts))
            if not texts:
                st.warning("No text found.")
            else:
                MAX = 200
                if len(texts) > MAX:
                    texts = texts[:MAX]
                    st.info(f"Showing first {MAX} texts.")

                rows = []
                progress_bar = st.progress(0, "Starting...")
                for i, t in enumerate(texts):
                    try:
                        emo, conf, _ = predict_emotion(t, MODELS[model_choice], TFIDF, ENCODER)
                        rows.append({
                            "#": i+1,
                            "Emotion": emo,
                            "Emoji": EMOTION_PALETTE.get(emo, {}).get("emoji", "❓"),
                            "Confidence": conf,
                            "Text": t[:100] + ("…" if len(t) > 100 else "")
                        })
                    except Exception:
                        rows.append({"#": i+1, "Emotion": "error", "Emoji": "⚠️", "Confidence": 0.0, "Text": t[:100]})
                    progress_bar.progress((i+1)/len(texts), text=f"Analyzed {i+1}/{len(texts)}")
                progress_bar.empty()

                df = pd.DataFrame(rows)
                counts = df["Emotion"].value_counts()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total", len(df))
                top_emo = counts.index[0] if len(counts) else "—"
                m2.metric("Top Emotion", f"{EMOTION_PALETTE.get(top_emo, {}).get('emoji','❓')} {top_emo}")
                m3.metric("Unique", len(counts))
                avg_conf = df[df["Emotion"] != "error"]["Confidence"].mean() * 100
                m4.metric("Avg Confidence", f"{avg_conf:.1f}%")

                st.markdown("#### Distribution")
                dist_html = ""
                for emo, cnt in counts.items():
                    cfg = EMOTION_PALETTE.get(emo, {"color": "#5b6ef5", "emoji": "❓"})
                    pct = cnt / len(df) * 100
                    dist_html += (
                        f'<div class="prob-row">'
                        f'<div class="prob-emo" style="color:{cfg["color"]}">{cfg["emoji"]} {emo}</div>'
                        f'<div class="prob-bg"><div class="prob-fill" style="width:{pct:.1f}%;background:{cfg["color"]};"></div></div>'
                        f'<div class="prob-val">{cnt} ({pct:.0f}%)</div>'
                        f'</div>'
                    )
                st.markdown(dist_html, unsafe_allow_html=True)

                st.markdown("#### Detailed Results")
                display_df = df[["#", "Emoji", "Emotion", "Confidence", "Text"]].copy()
                display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

                # BUG 5 FIX: export also uses formatted confidence, not raw float
                export_df = df[["#", "Emotion", "Confidence", "Text"]].copy()
                export_df["Confidence"] = export_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
                csv = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download CSV", data=csv, file_name="emotion_results.csv", mime="text/csv")
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:280px; color:var(--muted);">
                <div style="font-size:3rem; opacity:0.5;">📂</div>
                <p>Upload a file or paste text, then click <strong>Run Batch Analysis</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ---------- TAB 3: Examples ----------
with tab3:
    st.subheader("Try an example")
    EXAMPLES = {
        "joy": [
            "I just got the job I always dreamed of — I'm over the moon!",
            "This is the best birthday surprise ever!",
            "We won the championship! I'm so proud."
        ],
        "sadness": [
            "I feel so empty and lost.",
            "I miss him so much it hurts.",
            "Everything fell apart today."
        ],
        "anger": [
            "I'm absolutely furious they lied to me.",
            "This is outrageous!",
            "How dare they ignore me."
        ],
        "fear": [
            "I'm terrified of what might happen.",
            "The results come tomorrow and I can't sleep.",
            "I'm scared of losing everything."
        ],
        "love": [
            "You are the best thing in my life.",
            "My daughter smiled and my world lit up.",
            "I love my family more than anything."
        ],
        "surprise": [
            "Wait — they agreed? I didn't see that coming!",
            "She showed up after five years. Speechless.",
            "I won the lottery. Unbelievable."
        ],
    }

    for emo, sentences in EXAMPLES.items():
        cfg = EMOTION_PALETTE[emo]
        st.markdown(f"##### {cfg['emoji']} {emo.capitalize()}")
        cols = st.columns(3)
        for idx, s in enumerate(sentences):
            with cols[idx]:
                st.markdown(f"<div style='background:{cfg['bg']}; padding:12px; border-radius:12px; border:1px solid {cfg['color']}33;'>{s}</div>", unsafe_allow_html=True)
                if st.button("Try →", key=f"ex_{emo}_{idx}"):
                    st.session_state["ex_text"] = s
                    st.rerun()

    st.divider()
    all_examples = [s for lst in EXAMPLES.values() for s in lst]
    selected = st.selectbox("Or choose from list:", all_examples)
    if st.button("Analyze Selected Example"):
        with st.spinner("Analyzing..."):
            emo, conf, probs = predict_emotion(selected, MODELS[model_choice], TFIDF, ENCODER)
        c1, c2 = st.columns(2)
        with c1:
            # BUG 2 FIX: pass current theme so card gradient uses correct end color
            st.markdown(render_result_card(emo, conf, theme=st.session_state.theme), unsafe_allow_html=True)
            st.caption(f"\"{selected}\"")
        with c2:
            st.markdown("#### Probabilities")
            st.markdown(render_probability_bars(probs, emo), unsafe_allow_html=True)