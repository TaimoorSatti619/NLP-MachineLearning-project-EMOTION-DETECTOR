import sys
import os
from pathlib import Path

# Add project root to Python path so 'src' package is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from src import (
    load_vectorizer, load_encoder, load_all_models,
    process_uploaded_file, predict_emotion,
    get_emotion_color, get_emotion_emoji, plot_confidence_scores,
    PDF_SUPPORT, DOCX_SUPPORT,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🎭",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #FFFFFF; margin: 0.5rem 0 0 0; }

    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
        width: 100%;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .emotion-title { font-size: 3rem; font-weight: bold; margin: 0.5rem 0; }

    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 25px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        transition: all 0.3s ease; width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    .footer {
        text-align: center; padding: 2rem; color: #666;
        font-size: 0.8rem; border-top: 1px solid #e0e0e0; margin-top: 2rem;
    }

    /* Dark-mode-safe utility classes */
    .example-card {
        background-color: var(--secondary-background-color);
        padding: 10px; border-radius: 8px; margin: 5px 0;
        border: 1px solid var(--border-color); color: var(--text-color);
    }
    .info-box {
        background-color: var(--secondary-background-color);
        padding: 10px; border-radius: 8px; margin: 10px 0;
        font-size: 13px; border: 1px solid var(--border-color);
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 Emotion Detection System</h1>
    <p>Advanced NLP-based emotion recognition from text using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    vectorizer = load_vectorizer()
    encoder    = load_encoder()
    models     = load_all_models()

if not vectorizer or not models:
    st.error(
        "❌ Could not load models. "
        "Make sure all .joblib files are in the **Models/** folder "
        "and that the filenames match exactly."
    )
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Emotion Detection")
    st.markdown("---")

    st.markdown("### 🤖 Model")
    selected_model = st.selectbox("", list(models.keys()))

    st.markdown("---")
    st.markdown("### 📊 Emotions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("😠 **Anger**")
        st.markdown("😨 **Fear**")
        st.markdown("😊 **Joy**")
    with col2:
        st.markdown("❤️ **Love**")
        st.markdown("😢 **Sadness**")
        st.markdown("😲 **Surprise**")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Detects emotions from text using Machine Learning "
        "with TF-IDF features and three trained classifiers."
    )

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Text", "📂 Batch Processing", "💡 Examples"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Text
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Text for Analysis")

    user_input = st.text_area(
        "",
        placeholder="Example: I am really excited about this new project!",
        height=120,
        key="text_input",
    )

    if st.button("🔍 Analyze Emotion", use_container_width=True, key="analyze_single"):
        if user_input and len(user_input.strip()) >= 3:
            with st.spinner("Analyzing text…"):
                model = models[selected_model]
                emotion, confidence, probabilities = predict_emotion(
                    user_input, model, vectorizer, encoder
                )

            if emotion:
                color = get_emotion_color(emotion)
                emoji = get_emotion_emoji(emotion)

                st.markdown(f"""
                <div class="result-box" style="background-color:{color}15; border:2px solid {color};">
                    <div style="font-size:1.2rem; color:{color}; font-weight:bold;">Detected Emotion</div>
                    <div class="emotion-title" style="color:{color};">{emoji} {emotion.upper()}</div>
                    <div style="font-size:1.2rem; color:{color};">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Confidence Distribution")
                fig = plot_confidence_scores(probabilities)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.error("Unable to analyze. Please try different text.")

        elif user_input:
            st.warning("Please enter at least 3 characters.")
        else:
            st.info("Enter some text and click **Analyze Emotion**.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Processing
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload File for Batch Analysis")

    # Show library warnings upfront
    if not PDF_SUPPORT:
        st.warning("⚠️ PDF support not available. Run `pip install PyPDF2` to enable it.")
    if not DOCX_SUPPORT:
        st.warning("⚠️ DOCX support not available. Run `pip install python-docx` to enable it.")

    st.markdown("""
    <div class="info-box">
        📁 Supported formats: <strong>CSV</strong> (needs a <em>text</em> column) &nbsp;·&nbsp;
        <strong>TXT</strong> &nbsp;·&nbsp; <strong>PDF</strong> &nbsp;·&nbsp; <strong>DOCX</strong>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt', 'pdf', 'docx'],
        help="Upload CSV (with 'text' column), TXT, PDF, or Word document",
    )

    if uploaded_file:
        with st.spinner("Processing file…"):
            sentences = process_uploaded_file(uploaded_file)

        if sentences:
            st.success(f"✅ Loaded {len(sentences)} sentence(s)")

            with st.expander("Preview (first 10 sentences)"):
                for i, sent in enumerate(sentences[:10]):
                    st.write(f"{i+1}. {sent[:150]}{'…' if len(sent)>150 else ''}")

            if st.button("🚀 Start Analysis", use_container_width=True, key="batch_run"):
                model = models[selected_model]
                rows = []
                progress_bar = st.progress(0, text="Starting…")

                for idx, sent in enumerate(sentences):
                    emotion, confidence, _ = predict_emotion(
                        sent, model, vectorizer, encoder
                    )
                    rows.append({
                        'Text':       sent[:100] + ("…" if len(sent) > 100 else ""),
                        'Emotion':    emotion or "error",
                        'Confidence': f"{confidence:.2%}",
                    })
                    progress_bar.progress(
                        (idx + 1) / len(sentences),
                        text=f"Analyzed {idx+1} / {len(sentences)}",
                    )

                progress_bar.empty()
                results_df = pd.DataFrame(rows)

                st.success("✅ Analysis complete!")
                st.dataframe(results_df, use_container_width=True)

                # Download results
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "📥 Download Results (CSV)",
                    results_df.to_csv(index=False),
                    f"emotion_results_{ts}.csv",
                    "text/csv",
                )

                st.markdown("#### Emotion Distribution")
                st.bar_chart(results_df['Emotion'].value_counts())

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Examples
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Test with Examples")

    EXAMPLES = [
        "I just got promoted at work and couldn't be happier!",
        "I feel so empty and lost, nothing makes sense anymore.",
        "I'm absolutely furious — they broke my trust completely.",
        "The thought of this makes me so anxious I can't sleep.",
        "I love my family more than anything in this world.",
        "Wait, they actually agreed to this? I never saw it coming!",
        "This is the best birthday surprise I have ever had.",
        "I miss him so much, the grief is completely overwhelming.",
        "How dare they ignore my concerns without even listening.",
        "I'm scared of losing everything I've worked for.",
    ]

    # ── BUG-9 FIX: old code stored ALL results in session_state and re-rendered
    # every chart on every rerun — the page became a wall of bar charts.
    # Fix: track only the LAST clicked example index + its result.
    # Only that one example shows the chart; all others just show the card.
    if 'selected_ex' not in st.session_state:
        st.session_state.selected_ex     = None   # index of last-clicked example
        st.session_state.selected_result = None   # (emotion, confidence, probs)

    for idx, example in enumerate(EXAMPLES):
        col_text, col_btn = st.columns([5, 1])

        with col_text:
            st.markdown(
                f'<div class="example-card"><strong>{idx+1}.</strong> {example}</div>',
                unsafe_allow_html=True,
            )

        with col_btn:
            if st.button("Analyze", key=f"ex_{idx}", use_container_width=True):
                model = models[selected_model]
                emotion, confidence, probs = predict_emotion(
                    example, model, vectorizer, encoder
                )
                # Store only the most recently clicked result
                st.session_state.selected_ex     = idx
                st.session_state.selected_result = (emotion, confidence, probs)
                st.rerun()

        # Only show result for the ONE most recently analyzed example
        if st.session_state.selected_ex == idx and st.session_state.selected_result:
            emotion, confidence, probs = st.session_state.selected_result
            color = get_emotion_color(emotion)
            emoji = get_emotion_emoji(emotion)

            st.markdown(f"""
            <div style="background-color:{color}15; padding:12px; border-radius:8px;
                        margin-bottom:6px; border-left:4px solid {color};">
                <strong style="color:{color};">{emoji} {emotion.upper()}</strong>
                <span style="color:#888; margin-left:12px;">Confidence: {confidence:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

            # One chart, only for the selected example
            fig = plot_confidence_scores(probs, title=f"Distribution — Example {idx+1}")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")

    # ── Sample file downloads ─────────────────────────────────────────────────
    st.markdown("### 📥 Download Sample Files")
    dl1, dl2 = st.columns(2)
    with dl1:
        sample_df = pd.DataFrame({'text': EXAMPLES})
        st.download_button(
            "📥 Sample CSV",
            sample_df.to_csv(index=False),
            "sample_emotions.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📥 Sample TXT",
            "\n".join(EXAMPLES),
            "sample_emotions.txt",
            "text/plain",
            use_container_width=True,
        )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>🎭 <strong>Emotion Detection System</strong> | Built by <strong style="color:#667eea;">Shuban Ali</strong></p>
    <p style="font-size:0.7rem; opacity:0.7;">© 2026 | NLP-based Emotion Recognition</p>
</div>
""", unsafe_allow_html=True)
