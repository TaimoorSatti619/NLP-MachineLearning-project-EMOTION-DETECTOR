import streamlit as st
import joblib
from .config import MODELS_DIR, MODEL_FILES, VECTORIZER_FILE, ENCODER_FILE


@st.cache_resource
def load_vectorizer():
    """Load TF-IDF vectorizer from disk."""
    path = MODELS_DIR / VECTORIZER_FILE
    if path.exists():
        return joblib.load(path)
    st.error(f"Vectorizer not found at: {path}")
    return None


@st.cache_resource
def load_encoder():
    """Load label encoder from disk."""
    path = MODELS_DIR / ENCODER_FILE
    if path.exists():
        return joblib.load(path)
    st.error(f"Encoder not found at: {path}")
    return None


@st.cache_resource
def load_all_models() -> dict:
    """Load all classifier models. Missing files are skipped with a warning."""
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
        else:
            st.warning(f"Model file not found, skipping '{name}': {path}")
    return models
