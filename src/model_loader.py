# src/model_loader.py
import os
import joblib
from .config import MODEL_PATHS

# Get the absolute path to the project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_all_models():
    """Load TF‑IDF vectorizer, label encoder, and all available models."""
    tfidf_path = os.path.join(PROJECT_ROOT, MODEL_PATHS["tfidf"])
    encoder_path = os.path.join(PROJECT_ROOT, MODEL_PATHS["encoder"])
    
    tfidf = joblib.load(tfidf_path)
    encoder = joblib.load(encoder_path)
    
    models = {}
    for name in ["⚡ LinearSVC", "📈 Logistic Regression", "🌲 XGBoost"]:
        model_path = os.path.join(PROJECT_ROOT, MODEL_PATHS[name])
        try:
            models[name] = joblib.load(model_path)
        except Exception:
            pass
    return tfidf, encoder, models