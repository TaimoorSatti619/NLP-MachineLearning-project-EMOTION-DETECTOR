# src/predictor.py
import numpy as np
from .preprocess import preprocess_text


def predict_emotion(text: str, model, tfidf, encoder):
    """Return predicted emotion, confidence, and probability dict."""
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    label = encoder.inverse_transform([pred])[0]

    # Probabilities
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(vec)[0]
        probs = {encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(raw)}
    elif hasattr(model, "decision_function"):
        dv = model.decision_function(vec)[0]
        exp = np.exp(dv - dv.max())
        raw = exp / exp.sum()
        probs = {encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(raw)}
    else:
        probs = {c: (1.0 if c == label else 0.0) for c in encoder.classes_}

    return label, probs[label], probs