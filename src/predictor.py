import numpy as np
from .preprocess import clean_text


def predict_emotion(text, model, vectorizer, encoder):
    """Predict emotion from text. Returns (emotion, confidence, probabilities_dict)."""

    cleaned = clean_text(text)
    if not cleaned:
        return None, 0.0, {}

    features = vectorizer.transform([cleaned])
    pred_label = model.predict(features)[0]

    # ── BUG-6 FIX: always use encoder, never rely on a hardcoded dict
    # Raise a clear error if encoder is missing so the problem is obvious
    if encoder is None:
        raise ValueError(
            "Label encoder is None. Check that label_encoder.joblib loaded correctly."
        )
    emotion = encoder.inverse_transform([pred_label])[0]

    # ── BUG-5 FIX: LinearSVC has no predict_proba.
    # Old code set probabilities = {} for that case → chart showed only 1 bar.
    # Now we use decision_function + softmax to get a probability for every class.
    if hasattr(model, 'predict_proba'):
        raw_probs = model.predict_proba(features)[0]
        probabilities = {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(raw_probs)
        }
    elif hasattr(model, 'decision_function'):
        # Softmax over decision scores → valid probability distribution
        decision_scores = model.decision_function(features)[0]
        exp_scores = np.exp(decision_scores - decision_scores.max())   # numerical stability
        softmax_probs = exp_scores / exp_scores.sum()
        probabilities = {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(softmax_probs)
        }
    else:
        # Absolute last resort: one-hot
        probabilities = {
            c: (1.0 if c == emotion else 0.0)
            for c in encoder.classes_
        }

    confidence = probabilities.get(emotion, 0.0)
    return emotion, confidence, probabilities
