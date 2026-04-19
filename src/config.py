# src/config.py

EMOTION_PALETTE = {
    "anger":    {"emoji": "😡", "color": "#ff5c5c", "bg": "rgba(255,92,92,0.08)"},
    "fear":     {"emoji": "😨", "color": "#c084fc", "bg": "rgba(192,132,252,0.08)"},
    "joy":      {"emoji": "😄", "color": "#f5c542", "bg": "rgba(245,197,66,0.08)"},
    "love":     {"emoji": "❤️", "color": "#f97da8", "bg": "rgba(249,125,168,0.08)"},
    "sadness":  {"emoji": "😢", "color": "#4db8ff", "bg": "rgba(77,184,255,0.08)"},
    "surprise": {"emoji": "😲", "color": "#34d9c3", "bg": "rgba(52,217,195,0.08)"},
}

MODEL_PATHS = {
    "tfidf": "Models/tfidf_vectorizer.joblib",
    "encoder": "Models/label_encoder.joblib",
    "⚡ LinearSVC": "Models/linearsvc_model.joblib",
    "📈 Logistic Regression": "Models/logistic_regression_model.joblib",
    "🌲 XGBoost": "Models/xgboost_model.joblib",
}

MODEL_META = {
    "⚡ LinearSVC": ("Linear SVM", "TF‑IDF", "★★★ Fastest", "Best for sparse text."),
    "📈 Logistic Regression": ("Linear Classifier", "TF‑IDF", "★★★ Fast", "Interpretable probabilities."),
    "🌲 XGBoost": ("Gradient Boosting", "TF‑IDF", "★★ Moderate", "Handles non‑linear patterns."),
}