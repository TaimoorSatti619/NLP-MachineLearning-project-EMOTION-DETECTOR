from pathlib import Path

# Project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent

# ── BUG-1 FIX: was 'models' (lowercase) → folder is 'Models' (capital M)
# On Linux/macOS the filesystem is case-sensitive → FileNotFoundError on wrong case
MODELS_DIR = PROJECT_ROOT / 'Models'

# ── BUG-2 FIX: filenames were wrong (LogisticRegression_model.joblib etc.)
# Corrected to match actual files saved on disk by the notebook
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression_model.joblib',
    'Linear SVM':          'linearsvc_model.joblib',
    'XGBoost':             'xgboost_model.joblib',
}

# ── BUG-3 FIX: was 'vectorizer_BoW.joblib' → actual file is tfidf_vectorizer.joblib
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# ── BUG-4 FIX: was 'encoder.joblib' → actual file is label_encoder.joblib
ENCODER_FILE = 'label_encoder.joblib'

# Emotion colors for visualization
EMOTION_COLORS = {
    'anger':    '#FF4444',
    'fear':     '#9B59B6',
    'joy':      '#F1C40F',
    'love':     '#FF6B8A',
    'sadness':  '#3498DB',
    'surprise': '#2ECC71',
}

# Emotion emojis
EMOTION_EMOJIS = {
    'anger':    '😠',
    'fear':     '😨',
    'joy':      '😊',
    'love':     '❤️',
    'sadness':  '😢',
    'surprise': '😲',
}
