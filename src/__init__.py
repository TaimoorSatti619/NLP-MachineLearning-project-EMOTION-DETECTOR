# src/__init__.py
from .config import (
    PROJECT_ROOT, MODELS_DIR, MODEL_FILES,
    VECTORIZER_FILE, ENCODER_FILE,
    EMOTION_COLORS, EMOTION_EMOJIS,
)
from .preprocess import clean_text, split_into_sentences
from .file_processor import process_uploaded_file, PDF_SUPPORT, DOCX_SUPPORT
from .model_loader import load_vectorizer, load_encoder, load_all_models
from .predictor import predict_emotion
from .visualizer import get_emotion_color, get_emotion_emoji, plot_confidence_scores

__all__ = [
    # config
    'PROJECT_ROOT', 'MODELS_DIR', 'MODEL_FILES',
    'VECTORIZER_FILE', 'ENCODER_FILE',
    'EMOTION_COLORS', 'EMOTION_EMOJIS',
    # preprocess
    'clean_text', 'split_into_sentences',
    # file processor
    'process_uploaded_file', 'PDF_SUPPORT', 'DOCX_SUPPORT',
    # model loader
    'load_vectorizer', 'load_encoder', 'load_all_models',
    # predictor
    'predict_emotion',
    # visualizer
    'get_emotion_color', 'get_emotion_emoji', 'plot_confidence_scores',
]
