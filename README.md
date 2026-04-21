# 🎭 Emotion Detection System

A professional **Streamlit web application** that detects emotions in text using trained Machine Learning models. Built with a clean modular architecture, batch file processing, confidence visualization, and full dark/light mode support.

---

## 📁 Project Structure

```
EmotionDetector/
│
├── App/
│   └── main.py                        ← Streamlit app entry point
│
├── Data/
│   └── train.txt                      ← Training dataset (text;emotion format)
│
├── Models/
│   ├── tfidf_vectorizer.joblib        ← TF-IDF vectorizer
│   ├── label_encoder.joblib           ← Sklearn LabelEncoder
│   ├── linearsvc_model.joblib         ← Linear SVM classifier
│   ├── logistic_regression_model.joblib
│   └── xgboost_model.joblib           ← XGBoost classifier
│
├── NoteBooks/
│   └── ProjectFile.ipynb              ← Training & EDA notebook
│
├── src/                               ← Modular backend package
│   ├── __init__.py                    ← Package exports
│   ├── config.py                      ← Paths, filenames, emotion palette
│   ├── preprocess.py                  ← Text cleaning & sentence splitting
│   ├── file_processor.py              ← CSV / TXT / PDF / DOCX extractors
│   ├── model_loader.py                ← Joblib model loading (cached)
│   ├── predictor.py                   ← Prediction + probability logic
│   └── visualizer.py                  ← Matplotlib confidence bar chart
│
├── requirements.txt
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/EmotionDetector.git
cd EmotionDetector
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

> ⚠️ **Always run from the project root** (`EmotionDetector/`), not from inside `App/`.

```bash
streamlit run App/main.py
```

The app opens at **http://localhost:8501**

---

## 📦 Requirements

```
streamlit>=1.32.0
joblib>=1.3.0
scikit-learn>=1.3.0
xgboost>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
PyPDF2>=3.0.0
python-docx>=1.1.0
```

Install everything at once:

```bash
pip install -r requirements.txt
```

> **Optional libraries:**
> - `PyPDF2` — needed for PDF file upload support
> - `python-docx` — needed for Word (.docx) file upload support
>
> The app still works without them — it just disables those file types and shows an install hint.

---

## 🎯 Features

### 🔍 Single Text Tab
- Type or paste any sentence, tweet, review, or paragraph
- Instant emotion prediction with emoji + label
- Confidence percentage display
- Full **6-emotion bar chart** showing probability distribution for all classes

### 📂 Batch Processing Tab
- Upload **CSV, TXT, PDF, or DOCX** files
- Or type multiple sentences directly
- Live progress bar during analysis
- Results table with emotion + confidence per row
- **Download results as CSV** with timestamp in filename
- Emotion distribution bar chart summary

### 💡 Examples Tab
- 10 pre-written example sentences spanning all 6 emotions
- Click **Analyze** on any example to see prediction + chart
- Only shows one chart at a time (no page clutter)
- Download sample CSV or TXT files for batch testing

---

## 🤖 Models

| Model | Type | Vectorizer | Speed |
|---|---|---|---|
| **Logistic Regression** | Linear Classifier | TF-IDF | ★★★ Fast |
| **Linear SVM** | Linear SVM (LinearSVC) | TF-IDF | ★★★ Fastest |
| **XGBoost** | Gradient Boosting | TF-IDF | ★★ Moderate |

Switch between models using the **sidebar dropdown** — all three are loaded at startup and cached for instant switching.

---

## 🎭 Emotion Classes

| Emotion | Emoji | Color |
|---|---|---|
| Anger | 😠 | Red `#FF4444` |
| Fear | 😨 | Purple `#9B59B6` |
| Joy | 😊 | Yellow `#F1C40F` |
| Love | ❤️ | Pink `#FF6B8A` |
| Sadness | 😢 | Blue `#3498DB` |
| Surprise | 😲 | Green `#2ECC71` |

---

## 🧱 Architecture

The `src/` package is split into single-responsibility modules:

```
main.py  (Streamlit UI)
    │
    ├── src/config.py          →  All constants: paths, filenames, colors, emojis
    ├── src/preprocess.py      →  clean_text(), split_into_sentences()
    ├── src/file_processor.py  →  process_uploaded_file() → list of strings
    ├── src/model_loader.py    →  load_vectorizer(), load_encoder(), load_all_models()
    ├── src/predictor.py       →  predict_emotion() → (label, confidence, probs_dict)
    └── src/visualizer.py      →  plot_confidence_scores() → matplotlib Figure
```

### Text Preprocessing Pipeline

Every input goes through this sequence before prediction:

```
Raw text
  → lowercase
  → remove punctuation
  → remove digits
  → remove extra whitespace
  → TF-IDF vectorize
  → model.predict()
  → label_encoder.inverse_transform()
  → emotion + confidence + probability distribution
```

### Probability Handling

| Model | Method |
|---|---|
| Logistic Regression | `predict_proba()` — native probabilities |
| XGBoost | `predict_proba()` — native probabilities |
| Linear SVM | `decision_function()` → softmax — all 6 class scores converted to probabilities |

This ensures the confidence chart always shows all 6 bars regardless of the selected model.

---

## 📋 Batch File Format Guide

| Format | How it's processed |
|---|---|
| **CSV** | Uses `text` column if present; falls back to first text-type column |
| **TXT** | Split into sentences at `.` `!` `?` boundaries |
| **PDF** | Text extracted page by page, split into sentences |
| **DOCX** | Each paragraph becomes one text sample |

**Maximum batch size:** 200 sentences per run (configurable in `main.py`).

---

## 🛠️ Bugs Fixed (v1.1)

| # | File | Bug | Impact | Fix |
|---|---|---|---|---|
| 1 | `config.py` | `MODELS_DIR = 'models'` (wrong case) | 💥 Crash on Linux/macOS | Changed to `'Models'` |
| 2 | `config.py` | All 3 model filenames were wrong | 💥 Models never load | Fixed to match actual `.joblib` filenames |
| 3 | `config.py` | `VECTORIZER_FILE = 'vectorizer_BoW.joblib'` | 💥 Vectorizer never loads | Fixed to `tfidf_vectorizer.joblib` |
| 4 | `config.py` | `ENCODER_FILE = 'encoder.joblib'` | 💥 Encoder never loads | Fixed to `label_encoder.joblib` |
| 5 | `predictor.py` | LinearSVC has no `predict_proba` → chart showed 1 bar | 🔴 Wrong output | Added `decision_function` + softmax for all 6 classes |
| 6 | `predictor.py` | Hardcoded `{0:'anger'...}` fallback when encoder is `None` | 🔴 Wrong output | Raises clear `ValueError` if encoder missing |
| 7 | `file_processor.py` | Generic "unsupported" error when library missing | 🟡 Confusing UX | Shows exact `pip install` command needed |
| 8 | `visualizer.py` | White matplotlib background broke dark mode | 🟡 Visual glitch | Set transparent figure + axes background |
| 9 | `main.py` Tab3 | All previous example charts re-rendered on every click | 🟡 Page clutter | Only last-clicked example shows its chart |

---

## 🔮 Possible Future Improvements

- Add **confusion matrix** visualization from test set
- Support **multilingual** text input
- Add **word cloud** per emotion class
- REST API wrapper with **FastAPI**
- **Streamlit Cloud** deployment with `secrets.toml`
- Hyperparameter tuning results display
- User feedback / correction loop

---

## 👨‍💻 Author

Built as part of an **ML/NLP learning project** by **Taimoor Tahir**

Tech stack: Python · Scikit-learn · XGBoost · Streamlit · Pandas · Matplotlib · Joblib

---

## 📄 License

MIT License — free to use and modify.
