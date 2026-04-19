# 🧠 EmotionDetector AI

A professional **Streamlit web application** for detecting emotions in text using trained Machine Learning models. Built with a clean modular architecture, dark/light theme support, batch file processing, and real-time probability visualization.

---

## 📁 Project Structure

```
EmotionDetector/
│
├── App/
│   └── main.py                  ← Streamlit app entry point
│
├── Data/
│   └── train.txt                ← Training dataset (text;emotion format)
│
├── Models/
│   ├── tfidf_vectorizer.joblib  ← TF-IDF vectorizer
│   ├── label_encoder.joblib     ← Sklearn LabelEncoder
│   ├── linearsvc_model.joblib   ← LinearSVC classifier
│   ├── logistic_regression_model.joblib
│   └── xgboost_model.joblib     ← XGBoost classifier
│
├── NoteBooks/
│   └── ProjectFile.ipynb        ← Training & EDA notebook
│
├── src/                         ← Modular backend package
│   ├── __init__.py              ← Package exports
│   ├── config.py                ← Emotion palette & model metadata
│   ├── file_processor.py        ← CSV / TXT / PDF / DOCX extractors
│   ├── model_loader.py          ← Joblib model loading
│   ├── predictor.py             ← Prediction + probability logic
│   ├── preprocess.py            ← Text cleaning pipeline
│   └── visualizer.py            ← HTML result card & probability bars
│
├── requirements.txt
├── README.md
├── .gitignore
└── pyproject.toml
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

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit>=1.32.0
joblib>=1.3.0
scikit-learn>=1.3.0
xgboost>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
nltk>=3.8.0
PyPDF2>=3.0.0
python-docx>=1.1.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

> **Note on XGBoost:** If `xgboost` is not installed, the XGBoost model is silently skipped and the app still works with LinearSVC and Logistic Regression.

> **Note on NLTK:** NLTK data (stopwords, punkt, wordnet) is downloaded automatically on first run. If you're offline, preprocessing falls back to basic regex cleaning.

---

## 🎯 Features

### 🔍 Single Input Tab
- Type or paste any text
- Get instant emotion prediction
- See animated **probability distribution bars** for all 6 emotions
- Confidence badge: Very High / High / Moderate / Low

### 📂 Batch Processing Tab
- Upload **CSV, TXT, PDF, or DOCX** files
- Or paste multiple lines (one per line)
- Live progress bar during analysis
- Summary metrics: total, dominant emotion, unique emotions, avg confidence
- Distribution bar chart across all predictions
- Full results table
- **Download results as CSV**

### 💡 Examples Tab
- 18 curated example sentences across all 6 emotions
- Click **"Try →"** to instantly prefill the Single Input tab
- Quick-analyze dropdown with all examples

### 🌗 Dark / Light Theme
- Toggle between dark and light mode from the sidebar
- All components adapt including result cards and probability bars

### 🤖 3 Models
| Model | Type | Speed | Best For |
|---|---|---|---|
| ⚡ LinearSVC | Linear SVM | ★★★ Fastest | Sparse text — top performer |
| 📈 Logistic Regression | Linear Classifier | ★★★ Fast | Interpretable probabilities |
| 🌲 XGBoost | Gradient Boosting | ★★ Moderate | Non-linear patterns |

---

## 🎭 Emotion Classes

| Emotion | Emoji | Description |
|---|---|---|
| Joy | 😄 | Happiness, excitement, pride |
| Sadness | 😢 | Grief, loneliness, loss |
| Anger | 😡 | Fury, frustration, betrayal |
| Fear | 😨 | Anxiety, terror, dread |
| Love | ❤️ | Affection, warmth, care |
| Surprise | 😲 | Shock, disbelief, astonishment |

---

## 🧱 Architecture

The `src/` package follows a clean separation of concerns:

```
main.py (UI)
  │
  ├── src/config.py          → EMOTION_PALETTE, MODEL_META, MODEL_PATHS constants
  ├── src/model_loader.py    → load_all_models() → returns tfidf, encoder, models dict
  ├── src/preprocess.py      → preprocess_text() → clean + tokenize + lemmatize
  ├── src/predictor.py       → predict_emotion() → label, confidence, prob_dict
  ├── src/file_processor.py  → extract_texts_from_file() → list of strings
  └── src/visualizer.py      → render_result_card(), render_probability_bars() → HTML
```

### Text Preprocessing Pipeline

Every input goes through this pipeline before prediction:

```
Raw text
  → lowercase
  → remove punctuation
  → remove numbers
  → remove URLs
  → remove HTML tags
  → remove non-ASCII characters
  → tokenize (NLTK)
  → remove stopwords
  → lemmatize (WordNetLemmatizer)
  → TF-IDF vectorize
  → model.predict()
```

---

## 🔢 Batch File Format Guide

| Format | How it's processed |
|---|---|
| **CSV** | First column with `object` dtype is used automatically |
| **TXT** | Each line becomes one text sample (lines < 8 chars skipped) |
| **PDF** | Text extracted page by page, line by line |
| **DOCX** | Each paragraph becomes one text sample |

Maximum batch size: **200 texts per run**.

---

## 🛠️ Bugs Fixed (v1.1)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `App/main.py` | `ex_text` set in session_state but Tab1 `text_area` never read it — "Try →" button had no effect | Added `value=st.session_state.pop("ex_text", "")` to the `text_area` |
| 2 | `src/visualizer.py` | Hardcoded `#0f1626` dark gradient in `render_result_card` broke light mode | Made gradient end color theme-aware via new `theme` parameter |
| 3 | `App/main.py` | After rerun, `user_text` always reset to empty (no `value=` binding) | Solved by Bug 1 fix — `value=` now binds correctly |
| 4 | `App/main.py` | Sidebar model info used `var(--s2)` and `var(--muted)` CSS vars inside `st.markdown` HTML — unreliable in light mode | Replaced with concrete hex values computed from current theme |
| 5 | `App/main.py` | Download CSV exported raw float Confidence (0.0–1.0) while table showed strings like `"87.3%"` | Added `export_df` with formatted Confidence before encoding |
| 6 | `src/__init__.py` | `MODEL_PATHS` defined in `config.py` but not exported from the package | Added `MODEL_PATHS` to `__init__.py` exports and `__all__` |

---

## 🔮 Possible Future Improvements

- Add a **confusion matrix** visualization tab
- Support **multilingual** emotion detection
- Add **Streamlit Cloud** deployment config (`secrets.toml`)
- Hyperparameter tuning results display
- Word cloud per emotion class
- REST API wrapper with FastAPI

---

## 👨‍💻 Author

Built as part of an **ML/NLP learning project** using:
- Python · Scikit-learn · XGBoost · NLTK
- Streamlit · Pandas · Joblib
- TF-IDF vectorization · Label Encoding

---

## 📄 License

MIT License — free to use and modify.
