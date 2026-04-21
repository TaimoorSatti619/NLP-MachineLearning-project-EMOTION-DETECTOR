import streamlit as st
import pandas as pd
from .preprocess import split_into_sentences

# ── Optional library detection ────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_pdf(file) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def _extract_docx(file) -> str:
    """Extract all text from a Word (.docx) file."""
    try:
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def process_uploaded_file(uploaded_file) -> list:
    """
    Auto-detect file type and return a list of non-empty text sentences.

    ── BUG-7 FIX: the old else-branch gave a generic 'unsupported file type'
    error even when the file type IS supported but the library is not installed.
    Now we distinguish between truly unsupported types and missing libraries,
    and give the user an actionable install command in each case.
    """
    name = uploaded_file.name.lower()

    # ── CSV ──────────────────────────────────────────────────────────────────
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            return df['text'].dropna().astype(str).tolist()
        else:
            # Fallback: use the first object-dtype column
            for col in df.columns:
                if df[col].dtype == object:
                    st.info(f"No 'text' column found — using column '{col}' instead.")
                    return df[col].dropna().astype(str).tolist()
            st.error("CSV has no text column. Add a column named 'text'.")
            return []

    # ── TXT ──────────────────────────────────────────────────────────────────
    elif name.endswith('.txt'):
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        return split_into_sentences(content)

    # ── PDF ──────────────────────────────────────────────────────────────────
    elif name.endswith('.pdf'):
        if not PDF_SUPPORT:
            st.error(
                "PDF support is not installed. Run:  `pip install PyPDF2`  "
                "and restart the app."
            )
            return []
        content = _extract_pdf(uploaded_file)
        return split_into_sentences(content)

    # ── DOCX ─────────────────────────────────────────────────────────────────
    elif name.endswith('.docx'):
        if not DOCX_SUPPORT:
            st.error(
                "Word file support is not installed. Run:  `pip install python-docx`  "
                "and restart the app."
            )
            return []
        content = _extract_docx(uploaded_file)
        return split_into_sentences(content)

    # ── Truly unsupported ─────────────────────────────────────────────────────
    else:
        ext = name.rsplit('.', 1)[-1].upper() if '.' in name else 'unknown'
        st.error(
            f"File type '.{ext}' is not supported. "
            "Please upload a CSV, TXT, PDF, or DOCX file."
        )
        return []
