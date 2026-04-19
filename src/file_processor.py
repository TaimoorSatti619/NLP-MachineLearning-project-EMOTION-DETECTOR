# src/file_processor.py
import pandas as pd


def extract_texts_from_file(uploaded_file):
    """Extract list of text lines from CSV, TXT, PDF, or DOCX file."""
    name = uploaded_file.name.lower()
    texts = []

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            if df[col].dtype == object:
                texts = df[col].dropna().astype(str).tolist()
                break
    elif name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        texts = [line.strip() for line in content.split("\n") if len(line.strip()) > 8]
    elif name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                for line in page_text.split("\n"):
                    if len(line.strip()) > 8:
                        texts.append(line.strip())
        except ImportError:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(uploaded_file)
            texts = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 8]
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
    else:
        raise ValueError(f"Unsupported file type: {name}")

    return list(dict.fromkeys(texts))  # remove duplicates