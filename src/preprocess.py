# src/preprocess.py
import re
import string

# Optional NLTK
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
        nltk.download(pkg, quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception:
    pass


def preprocess_text(text: str) -> str:
    """Clean and normalize input text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = "".join(c for c in text if c.isascii())
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            text = " ".join(
                LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS
            )
        except Exception:
            pass
    return text.strip()