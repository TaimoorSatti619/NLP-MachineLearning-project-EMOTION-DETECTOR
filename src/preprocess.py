import re
import string


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove punctuation, digits, extra spaces."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = ' '.join(text.split())
    return text


def split_into_sentences(text: str) -> list:
    """Split a block of text into individual non-empty sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)   # split AFTER punctuation + whitespace
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences
