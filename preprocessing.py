"""
Text preprocessing utilities shared by train.py and app.py.
Kept dependency-light (no downloaded corpora) so it works anywhere.
"""

import re

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+")
_PHONE_RE = re.compile(r"\b\d{5,}\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, normalize URLs/emails/numbers, strip punctuation, and stem."""
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _PHONE_RE.sub(" numtoken ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    tokens = [_stemmer.stem(tok) for tok in text.split()]
    return " ".join(tokens)


def message_length_features(text: str) -> dict:
    """Extra signal features that TF-IDF alone doesn't capture."""
    return {
        "char_count": len(text),
        "exclaim_count": text.count("!"),
        "upper_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
    }
