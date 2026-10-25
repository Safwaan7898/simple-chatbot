"""
preprocess.py
-------------
Text preprocessing utilities for the NLP chatbot.

Covers:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stopword removal

These steps clean raw user input so it can be compared
against stored patterns using TF-IDF and cosine similarity.
"""

import string
import warnings

# ---------------------------------------------------------------------------
# Built-in stopword list (no external download required as a fallback)
# ---------------------------------------------------------------------------
BUILTIN_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "what", "which", "who", "how", "when", "where",
    "why", "so", "if", "as", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "then", "once", "just",
    "also", "than", "too", "very", "s", "t", "don", "now", "ll", "re"
}

# Try to load NLTK stopwords; fall back to built-in list if unavailable
def _load_stopwords() -> set[str]:
    """
    Load stopwords from NLTK if available, otherwise use the built-in list.

    Returns:
        set: A set of stopword strings.
    """
    try:
        import nltk
    except ImportError:
        return BUILTIN_STOPWORDS

    from nltk.corpus import stopwords
    try:
        return set(stopwords.words("english"))
    except LookupError:
        if nltk.download("stopwords", quiet=True):
            return set(stopwords.words("english"))

        warnings.warn(
            "NLTK stopwords could not be downloaded; using the built-in stopword list.",
            RuntimeWarning,
            stacklevel=2,
        )
        return BUILTIN_STOPWORDS


STOPWORDS = _load_stopwords()


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """
    Convert all characters in text to lowercase.

    Args:
        text (str): Raw input string.

    Returns:
        str: Lowercased string.

    Example:
        >>> to_lowercase("Hello World")
        'hello world'
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from text.

    Args:
        text (str): Input string.

    Returns:
        str: String with punctuation removed.

    Example:
        >>> remove_punctuation("Hello, world!")
        'Hello world'
    """
    # Replace punctuation with a space to avoid merging words
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def tokenize(text: str) -> list[str]:
    """
    Split text into individual word tokens.

    Punctuation has already been removed, so whitespace tokenization is enough.

    Args:
        text (str): Cleaned input string.

    Returns:
        list: List of word tokens.

    Example:
        >>> tokenize("hello world")
        ['hello', 'world']
    """
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Filter out stopwords from a list of tokens.

    Args:
        tokens (list): List of word tokens.

    Returns:
        list: Tokens with stopwords removed.

    Example:
        >>> remove_stopwords(['what', 'is', 'nlp'])
        ['nlp']
    """
    return [token for token in tokens if token not in STOPWORDS]


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: lowercase → remove punctuation →
    tokenize → remove stopwords → rejoin into a string.

    This cleaned string is used for TF-IDF vectorization and
    cosine similarity matching.

    Args:
        text (str): Raw user input.

    Returns:
        str: Cleaned, preprocessed text ready for matching.

    Example:
        >>> preprocess("What is Natural Language Processing?")
        'natural language processing'
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    # Remove any empty strings or single characters that slipped through
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)
