"""
chatbot.py
----------
Main entry point for the NLP Chatbot.

How it works:
  1. Load intents from intents.json
  2. Preprocess all stored patterns using preprocess.py
  3. Build a TF-IDF matrix from all patterns
  4. In a loop, accept user input, preprocess it, compute cosine
     similarity against the TF-IDF matrix, pick the best matching
     intent, and return a random response from that intent.
  5. Handle dynamic responses (time/date), fallback replies,
     and graceful exit.

Run:
    python chatbot.py
"""

import json
import random
import sys
import time
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path

OUTPUT_ENCODING = getattr(sys.stdout, "encoding", None)

# Third-party imports (installed via requirements.txt)
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

# ---------------------------------------------------------------------------
# Colour helpers — gracefully degrade if colorama is missing
# ---------------------------------------------------------------------------

def _c(text: str, colour: str) -> str:
    """Wrap text in a colorama colour code if available."""
    if OUTPUT_ENCODING:
        text = text.replace("─", "-").replace("—", "-").replace("–", "-").encode(
            OUTPUT_ENCODING, errors="ignore"
        ).decode(OUTPUT_ENCODING)
    if not COLORAMA_AVAILABLE:
        return text
    colours = {
        "cyan":   Fore.CYAN,
        "green":  Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red":    Fore.RED,
        "blue":   Fore.BLUE,
        "white":  Fore.WHITE,
        "bold":   Style.BRIGHT,
    }
    return colours.get(colour, "") + text + Style.RESET_ALL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENTS_FILE = Path(__file__).resolve().with_name("intents.json")
SIMILARITY_THRESHOLD = 0.15   # Minimum cosine similarity to accept a match
FUZZY_THRESHOLD = 0.78        # Minimum difflib ratio for typo-tolerant matches
EXIT_COMMANDS  = {"exit", "quit", "bye", "goodbye", "farewell"}
FALLBACK_RESPONSES = [
    "Sorry, I didn't understand that. Can you rephrase?",
    "Hmm, I'm not sure about that. Could you ask differently?",
    "I didn't quite catch that. Try asking in a different way!",
    "That's outside my knowledge for now. Can you rephrase?",
]


# ---------------------------------------------------------------------------
# Load intents
# ---------------------------------------------------------------------------

def load_intents(filepath: str | Path) -> list[dict]:
    """
    Load and parse the intents JSON file.

    Args:
        filepath (str): Path to intents.json.

    Returns:
        list: List of intent dictionaries.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(_c(f"[ERROR] '{filepath}' not found. Make sure it's in the same folder.", "red"))
        sys.exit(1)
    except (OSError, json.JSONDecodeError) as e:
        print(_c(f"[ERROR] Failed to parse '{filepath}': {e}", "red"))
        sys.exit(1)

    intents = data.get("intents") if isinstance(data, dict) else None
    if not isinstance(intents, list) or not intents:
        print(_c(f"[ERROR] '{filepath}' must contain a non-empty 'intents' list.", "red"))
        sys.exit(1)

    for index, intent in enumerate(intents, start=1):
        valid_intent = (
            isinstance(intent, dict)
            and isinstance(intent.get("tag"), str)
            and bool(intent["tag"].strip())
            and isinstance(intent.get("patterns"), list)
            and bool(intent["patterns"])
            and all(isinstance(item, str) and item.strip() for item in intent["patterns"])
            and isinstance(intent.get("responses"), list)
            and bool(intent["responses"])
            and all(isinstance(item, str) and item.strip() for item in intent["responses"])
        )
        if not valid_intent:
            print(_c(f"[ERROR] Intent {index} in '{filepath}' has an invalid shape.", "red"))
            sys.exit(1)

    return intents


# ---------------------------------------------------------------------------
# Build TF-IDF index
# ---------------------------------------------------------------------------

def build_tfidf_index(
    intents: list[dict],
) -> tuple[TfidfVectorizer, csr_matrix, list[str], list[str]]:
    """
    Flatten all patterns from all intents, preprocess them, and fit
    a TF-IDF vectorizer. Returns the vectorizer, the TF-IDF matrix,
    the matching intent tags, and the cleaned patterns.

    Args:
        intents (list): List of intent dicts from intents.json.

    Returns:
        tuple: Vectorizer, TF-IDF matrix, matching tags, and cleaned patterns.
    """
    all_patterns = []   # cleaned pattern strings
    index_to_tag = []   # parallel list: which tag does each pattern belong to?

    for intent in intents:
        for pattern in intent["patterns"]:
            cleaned = preprocess(pattern)
            if not cleaned:
                continue
            all_patterns.append(cleaned)
            index_to_tag.append(intent["tag"])

    if not all_patterns:
        raise ValueError("No usable intent patterns remain after preprocessing.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_patterns)

    return vectorizer, tfidf_matrix, index_to_tag, all_patterns


# ---------------------------------------------------------------------------
# Get response for a matched tag
# ---------------------------------------------------------------------------

def get_response(tag: str, intents: list[dict]) -> str:
    """
    Find the intent with the given tag and return a random response.
    Handles dynamic DYNAMIC_TIME and DYNAMIC_DATE placeholders.

    Args:
        tag (str): The matched intent tag.
        intents (list): Full list of intents.

    Returns:
        str: A response string.
    """
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])

            # Replace dynamic placeholders
            if response == "DYNAMIC_TIME":
                now = datetime.now()
                return f"The current time is {now.strftime('%I:%M %p')}."

            if response == "DYNAMIC_DATE":
                now = datetime.now()
                return f"Today is {now.strftime('%A, %B %d, %Y')}."

            return response

    return random.choice(FALLBACK_RESPONSES)


# ---------------------------------------------------------------------------
# Core matching function
# ---------------------------------------------------------------------------

def find_best_match(user_input: str, intents: list,
                    vectorizer: TfidfVectorizer, tfidf_matrix: csr_matrix,
                    index_to_tag: list[str], index_to_pattern: list[str]) -> str:
    """
    Find the best matching intent for the user's input.

    Strategy:
      1. Preprocess the user input.
      2. Use TF-IDF + cosine similarity to rank the known patterns.
      3. If similarity score >= SIMILARITY_THRESHOLD, return that tag.
      4. If needed, use fuzzy matching for small spelling mistakes.
      5. If neither score reaches its threshold, return a fallback response.

    Args:
        user_input (str): Raw user input string.
        intents (list): List of intent dicts.
        vectorizer: Fitted TfidfVectorizer (or None).
        tfidf_matrix: Fitted TF-IDF matrix (or None).
        index_to_tag (list): Maps matrix row → intent tag.
        index_to_pattern (list): Cleaned pattern text for fuzzy matching.

    Returns:
        str: The chatbot's response.
    """
    cleaned_input = preprocess(user_input)

    if not cleaned_input.strip():
        return "Please type something so I can help you!"

    user_vec = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix).ravel()
    best_idx = int(similarities.argmax())

    if similarities[best_idx] >= SIMILARITY_THRESHOLD:
        return get_response(index_to_tag[best_idx], intents)

    fuzzy_scores = [
        SequenceMatcher(None, cleaned_input, pattern).ratio()
        for pattern in index_to_pattern
    ]
    fuzzy_idx = max(range(len(fuzzy_scores)), key=fuzzy_scores.__getitem__)
    if fuzzy_scores[fuzzy_idx] >= FUZZY_THRESHOLD:
        return get_response(index_to_tag[fuzzy_idx], intents)

    return random.choice(FALLBACK_RESPONSES)


# ---------------------------------------------------------------------------
# Terminal UX helpers
# ---------------------------------------------------------------------------

def print_banner():
    """Print a welcome banner to the terminal."""
    banner = r"""
  _   _ _     ____  ____        _
 | \ | | |   |  _ \| __ )  ___ | |_
 |  \| | |   | |_) |  _ \ / _ \| __|
 | |\  | |___|  __/| |_) | (_) | |_
 |_| \_|_____|_|   |____/ \___/ \__|

    """
    print(_c(banner, "cyan"))
    print(_c("  🤖  NLPBot — AI Chatbot for NLP Internship", "bold"))
    print(_c("  Built with Python, NLTK, scikit-learn & colorama", "yellow"))
    print(_c("  " + "─" * 50, "cyan"))
    print(_c("  Type your message and press Enter to chat.", "white"))
    print(_c("  Type 'exit', 'quit', or 'bye' to stop.\n", "white"))


def thinking_effect():
    """Display a short 'thinking' animation in the terminal."""
    sys.stdout.write(_c("  Bot is thinking", "yellow"))
    for _ in range(3):
        time.sleep(0.3)
        sys.stdout.write(_c(".", "yellow"))
        sys.stdout.flush()
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the line
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat():
    """
    Main chat loop.

    Loads intents, builds the TF-IDF index, then enters an interactive
    loop reading user input and printing bot responses until the user
    types an exit command or presses Ctrl+C.
    """
    print_banner()

    # Load knowledge base
    intents = load_intents(INTENTS_FILE)
    print(_c(f"  ✅  Loaded {len(intents)} intents from {INTENTS_FILE}", "green"))

    # Build TF-IDF index
    vectorizer, tfidf_matrix, index_to_tag, index_to_pattern = build_tfidf_index(intents)
    print(_c("  ✅  TF-IDF index built successfully", "green"))

    print(_c("  " + "─" * 50 + "\n", "cyan"))

    # Greet the user
    print(_c("  Bot: ", "green") + "Hello! I'm NLPBot. How can I help you today?")
    print()

    while True:
        try:
            # Read user input
            user_input = input(_c("  You: ", "blue")).strip()

            # Handle empty input
            if not user_input:
                print(_c("  Bot: ", "green") + "Please type something!\n")
                continue

            # Check for exit commands
            if user_input.lower() in EXIT_COMMANDS:
                print(
                    _c("  Bot: ", "green")
                    + _c("Goodbye! It was great chatting with you. 👋", "white")
                )
                print(_c("\n  Thanks for using NLPBot! #incodevision\n", "cyan"))
                break

            # Show thinking effect
            thinking_effect()

            # Get and print response
            response = find_best_match(
                user_input, intents, vectorizer, tfidf_matrix,
                index_to_tag, index_to_pattern
            )
            print(_c("  Bot: ", "green") + _c(response, "white"))
            print()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print(
                _c("\n\n  Bot: ", "green")
                + _c("Caught a Ctrl+C! Goodbye! 👋", "white")
            )
            print(_c("  Thanks for using NLPBot! #incodevision\n", "cyan"))
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chat()
