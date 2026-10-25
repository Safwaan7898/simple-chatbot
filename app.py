"""Local Flask web interface for the NLPBot chatbot."""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from chatbot import build_tfidf_index, find_best_match, load_intents

app = Flask(__name__)

INTENTS = load_intents(Path(__file__).with_name("intents.json"))
VECTORIZER, TFIDF_MATRIX, INDEX_TO_TAG, INDEX_TO_PATTERN = build_tfidf_index(INTENTS)


@app.get("/")
def index():
    """Serve the chatbot interface."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Report whether the chatbot matcher is ready."""
    return jsonify({"ready": True})


@app.post("/api/chat")
def chat():
    """Match a submitted message and return the chatbot response."""
    if not request.is_json:
        return jsonify({"error": "Send your message as JSON."}), 415

    payload = request.get_json(silent=True)
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message before sending."}), 400
    if len(message) > 1000:
        return jsonify({"error": "Messages must be 1,000 characters or fewer."}), 400

    response = find_best_match(
        message,
        INTENTS,
        VECTORIZER,
        TFIDF_MATRIX,
        INDEX_TO_TAG,
        INDEX_TO_PATTERN,
    )
    return jsonify({"response": response})


if __name__ == "__main__":
    print("NLPBot web UI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
