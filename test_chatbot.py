"""Tests for chatbot preprocessing, matching, and input validation."""

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app import app as web_app
from chatbot import (
    FALLBACK_RESPONSES,
    build_tfidf_index,
    find_best_match,
    get_response,
    load_intents,
)
from preprocess import preprocess, remove_stopwords, tokenize


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_lowercases_removes_punctuation_and_stopwords(self):
        self.assertEqual(preprocess("What is NLP?"), "nlp")

    def test_tokenize_splits_text_into_words(self):
        self.assertEqual(tokenize("natural language processing"), [
            "natural", "language", "processing"
        ])

    def test_stopword_removal_keeps_meaningful_tokens(self):
        self.assertEqual(remove_stopwords(["what", "is", "python"]), ["python"])


class ChatbotMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intents = load_intents(Path(__file__).with_name("intents.json"))
        (
            cls.vectorizer,
            cls.tfidf_matrix,
            cls.index_to_tag,
            cls.index_to_pattern,
        ) = build_tfidf_index(cls.intents)

    def match(self, text):
        return find_best_match(
            text,
            self.intents,
            self.vectorizer,
            self.tfidf_matrix,
            self.index_to_tag,
            self.index_to_pattern,
        )

    def test_greeting_matches(self):
        self.assertIn(self.match("hello"), self.responses_for("greeting"))

    def test_fuzzy_matching_handles_a_typo(self):
        self.assertIn(self.match("helllo"), self.responses_for("greeting"))

    def test_nonsense_uses_polite_fallback(self):
        self.assertIn(self.match("asdf qwerty nonsense"), FALLBACK_RESPONSES)

    def test_empty_input_is_handled(self):
        self.assertEqual(self.match(""), "Please type something so I can help you!")

    def test_time_response_is_computed(self):
        response = self.match("what time is it")
        self.assertRegex(response, r"The current time is \d{2}:\d{2} [AP]M\.")

    def test_date_response_is_computed(self):
        response = self.match("what is today's date")
        expected_date = datetime.now().strftime("%A, %B %d, %Y")
        self.assertEqual(response, f"Today is {expected_date}.")

    def test_load_intents_rejects_invalid_json(self):
        with TemporaryDirectory() as directory:
            invalid_file = Path(directory) / "intents.json"
            invalid_file.write_text("{invalid json", encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_intents(invalid_file)

    @classmethod
    def responses_for(cls, tag):
        intent = next(item for item in cls.intents if item["tag"] == tag)
        return intent["responses"]


class WebAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        web_app.config.update(TESTING=True)
        cls.client = web_app.test_client()

    def test_homepage_serves_chat_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Big questions.", response.data)

    def test_health_endpoint_reports_ready(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ready": True})

    def test_chat_endpoint_returns_bot_response(self):
        response = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.get_json())

    def test_chat_endpoint_rejects_blank_message(self):
        response = self.client.post("/api/chat", json={"message": "  "})
        self.assertEqual(response.status_code, 400)

    def test_chat_endpoint_rejects_oversized_message(self):
        response = self.client.post("/api/chat", json={"message": "x" * 1001})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
