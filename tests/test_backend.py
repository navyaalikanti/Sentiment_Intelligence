from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from main import create_app
from sentiment_analyzer import (
    analyze_text,
    detect_unusual_reviews,
    extract_named_entities,
    get_agreement_summary,
    normalize_sentiment_label,
    sentiment_distribution_counts,
)


class BackendLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_frame = pd.DataFrame(
            [
                {
                    "Id": 1,
                    "Score": 1,
                    "Summary": "Great but low rating",
                    "Text": "this is the best",
                    "vader_label": "Positive",
                    "vader_label_code": "POS",
                    "vader_compound": 0.864,
                    "roberta_label": "Positive",
                    "roberta_label_code": "POS",
                    "roberta_pos": 0.91,
                    "roberta_neu": 0.05,
                    "roberta_neg": 0.04,
                    "ml_label": "Positive",
                    "ml_label_code": "POS",
                    "ml_pos": 0.9,
                    "ml_neu": 0.05,
                    "ml_neg": 0.05,
                    "ml_confidence": 0.9,
                },
                {
                    "Id": 2,
                    "Score": 5,
                    "Summary": "Bad but high rating",
                    "Text": "this is awful",
                    "vader_label": "Negative",
                    "vader_label_code": "NEG",
                    "vader_compound": -0.91,
                    "roberta_label": "Negative",
                    "roberta_label_code": "NEG",
                    "roberta_pos": 0.03,
                    "roberta_neu": 0.04,
                    "roberta_neg": 0.93,
                    "ml_label": "Negative",
                    "ml_label_code": "NEG",
                    "ml_pos": 0.03,
                    "ml_neu": 0.04,
                    "ml_neg": 0.93,
                    "ml_confidence": 0.93,
                },
                {
                    "Id": 3,
                    "Score": 3,
                    "Summary": "Neutral",
                    "Text": "okay",
                    "vader_label": "Neutral",
                    "vader_label_code": "NEU",
                    "vader_compound": 0.0,
                    "roberta_label": "Neutral",
                    "roberta_label_code": "NEU",
                    "roberta_pos": 0.1,
                    "roberta_neu": 0.8,
                    "roberta_neg": 0.1,
                    "ml_label": "Neutral",
                    "ml_label_code": "NEU",
                    "ml_pos": 0.1,
                    "ml_neu": 0.8,
                    "ml_neg": 0.1,
                    "ml_confidence": 0.8,
                },
            ]
        )

        self.ml_evaluation_report = {
            "dataset_size": 3,
            "training_samples": 2,
            "testing_samples": 1,
            "tfidf_features": 42,
            "label_order": ["Negative", "Neutral", "Positive"],
            "model_metrics": {
                "logistic_regression": {"accuracy": 0.81, "precision": 0.82, "recall": 0.8, "f1_score": 0.81},
                "vader": {"accuracy": 0.72, "precision": 0.73, "recall": 0.71, "f1_score": 0.72},
                "roberta": {"accuracy": 0.78, "precision": 0.79, "recall": 0.77, "f1_score": 0.78},
            },
            "classification_report": {
                "Negative": {"precision": 0.8, "recall": 0.81, "f1_score": 0.8, "support": 1},
                "Neutral": {"precision": 0.79, "recall": 0.77, "f1_score": 0.78, "support": 1},
                "Positive": {"precision": 0.84, "recall": 0.82, "f1_score": 0.83, "support": 1},
                "accuracy": 0.81,
                "macro avg": {"precision": 0.81, "recall": 0.8, "f1_score": 0.8, "support": 3},
                "weighted avg": {"precision": 0.81, "recall": 0.81, "f1_score": 0.81, "support": 3},
            },
            "confusion_matrix": {
                "labels": ["Negative", "Neutral", "Positive"],
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "comparison_chart_data": [
                {"model": "Logistic Regression", "accuracy": 0.81, "precision": 0.82, "recall": 0.8, "f1": 0.81},
                {"model": "VADER", "accuracy": 0.72, "precision": 0.73, "recall": 0.71, "f1": 0.72},
                {"model": "RoBERTa", "accuracy": 0.78, "precision": 0.79, "recall": 0.77, "f1": 0.78},
            ],
            "summary": {"dataset_size": 3, "training_samples": 2, "testing_samples": 1, "tfidf_features": 42},
        }

    def test_normalize_sentiment_label(self) -> None:
        self.assertEqual(normalize_sentiment_label("Positive"), "POS")
        self.assertEqual(normalize_sentiment_label("Pos"), "POS")
        self.assertEqual(normalize_sentiment_label("Negative"), "NEG")
        self.assertEqual(normalize_sentiment_label("Neutral"), "NEU")
        self.assertEqual(normalize_sentiment_label("Unknown"), "UNKNOWN")

    def test_agreement_summary_uses_normalized_labels(self) -> None:
        result = get_agreement_summary(self.sample_frame)
        self.assertEqual(result["agreement_count"], 3)
        self.assertEqual(result["disagreement_count"], 0)
        self.assertEqual(result["agreement_percentage"], 100.0)
        self.assertEqual(result["agree"], 3)
        self.assertEqual(result["disagree"], 0)

    def test_sentiment_distribution_counts_are_normalized(self) -> None:
        result = sentiment_distribution_counts(self.sample_frame)
        vader_counts = {item["code"]: item["count"] for item in result["vader"]}
        roberta_counts = {item["code"]: item["count"] for item in result["roberta"]}
        self.assertEqual(vader_counts["POS"], 1)
        self.assertEqual(vader_counts["NEG"], 1)
        self.assertEqual(vader_counts["NEU"], 1)
        self.assertEqual(roberta_counts["POS"], 1)
        self.assertEqual(roberta_counts["NEG"], 1)
        self.assertEqual(roberta_counts["NEU"], 1)

    def test_unusual_review_detection_uses_model_labels(self) -> None:
        with patch("roberta_model.analyze_roberta") as mocked_roberta:
            mocked_roberta.side_effect = [
                {"roberta_label": "Positive", "roberta_label_code": "POS", "roberta_pos": 0.91, "roberta_neu": 0.05, "roberta_neg": 0.04},
                {"roberta_label": "Negative", "roberta_label_code": "NEG", "roberta_pos": 0.03, "roberta_neu": 0.04, "roberta_neg": 0.93},
                {"roberta_label": "Neutral", "roberta_label_code": "NEU", "roberta_pos": 0.1, "roberta_neu": 0.8, "roberta_neg": 0.1},
            ]

            result = detect_unusual_reviews(self.sample_frame, top_n=10)

        self.assertEqual(result["counts"]["positive_sentiment_low_rating"], 1)
        self.assertEqual(result["counts"]["negative_sentiment_high_rating"], 1)
        self.assertEqual(result["positive_sentiment_low_rating"][0]["vader"]["label"], "Positive")
        self.assertEqual(result["negative_sentiment_high_rating"][0]["vader"]["label"], "Negative")
        self.assertEqual(result["positive_sentiment_low_rating"][0]["vader"]["code"], "POS")
        self.assertEqual(result["negative_sentiment_high_rating"][0]["vader"]["code"], "NEG")

    def test_ner_returns_expected_structure(self) -> None:
        entities = extract_named_entities("Amazon delivered quickly and the packaging was clean.")
        self.assertTrue(all("text" in entity and "label" in entity and "confidence" in entity for entity in entities))
        amazon = next((entity for entity in entities if entity["text"].lower() == "amazon"), None)
        self.assertIsNotNone(amazon)
        self.assertEqual(amazon["label"], "ORG")

    def test_analyze_text_includes_logistic_regression_prediction(self) -> None:
        with patch("backend.ml_model.analyze_logistic_regression") as mocked_logistic, patch(
            "roberta_model.analyze_roberta"
        ) as mocked_roberta:
            mocked_logistic.return_value = {
                "label": "Positive",
                "label_code": "POS",
                "pos": 0.91,
                "neu": 0.05,
                "neg": 0.04,
                "confidence": 0.91,
            }
            mocked_roberta.return_value = {
                "roberta_label": "Positive",
                "roberta_label_code": "POS",
                "roberta_pos": 0.91,
                "roberta_neu": 0.05,
                "roberta_neg": 0.04,
            }

            result = analyze_text("I love this product")

        self.assertIn("logistic_regression", result)
        self.assertIn("ml", result)
        self.assertEqual(result["logistic_regression"]["label"], "Positive")
        self.assertEqual(result["ml"]["label_code"], "POS")

    def test_api_endpoints_still_work(self) -> None:
        app = create_app()
        client = app.test_client()

        health = client.get("/api/health")
        self.assertEqual(health.status_code, 200)

        with patch("main.get_roberta_status", return_value={"available": True, "model": "mock"}), patch(
            "main.load_reviews"
        ) as mocked_load_reviews, patch("main.build_combined_dataset") as mocked_combined, patch(
            "main.detect_unusual_reviews"
        ) as mocked_unusual, patch("main.analyze_text") as mocked_analyze_text, patch(
            "main.load_ml_evaluation_report", return_value=self.ml_evaluation_report
        ), patch("backend.ml_model.analyze_logistic_regression") as mocked_logistic, patch(
            "roberta_model.analyze_roberta"
        ) as mocked_roberta:
            mocked_load_reviews.return_value = self.sample_frame.copy()
            mocked_combined.return_value = self.sample_frame.copy()
            mocked_unusual.return_value = {
                "counts": {
                    "positive_sentiment_low_rating": 1,
                    "negative_sentiment_high_rating": 1,
                    "total_unusual": 2,
                },
                "positive_sentiment_low_rating": [],
                "negative_sentiment_high_rating": [],
            }
            mocked_logistic.return_value = {
                "label": "Positive",
                "label_code": "POS",
                "pos": 0.91,
                "neu": 0.05,
                "neg": 0.04,
                "confidence": 0.91,
            }
            mocked_roberta.return_value = {
                "roberta_label": "Positive",
                "roberta_label_code": "POS",
                "roberta_pos": 0.92,
                "roberta_neu": 0.04,
                "roberta_neg": 0.04,
            }
            mocked_analyze_text.return_value = {
                "text": "I love this product",
                "clean_text": "I love this product",
                "token_count": 4,
                "tokens": ["I", "love", "this", "product"],
                "pos_tags": [
                    {"token": "I", "tag": "PRP"},
                    {"token": "love", "tag": "VBP"},
                    {"token": "this", "tag": "DT"},
                    {"token": "product", "tag": "NN"},
                ],
                "named_entities": [],
                "vader": {"label": "Positive", "label_code": "POS", "compound": 0.9},
                "roberta": {"roberta_label": "Positive", "roberta_label_code": "POS", "roberta_pos": 0.92, "roberta_neu": 0.04, "roberta_neg": 0.04},
                "logistic_regression": {"label": "Positive", "label_code": "POS", "pos": 0.91, "neu": 0.05, "neg": 0.04, "confidence": 0.91},
                "ml": {"label": "Positive", "label_code": "POS", "pos": 0.91, "neu": 0.05, "neg": 0.04, "confidence": 0.91},
            }

            stats = client.get("/api/dataset-statistics?limit=3")
            ratings = client.get("/api/rating-distribution?limit=3")
            sentiment = client.get("/api/sentiment-distribution?limit=3")
            comparison = client.get("/api/model-comparison?limit=3")
            unusual = client.get("/api/unusual-reviews?limit=3")
            preview = client.get("/api/dataset-preview?limit=3")
            analyze = client.post("/api/analyze-review", json={"review": "I love this product"})
            nlp = client.post("/api/nlp-analysis", json={"review": "I love this product"})
            ml_eval = client.get("/api/ml-evaluation")
            analyze_missing = client.post("/api/analyze-review", json={})

        self.assertEqual(stats.status_code, 200)
        self.assertEqual(ratings.status_code, 200)
        self.assertEqual(sentiment.status_code, 200)
        self.assertEqual(comparison.status_code, 200)
        self.assertEqual(unusual.status_code, 200)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(analyze.status_code, 200)
        self.assertEqual(nlp.status_code, 200)
        self.assertEqual(ml_eval.status_code, 200)
        self.assertEqual(analyze_missing.status_code, 400)
        self.assertEqual(analyze_missing.get_json()["error"], "review is required")

    def test_ml_evaluation_endpoint_returns_payload(self) -> None:
        app = create_app()
        client = app.test_client()

        with patch("main.load_ml_evaluation_report", return_value=self.ml_evaluation_report):
            response = client.get("/api/ml-evaluation")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("model_metrics", payload)
        self.assertIn("confusion_matrix", payload)
        self.assertIn("classification_report", payload)


if __name__ == "__main__":
    unittest.main()
