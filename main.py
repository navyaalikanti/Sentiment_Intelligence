from __future__ import annotations

from functools import lru_cache

from flask import Flask, jsonify, request
from flask_cors import CORS

from roberta_model import get_roberta_status
from sentiment_analyzer import (
    add_ml_columns,
    agreement_and_distribution,
    analyze_text,
    build_combined_dataset,
    build_sentiment_word_clouds,
    dataset_statistics,
    detect_unusual_reviews,
    load_reviews,
    sanitize_limit,
)
from backend.ml_model import load_ml_evaluation_report
from visualization import (
    create_pairplot_image,
    model_comparison_chart_data,
    rating_distribution_data,
)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @lru_cache(maxsize=32)
    def _cached_sentiment_distribution(limit: int) -> dict:
        combined = build_combined_dataset(limit)
        payload = agreement_and_distribution(combined)
        return {
            "limit": limit,
            "data": payload["sentiments"],
            "agreement_summary": payload["agreement"],
        }

    @lru_cache(maxsize=16)
    def _cached_model_comparison(limit: int) -> dict:
        combined = build_combined_dataset(limit)
        payload = model_comparison_chart_data(combined)
        payload["pairplot_base64"] = create_pairplot_image(add_ml_columns(combined))
        payload["agreement"] = payload.get("agreement") or {}
        return payload

    @lru_cache(maxsize=32)
    def _cached_unusual_reviews(limit: int, top_n: int) -> dict:
        frame = load_reviews(limit)
        return detect_unusual_reviews(frame, top_n=top_n)

    @lru_cache(maxsize=16)
    def _cached_word_clouds(limit: int) -> dict:
        frame = load_reviews(limit)
        return build_sentiment_word_clouds(frame)

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "roberta": get_roberta_status(),
            }
        )

    @app.post("/api/analyze-review")
    def analyze_review():
        try:
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "JSON body is required"}), 400

            review_text = (
                payload.get("review")
                or payload.get("text")
                or payload.get("content")
                or ""
            ).strip()

            if not review_text:
                return jsonify({"error": "review is required"}), 400

            return jsonify(analyze_text(review_text))
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Analyze review failed")
            return jsonify(
                {
                    "error": "analysis failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/dataset-statistics")
    def api_dataset_statistics():
        limit = sanitize_limit(request.args.get("limit"), default=5000)
        frame = load_reviews(limit)
        return jsonify(dataset_statistics(frame))

    @app.get("/api/rating-distribution")
    def api_rating_distribution():
        limit = sanitize_limit(request.args.get("limit"), default=5000)
        frame = load_reviews(limit)
        return jsonify(
            {
                "limit": limit,
                "data": rating_distribution_data(frame),
            }
        )

    @app.get("/api/sentiment-distribution")
    def api_sentiment_distribution():
        try:
            limit = sanitize_limit(request.args.get("limit"), default=500)
            return jsonify(_cached_sentiment_distribution(limit).copy())
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Sentiment distribution failed")
            return jsonify(
                {
                    "error": "sentiment distribution failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/model-comparison")
    def api_model_comparison():
        try:
            limit = sanitize_limit(request.args.get("limit"), default=500)
            payload = _cached_model_comparison(limit).copy()
            payload["evaluation"] = load_ml_evaluation_report()
            return jsonify(payload)
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Model comparison failed")
            return jsonify(
                {
                    "error": "model comparison failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/ml-evaluation")
    def api_ml_evaluation():
        try:
            return jsonify(load_ml_evaluation_report())
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("ML evaluation failed")
            return jsonify(
                {
                    "error": "ml evaluation failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/unusual-reviews")
    def api_unusual_reviews():
        try:
            limit = sanitize_limit(request.args.get("limit"), default=500)
            top_n = sanitize_limit(request.args.get("top_n"), default=10)
            return jsonify(_cached_unusual_reviews(limit, top_n).copy())
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Unusual reviews failed")
            return jsonify(
                {
                    "error": "unusual reviews failed",
                    "details": str(exc),
                }
            ), 500

    @app.post("/api/nlp-analysis")
    def api_nlp_analysis():
        try:
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "JSON body is required"}), 400

            text = (
                payload.get("review")
                or payload.get("text")
                or payload.get("content")
                or ""
            ).strip()

            if not text:
                return jsonify({"error": "review is required"}), 400

            return jsonify(analyze_text(text))
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("NLP analysis failed")
            return jsonify(
                {
                    "error": "nlp analysis failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/dataset-preview")
    def api_dataset_preview():
        try:
            limit = sanitize_limit(request.args.get("limit"), default=25)
            with_sentiment = str(request.args.get("with_sentiment", "")).lower() in {"1", "true", "yes"}
            frame = load_reviews(limit) if not with_sentiment else build_combined_dataset(limit)
            rows = frame.replace({float("nan"): None}).to_dict(orient="records")
            if with_sentiment:
                for row in rows:
                    row["sentiment"] = row.get("vader_label")
            return jsonify(
                {
                    "limit": limit,
                    "rows": rows,
                    "with_sentiment": with_sentiment,
                }
            )
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Dataset preview failed")
            return jsonify(
                {
                    "error": "dataset preview failed",
                    "details": str(exc),
                }
            ), 500

    @app.get("/api/word-clouds")
    def api_word_clouds():
        try:
            limit = sanitize_limit(request.args.get("limit"), default=5000)
            return jsonify(
                {
                    "limit": limit,
                    "data": _cached_word_clouds(limit).copy(),
                }
            )
        except Exception as exc:  # pragma: no cover - route hardening
            app.logger.exception("Word clouds failed")
            return jsonify(
                {
                    "error": "word clouds failed",
                    "details": str(exc),
                }
            ), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
