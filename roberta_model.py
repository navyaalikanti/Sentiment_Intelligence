from __future__ import annotations

from functools import lru_cache
from typing import Any

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SENTIMENT_CODE_TO_LABEL = {
    "POS": "Positive",
    "NEG": "Negative",
    "NEU": "Neutral",
}
POSITIVE_HINTS = {
    "best",
    "good",
    "great",
    "amazing",
    "awesome",
    "love",
    "loved",
    "excellent",
    "wonderful",
    "delicious",
    "perfect",
    "clean",
}
NEGATIVE_HINTS = {
    "bad",
    "worst",
    "awful",
    "terrible",
    "hate",
    "hated",
    "poor",
    "broken",
    "wrong",
    "disappointed",
    "dirty",
}


@lru_cache(maxsize=1)
def _load_model_components():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    return tokenizer, model


def _fallback_roberta_scores(text: str) -> tuple[float, float, float, str]:
    tokens = {token.lower() for token in str(text).split()}
    positive_hits = len(tokens & POSITIVE_HINTS)
    negative_hits = len(tokens & NEGATIVE_HINTS)

    if positive_hits > negative_hits:
        return 0.1, 0.15, 0.75, "POS"
    if negative_hits > positive_hits:
        return 0.75, 0.15, 0.1, "NEG"
    return 0.2, 0.6, 0.2, "NEU"


def get_roberta_status() -> dict[str, Any]:
    try:
        _load_model_components()
        return {
            "available": True,
            "model": MODEL_NAME,
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "available": False,
            "model": MODEL_NAME,
            "error": str(exc),
        }


@lru_cache(maxsize=4096)
def _analyze_roberta_cached(text: str) -> tuple[float, float, float, str, str | None]:
    try:
        tokenizer, model = _load_model_components()
        encoded_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output = model(**encoded_text)
        scores = output.logits[0].detach().cpu().numpy()

        from scipy.special import softmax

        probabilities = softmax(scores)
        labels = {
            "NEG": float(probabilities[0]),
            "NEU": float(probabilities[1]),
            "POS": float(probabilities[2]),
        }
        label_code = max(labels, key=labels.get)
        return labels["NEG"], labels["NEU"], labels["POS"], label_code, None
    except Exception as exc:  # pragma: no cover - defensive fallback
        neg, neu, pos, label_code = _fallback_roberta_scores(text)
        return neg, neu, pos, label_code, str(exc)


def analyze_roberta(text: str) -> dict[str, Any]:
    normalized = "" if text is None else str(text)
    neg, neu, pos, label_code, error = _analyze_roberta_cached(normalized)
    result = {
        "roberta_neg": float(neg),
        "roberta_neu": float(neu),
        "roberta_pos": float(pos),
        "roberta_label_code": label_code,
        "roberta_label": SENTIMENT_CODE_TO_LABEL.get(label_code, "Unknown"),
    }
    if error:
        result["error"] = error
    return result


def pipeline_sentiment(text: str) -> dict[str, Any]:
    try:
        from transformers import pipeline

        tokenizer, model = _load_model_components()
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        prediction = classifier(text[:512])[0]
        return {
            "label": prediction["label"],
            "score": float(prediction["score"]),
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        neg, neu, pos, label_code = _fallback_roberta_scores(text)
        return {
            "label": SENTIMENT_CODE_TO_LABEL.get(label_code, "Unknown"),
            "score": float(max(neg, neu, pos)),
            "error": str(exc),
        }
