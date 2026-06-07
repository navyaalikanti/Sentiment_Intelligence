from __future__ import annotations

import html
import re
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any
from threading import RLock

import nltk
import pandas as pd
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tree import Tree


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Reviews.csv"
DEFAULT_SAMPLE_SIZE = 5000
SENTIMENT_NORMALIZATION_MAP = {
    "positive": "POS",
    "pos": "POS",
    "p": "POS",
    "negative": "NEG",
    "neg": "NEG",
    "n": "NEG",
    "neutral": "NEU",
    "neu": "NEU",
    "u": "NEU",
    "unknown": "UNKNOWN",
    "unavailable": "UNKNOWN",
}
KNOWN_ORG_ENTITIES = {
    "amazon",
    "amazon.com",
    "whole foods",
    "whole foods market",
    "amazon fresh",
}
DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "was",
    "are",
    "but",
    "not",
    "you",
    "your",
    "have",
    "has",
    "had",
    "from",
    "they",
    "them",
    "their",
    "what",
    "when",
    "where",
    "which",
    "will",
    "would",
    "about",
    "there",
    "here",
    "very",
    "too",
    "all",
    "out",
    "just",
    "really",
    "like",
    "into",
    "than",
    "then",
    "been",
    "were",
    "its",
    "it's",
    "because",
    "over",
    "under",
    "while",
    "after",
    "before",
    "again",
    "each",
    "some",
    "more",
    "most",
    "such",
    "own",
    "also",
    "can",
    "could",
    "should",
    "may",
    "might",
    "do",
    "does",
    "did",
    "done",
    "i",
    "me",
    "my",
    "we",
    "our",
    "us",
    "he",
    "she",
    "it",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "is",
    "be",
    "as",
    "at",
    "by",
    "or",
    "if",
    "so",
}
ANALYSIS_COLUMNS = {
    "vader_neg",
    "vader_neu",
    "vader_pos",
    "vader_compound",
    "vader_label",
    "vader_label_code",
    "roberta_neg",
    "roberta_neu",
    "roberta_pos",
    "roberta_label",
    "roberta_label_code",
    "ml_neg",
    "ml_neu",
    "ml_pos",
    "ml_confidence",
    "ml_label",
    "ml_label_code",
}
_COMBINED_CACHE_LOCK = RLock()
_COMBINED_DATASET_CACHE: dict[int | None, pd.DataFrame] = {}


def sanitize_limit(raw_value: Any, default: int = DEFAULT_SAMPLE_SIZE) -> int:
    try:
        if raw_value is None or raw_value == "":
            return int(default)
        value = int(raw_value)
        return max(1, value)
    except (TypeError, ValueError):
        return int(default)


def normalize_sentiment_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    normalized = str(value).strip().lower()
    if not normalized:
        return "UNKNOWN"
    if normalized in {"pos", "neg", "neu"}:
        return normalized.upper()
    return SENTIMENT_NORMALIZATION_MAP.get(normalized, "UNKNOWN")


def sentiment_code_to_label(code: Any) -> str:
    normalized = normalize_sentiment_label(code)
    if normalized == "POS":
        return "Positive"
    if normalized == "NEG":
        return "Negative"
    if normalized == "NEU":
        return "Neutral"
    return "Unknown"


def _ensure_nltk_data() -> None:
    resources = {
        "vader_lexicon": "sentiment/vader_lexicon.zip",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
        "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
        "words": "corpora/words",
    }

    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            continue


class _FallbackSentimentIntensityAnalyzer:
    POSITIVE_WORDS = {
        "best",
        "good",
        "great",
        "amazing",
        "awesome",
        "love",
        "loved",
        "excellent",
        "clean",
        "quickly",
        "quick",
        "happy",
        "wonderful",
        "delicious",
        "perfect",
    }
    NEGATIVE_WORDS = {
        "bad",
        "worst",
        "awful",
        "terrible",
        "hate",
        "hated",
        "dirty",
        "slow",
        "poor",
        "broken",
        "wrong",
        "disappointed",
    }

    def polarity_scores(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\b[\w']+\b", normalize_text(text).lower())
        pos = sum(token in self.POSITIVE_WORDS for token in tokens)
        neg = sum(token in self.NEGATIVE_WORDS for token in tokens)
        total = max(len(tokens), 1)
        compound = (pos - neg) / total
        if compound > 1:
            compound = 1.0
        elif compound < -1:
            compound = -1.0
        if pos == 0 and neg == 0:
            neu = 1.0
        else:
            neu = max(0.0, 1.0 - (pos + neg) / total)
        return {
            "neg": round(min(1.0, neg / total), 4),
            "neu": round(min(1.0, neu), 4),
            "pos": round(min(1.0, pos / total), 4),
            "compound": round(float(compound), 4),
        }


def _build_sia() -> Any:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        return _FallbackSentimentIntensityAnalyzer()


_ensure_nltk_data()
SIA = _build_sia()


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


@lru_cache(maxsize=8)
def _load_reviews_cached(limit: int | None = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    read_kwargs: dict[str, Any] = {"low_memory": False, "encoding_errors": "ignore"}
    if limit is not None:
        read_kwargs["nrows"] = int(limit)

    frame = pd.read_csv(DATASET_PATH, **read_kwargs)
    frame = frame.copy()

    for column in ("Summary", "Text"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").map(normalize_text)

    if "Score" in frame.columns:
        frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
        frame = frame.dropna(subset=["Score"]).copy()
        frame["Score"] = frame["Score"].astype(int)

    text_series = frame["Text"] if "Text" in frame.columns else pd.Series([""] * len(frame), index=frame.index)
    frame["review_length"] = text_series.fillna("").map(lambda text: len(str(text).split()))
    frame["clean_text"] = text_series.fillna("").map(normalize_text)
    return frame


def load_reviews(limit: int | None = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    return _load_reviews_cached(limit).copy()


@lru_cache(maxsize=32768)
def vader_scores(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    scores = SIA.polarity_scores(normalized)
    code = compound_to_label_code(scores["compound"])
    scores["label"] = sentiment_code_to_label(code)
    scores["label_code"] = code
    return scores


def compound_to_label_code(compound: float) -> str:
    if compound >= 0.05:
        return "POS"
    if compound <= -0.05:
        return "NEG"
    return "NEU"


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    try:
        return word_tokenize(normalized)
    except LookupError:
        return re.findall(r"\b[\w']+\b", normalized)


def pos_tag_text(text: str) -> list[dict[str, str]]:
    tokens = tokenize_text(text)
    if not tokens:
        return []
    try:
        tagged = pos_tag(tokens)
    except LookupError:
        tagged = []
        for token in tokens:
            if token.isdigit():
                tag = "CD"
            elif token.endswith("ing"):
                tag = "VBG"
            elif token.endswith("ed"):
                tag = "VBD"
            elif token[:1].isupper():
                tag = "NNP"
            else:
                tag = "NN"
            tagged.append((token, tag))
    return [{"token": token, "tag": tag} for token, tag in tagged]


def extract_named_entities(text: str) -> list[dict[str, str]]:
    tokens = tokenize_text(text)
    if not tokens:
        return []

    entities: list[dict[str, str]] = []
    normalized_text = text.lower()

    try:
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged)
    except LookupError:
        tree = []

    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_text = " ".join(token for token, _ in subtree.leaves())
            subtree_label = str(subtree.label()).upper()
            entity_key = entity_text.strip().lower()
            confidence = 0.9
            normalized_label = "UNKNOWN"

            if entity_key in KNOWN_ORG_ENTITIES or entity_key.startswith("amazon"):
                normalized_label = "ORG"
                confidence = 0.99
            elif subtree_label in {"ORGANIZATION", "ORG"}:
                normalized_label = "ORG"
            elif subtree_label in {"PERSON", "PER"}:
                normalized_label = "PERSON"
                confidence = 0.65 if len(entity_text.split()) > 1 else 0.45
            elif subtree_label in {"GPE", "LOCATION", "LOC", "FACILITY"}:
                normalized_label = "LOC"
            else:
                normalized_label = "UNKNOWN"
                confidence = 0.4

            if confidence < 0.6:
                normalized_label = "UNKNOWN"

            entities.append(
                {
                    "text": entity_text,
                    "label": normalized_label,
                    "raw_label": subtree.label(),
                    "confidence": round(float(confidence), 3),
                }
            )

    if not entities:
        for known_org in KNOWN_ORG_ENTITIES:
            if known_org in normalized_text:
                entities.append(
                    {
                        "text": "Amazon" if known_org == "amazon" else known_org.title(),
                        "label": "ORG",
                        "raw_label": "KNOWN_ORG",
                        "confidence": 0.99,
                    }
                )
                break

    found_entities = {entity["text"].strip().lower() for entity in entities}
    for known_org in KNOWN_ORG_ENTITIES:
        if known_org in text.lower() and known_org not in found_entities:
            entities.append(
                {
                    "text": "Amazon" if known_org == "amazon" else known_org.title(),
                    "label": "ORG",
                    "raw_label": "KNOWN_ORG",
                    "confidence": 0.99,
                }
            )

    return entities


def analyze_text(text: str) -> dict[str, Any]:
    from roberta_model import analyze_roberta
    from backend.ml_model import analyze_logistic_regression

    normalized = normalize_text(text)
    vader = vader_scores(normalized)
    roberta = analyze_roberta(normalized)
    logistic_regression = analyze_logistic_regression(normalized)
    explanations = explain_sentiment(normalized, vader, roberta)
    tokens = tokenize_text(normalized)
    pos_tags = pos_tag_text(normalized)
    entities = extract_named_entities(normalized)

    return {
        "text": text,
        "clean_text": normalized,
        "token_count": len(tokens),
        "tokens": tokens,
        "pos_tags": pos_tags,
        "named_entities": entities,
        "vader": vader,
        "roberta": roberta,
        "logistic_regression": logistic_regression,
        "ml": logistic_regression,
        "explanations": explanations,
    }


def _token_candidate_words(text: str) -> list[str]:
    tokens = re.findall(r"\b[\w']+\b", normalize_text(text).lower())
    return [
        token
        for token in tokens
        if len(token) > 2 and token not in DEFAULT_STOPWORDS and not token.isdigit()
    ]


def _score_lexicon_words(text: str) -> tuple[list[str], list[str]]:
    scored_positive: list[tuple[str, float]] = []
    scored_negative: list[tuple[str, float]] = []
    for token in _token_candidate_words(text):
        score = float(getattr(SIA, "lexicon", {}).get(token, 0.0))
        if score > 0:
            scored_positive.append((token, score))
        elif score < 0:
            scored_negative.append((token, abs(score)))

    scored_positive.sort(key=lambda item: (-item[1], item[0]))
    scored_negative.sort(key=lambda item: (-item[1], item[0]))

    positive_words = [token for token, _ in scored_positive[:3]]
    negative_words = [token for token, _ in scored_negative[:3]]
    return positive_words, negative_words


def _hint_words(text: str, hints: set[str]) -> list[str]:
    tokens = _token_candidate_words(text)
    return [token for token in tokens if token in hints][:3]


def explain_sentiment(text: str, vader: dict[str, Any], roberta: dict[str, Any]) -> dict[str, Any]:
    vader_positive, vader_negative = _score_lexicon_words(text)
    if not vader_positive:
        vader_positive = _hint_words(text, {"good", "great", "amazing", "excellent", "love", "loved", "best", "wonderful", "perfect", "clean"})
    if not vader_negative:
        vader_negative = _hint_words(text, {"bad", "worst", "awful", "terrible", "hate", "hated", "poor", "broken", "wrong", "disappointed", "dirty"})

    roberta_positive, roberta_negative = _score_lexicon_words(text)
    if roberta.get("roberta_label_code") == "POS":
        roberta_positive = roberta_positive or _hint_words(text, {"good", "great", "amazing", "excellent", "love", "loved", "best", "wonderful", "perfect", "clean"})
    elif roberta.get("roberta_label_code") == "NEG":
        roberta_negative = roberta_negative or _hint_words(text, {"bad", "worst", "awful", "terrible", "hate", "hated", "poor", "broken", "wrong", "disappointed", "dirty"})

    if not vader_positive and not vader_negative and not roberta_positive and not roberta_negative:
        return {
            "vader": {
                "positive_indicators": [],
                "negative_indicators": [],
                "message": "No strong sentiment-bearing words detected.",
            },
            "roberta": {
                "positive_indicators": [],
                "negative_indicators": [],
                "message": "No strong sentiment-bearing words detected.",
            },
            "summary": "No strong sentiment-bearing words detected.",
        }

    return {
        "vader": {
            "positive_indicators": vader_positive,
            "negative_indicators": vader_negative,
            "message": "VADER uses lexicon-driven sentiment cues from the review text.",
        },
        "roberta": {
            "positive_indicators": roberta_positive,
            "negative_indicators": roberta_negative,
            "message": "RoBERTa indicators are approximated from strong sentiment-bearing tokens.",
        },
        "summary": "Highlighted tokens reflect the strongest sentiment-bearing words detected in the review.",
    }


def normalize_word_token(token: str) -> str | None:
    normalized = re.sub(r"[^a-zA-Z']", "", str(token).lower())
    if len(normalized) <= 2 or normalized in DEFAULT_STOPWORDS:
        return None
    return normalized


def build_sentiment_word_clouds(frame: pd.DataFrame, top_n: int = 40) -> dict[str, list[dict[str, Any]]]:
    if "Score" not in frame.columns or "Text" not in frame.columns:
        return {"positive": [], "negative": []}

    positive_counter: Counter[str] = Counter()
    negative_counter: Counter[str] = Counter()

    for _, row in frame.iterrows():
        score = row.get("Score")
        text = normalize_text(row.get("Summary", "")) + " " + normalize_text(row.get("Text", ""))
        words = [normalize_word_token(token) for token in re.findall(r"\b[\w']+\b", text)]
        words = [word for word in words if word]

        if score >= 4:
            positive_counter.update(words)
        elif score <= 2:
            negative_counter.update(words)

    def _serialize(counter: Counter[str]) -> list[dict[str, Any]]:
        items = counter.most_common(top_n)
        return [{"word": word, "count": int(count)} for word, count in items]

    return {
        "positive": _serialize(positive_counter),
        "negative": _serialize(negative_counter),
    }


def add_vader_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=[column for column in ANALYSIS_COLUMNS if column in frame.columns], errors="ignore").copy()
    vader_rows = output["Text"].fillna("").map(vader_scores)
    vader_df = pd.DataFrame(vader_rows.tolist())
    vader_df = vader_df.rename(
        columns={
            "neg": "vader_neg",
            "neu": "vader_neu",
            "pos": "vader_pos",
            "compound": "vader_compound",
            "label": "vader_label",
            "label_code": "vader_label_code",
        }
    )
    output = pd.concat([output.reset_index(drop=True), vader_df.reset_index(drop=True)], axis=1)
    return output


def add_ml_columns(frame: pd.DataFrame) -> pd.DataFrame:
    from backend.ml_model import analyze_logistic_regression

    output = frame.drop(
        columns=[column for column in ("ml_neg", "ml_neu", "ml_pos", "ml_confidence", "ml_label", "ml_label_code") if column in frame.columns],
        errors="ignore",
    ).copy()
    ml_rows = output["Text"].fillna("").map(analyze_logistic_regression)
    ml_df = pd.DataFrame(ml_rows.tolist())
    ml_df = ml_df.rename(
        columns={
            "label": "ml_label",
            "label_code": "ml_label_code",
            "pos": "ml_pos",
            "neu": "ml_neu",
            "neg": "ml_neg",
            "confidence": "ml_confidence",
        }
    )
    output = pd.concat([output.reset_index(drop=True), ml_df.reset_index(drop=True)], axis=1)
    return output


@lru_cache(maxsize=16)
def _build_combined_dataset_cached(limit: int | None = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    from roberta_model import analyze_roberta

    frame = load_reviews(limit)
    enriched = add_vader_columns(frame)

    roberta_records = enriched["Text"].fillna("").map(analyze_roberta)
    roberta_df = pd.DataFrame(roberta_records.tolist())
    combined = pd.concat([enriched.reset_index(drop=True), roberta_df.reset_index(drop=True)], axis=1)
    return combined


def build_combined_dataset(limit: int | None = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    cache_key = None if limit is None else int(limit)
    with _COMBINED_CACHE_LOCK:
        cached = _COMBINED_DATASET_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

    combined = _build_combined_dataset_cached(limit)

    with _COMBINED_CACHE_LOCK:
        _COMBINED_DATASET_CACHE[cache_key] = combined.copy()

    return combined.copy()


def normalize_combined_sentiments(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "vader_label" in output.columns:
        output["vader_label_code"] = output["vader_label"].map(normalize_sentiment_label)
        output["vader_label"] = output["vader_label_code"].map(sentiment_code_to_label)
    if "roberta_label_code" in output.columns:
        output["roberta_label_code"] = output["roberta_label_code"].map(normalize_sentiment_label)
    elif "roberta_label" in output.columns:
        output["roberta_label_code"] = output["roberta_label"].map(normalize_sentiment_label)
    if "roberta_label" in output.columns:
        output["roberta_label"] = output["roberta_label_code"].map(sentiment_code_to_label)
    if "ml_label_code" in output.columns:
        output["ml_label_code"] = output["ml_label_code"].map(normalize_sentiment_label)
    elif "ml_label" in output.columns:
        output["ml_label_code"] = output["ml_label"].map(normalize_sentiment_label)
    if "ml_label" in output.columns:
        output["ml_label"] = output["ml_label_code"].map(sentiment_code_to_label)
    return output


def pairwise_agreement_summary(frame: pd.DataFrame, left_prefix: str, right_prefix: str) -> dict[str, Any]:
    normalized = normalize_combined_sentiments(frame)
    left_column = f"{left_prefix}_label_code"
    right_column = f"{right_prefix}_label_code"
    if left_column not in normalized.columns or right_column not in normalized.columns:
        return {
            "agreement_count": 0,
            "disagreement_count": 0,
            "agreement_percentage": 0.0,
            "agree": 0,
            "disagree": 0,
        }

    comparable = normalized[
        normalized[left_column].isin({"POS", "NEG", "NEU"})
        & normalized[right_column].isin({"POS", "NEG", "NEU"})
    ]
    if comparable.empty:
        return {
            "agreement_count": 0,
            "disagreement_count": 0,
            "agreement_percentage": 0.0,
            "agree": 0,
            "disagree": 0,
        }

    agreement_count = int((comparable[left_column] == comparable[right_column]).sum())
    disagreement_count = int(len(comparable) - agreement_count)
    agreement_percentage = round((agreement_count / len(comparable)) * 100, 2)
    return {
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_percentage": agreement_percentage,
        "agree": agreement_count,
        "disagree": disagreement_count,
    }


def get_agreement_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return pairwise_agreement_summary(frame, "vader", "roberta")


def sentiment_distribution_counts(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    normalized = normalize_combined_sentiments(frame)
    output: dict[str, list[dict[str, Any]]] = {}
    for prefix in ("vader", "roberta", "ml"):
        code_column = f"{prefix}_label_code"
        if code_column not in normalized.columns:
            output[prefix] = []
            continue
        counts = normalized[code_column].value_counts()
        output[prefix] = [
            {
                "label": sentiment_code_to_label(code),
                "code": code,
                "count": int(count),
            }
            for code, count in counts.items()
            if code in {"POS", "NEG", "NEU"}
        ]
    return output


def agreement_and_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_combined_sentiments(frame)
    return {
        "sentiments": sentiment_distribution_counts(normalized),
        "agreement": get_agreement_summary(normalized),
    }


def dataset_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "score_distribution": (
            frame["Score"].value_counts().sort_index().rename_axis("score").reset_index(name="count").to_dict(orient="records")
            if "Score" in frame.columns
            else []
        ),
        "missing_values": frame.isna().sum().to_dict(),
        "unique_products": int(frame["ProductId"].nunique()) if "ProductId" in frame.columns else None,
        "unique_users": int(frame["UserId"].nunique()) if "UserId" in frame.columns else None,
        "average_review_length": float(frame["review_length"].mean()) if "review_length" in frame.columns else None,
        "median_review_length": float(frame["review_length"].median()) if "review_length" in frame.columns else None,
        "average_score": float(frame["Score"].mean()) if "Score" in frame.columns else None,
        "score_counts": frame["Score"].value_counts().sort_index().to_dict() if "Score" in frame.columns else {},
    }
    return stats


def detect_unusual_reviews(frame: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    from roberta_model import analyze_roberta

    combined = add_vader_columns(frame)
    roberta_records = combined["Text"].fillna("").map(analyze_roberta)
    roberta_df = pd.DataFrame(roberta_records.tolist())
    combined = pd.concat([combined.reset_index(drop=True), roberta_df.reset_index(drop=True)], axis=1)
    combined = normalize_combined_sentiments(combined)

    combined["is_positive_low_rating"] = (
        (combined["Score"] <= 2)
        & (
            (combined["vader_label_code"] == "POS")
            | (combined["roberta_label_code"] == "POS")
        )
    )
    combined["is_negative_high_rating"] = (
        (combined["Score"] >= 4)
        & (
            (combined["vader_label_code"] == "NEG")
            | (combined["roberta_label_code"] == "NEG")
        )
    )
    combined["is_unusual"] = combined["is_positive_low_rating"] | combined["is_negative_high_rating"]

    positive_low = combined[combined["is_positive_low_rating"]].copy()
    negative_high = combined[combined["is_negative_high_rating"]].copy()

    def _serialize(records: pd.DataFrame, reason: str) -> list[dict[str, Any]]:
        columns = [
            "Id",
            "Score",
            "Summary",
            "Text",
            "vader_label",
            "vader_label_code",
            "vader_compound",
            "roberta_label",
            "roberta_label_code",
            "roberta_pos",
            "roberta_neu",
            "roberta_neg",
        ]
        trimmed = records.sort_values(
            by=["vader_compound", "roberta_pos", "roberta_neg"],
            ascending=[False, False, True],
        ).head(top_n)
        payload: list[dict[str, Any]] = []
        for _, row in trimmed[columns].iterrows():
            payload.append(
                {
                    "reason": reason,
                    "review": {
                        "Id": int(row["Id"]) if pd.notna(row["Id"]) else None,
                        "Score": int(row["Score"]) if pd.notna(row["Score"]) else None,
                        "Summary": row["Summary"],
                        "Text": row["Text"],
                    },
                    "vader": {
                        "label": row["vader_label"],
                        "code": row["vader_label_code"],
                        "compound": float(row["vader_compound"]),
                    },
                    "roberta": {
                        "label": row["roberta_label"],
                        "code": row["roberta_label_code"],
                        "positive": float(row["roberta_pos"]),
                        "neutral": float(row["roberta_neu"]),
                        "negative": float(row["roberta_neg"]),
                    },
                }
            )
        return payload

    return {
        "limit": int(len(frame)),
        "counts": {
            "positive_sentiment_low_rating": int(len(positive_low)),
            "negative_sentiment_high_rating": int(len(negative_high)),
            "total_unusual": int(len(combined[combined["is_unusual"]])),
        },
        "positive_sentiment_low_rating": _serialize(positive_low, "Positive sentiment but low rating"),
        "negative_sentiment_high_rating": _serialize(negative_high, "Negative sentiment but high rating"),
    }
