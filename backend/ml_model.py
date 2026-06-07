from __future__ import annotations

import html
import json
import math
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "Reviews.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
EVALUATION_PATH = RESULTS_DIR / "ml_evaluation.json"

LABEL_ORDER = ["NEG", "NEU", "POS"]
LABEL_CODE_TO_NAME = {
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive",
}
LABEL_NAME_TO_CODE = {value: key for key, value in LABEL_CODE_TO_NAME.items()}
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


def _ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_for_model(text: Any) -> str:
    normalized = clean_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize(text: Any) -> list[str]:
    normalized = _normalize_for_model(text)
    if not normalized:
        return []
    tokens = re.findall(r"[a-z][a-z']+", normalized)
    return [token for token in tokens if token not in DEFAULT_STOPWORDS]


def score_to_label_code(score: Any) -> str:
    try:
        numeric = int(float(score))
    except (TypeError, ValueError):
        return "NEU"
    if numeric >= 4:
        return "POS"
    if numeric <= 2:
        return "NEG"
    return "NEU"


def _label_codes_to_names(labels: Iterable[str]) -> list[str]:
    return [LABEL_CODE_TO_NAME.get(label, "Neutral") for label in labels]


class SimpleTfidfVectorizer:
    def __init__(self, max_features: int = 7000, min_df: int = 2, ngram_range: tuple[int, int] = (1, 2)):
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def _terms_for_tokens(self, tokens: list[str]) -> list[str]:
        terms: list[str] = []
        min_n, max_n = self.ngram_range
        if min_n <= 1 <= max_n:
            terms.extend(tokens)
        if max_n >= 2:
            terms.extend(f"{tokens[index]}_{tokens[index + 1]}" for index in range(len(tokens) - 1))
        return terms

    def fit(self, texts: Iterable[Any]) -> "SimpleTfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        document_count = 0

        for text in texts:
            tokens = _tokenize(text)
            if not tokens:
                continue
            document_count += 1
            document_frequency.update(set(self._terms_for_tokens(tokens)))

        ordered_terms = [
            term
            for term, freq in sorted(document_frequency.items(), key=lambda item: (-item[1], item[0]))
            if freq >= self.min_df
        ]
        ordered_terms = ordered_terms[: self.max_features]

        self.vocabulary_ = {term: index for index, term in enumerate(ordered_terms)}
        self.feature_names_ = ordered_terms
        if ordered_terms:
            self.idf_ = np.array(
                [
                    math.log((1 + document_count) / (1 + document_frequency[term])) + 1.0
                    for term in ordered_terms
                ],
                dtype=np.float64,
            )
        else:
            self.idf_ = np.zeros(0, dtype=np.float64)
        return self

    def transform(self, texts: Iterable[Any]) -> csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        texts_list = list(texts)

        for row_index, text in enumerate(texts_list):
            tokens = _tokenize(text)
            if not tokens:
                continue

            terms = self._terms_for_tokens(tokens)
            term_counts = Counter(term for term in terms if term in self.vocabulary_)
            if not term_counts:
                continue

            total_terms = float(sum(term_counts.values()))
            if total_terms <= 0:
                continue

            for term, count in term_counts.items():
                column_index = self.vocabulary_[term]
                tf = count / total_terms
                value = tf * float(self.idf_[column_index])
                rows.append(row_index)
                cols.append(column_index)
                data.append(value)

        matrix = csr_matrix((data, (rows, cols)), shape=(len(texts_list), len(self.vocabulary_)), dtype=np.float64)
        if matrix.nnz:
            squared = matrix.multiply(matrix).sum(axis=1)
            norms = np.sqrt(np.asarray(squared).ravel())
            norms[norms == 0] = 1.0
            matrix = matrix.multiply((1.0 / norms)[:, None]).tocsr()
        return matrix

    def fit_transform(self, texts: Iterable[Any]) -> csr_matrix:
        return self.fit(texts).transform(texts)


def _softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    scores = scores - scores.max(axis=1, keepdims=True)
    exps = np.exp(scores)
    denominators = exps.sum(axis=1, keepdims=True)
    denominators[denominators == 0] = 1.0
    return exps / denominators


class LogisticRegressionSentimentModel:
    def __init__(
        self,
        *,
        learning_rate: float = 0.35,
        epochs: int = 18,
        batch_size: int = 256,
        regularization: float = 0.0005,
        random_state: int = 42,
        label_order: list[str] | None = None,
    ):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.regularization = float(regularization)
        self.random_state = int(random_state)
        self.label_order = label_order or LABEL_ORDER
        self.classes_: list[str] = []
        self.class_to_index_: dict[str, int] = {}
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, X: csr_matrix, y: Iterable[str]) -> "LogisticRegressionSentimentModel":
        y_list = [str(label).upper() for label in y]
        self.classes_ = list(self.label_order)
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        y_indices = np.array([self.class_to_index_.get(label, 1) for label in y_list], dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self.weights_ = np.zeros((n_features, n_classes), dtype=np.float64)
        self.bias_ = np.zeros(n_classes, dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        learning_rate = self.learning_rate

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_indices[batch_indices]

                logits = X_batch.dot(self.weights_) + self.bias_
                probabilities = _softmax(logits)
                one_hot = np.zeros((len(batch_indices), n_classes), dtype=np.float64)
                one_hot[np.arange(len(batch_indices)), y_batch] = 1.0

                error = probabilities - one_hot
                grad_weights = X_batch.T.dot(error) / max(len(batch_indices), 1)
                grad_bias = error.mean(axis=0)
                grad_weights += self.regularization * self.weights_

                self.weights_ -= learning_rate * grad_weights
                self.bias_ -= learning_rate * grad_bias

            learning_rate *= 0.92

        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        logits = X.dot(self.weights_) + self.bias_
        return _softmax(np.asarray(logits))

    def predict(self, X: csr_matrix) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices = probabilities.argmax(axis=1)
        return np.array([self.classes_[index] for index in indices], dtype=object)

    def predict_single(self, vectorizer: SimpleTfidfVectorizer, text: str) -> dict[str, Any]:
        matrix = vectorizer.transform([text])
        probabilities = self.predict_proba(matrix)[0]
        best_index = int(probabilities.argmax())
        label_code = self.classes_[best_index]
        return {
            "label": LABEL_CODE_TO_NAME.get(label_code, "Neutral"),
            "label_code": label_code,
            "pos": float(probabilities[self.class_to_index_["POS"]]),
            "neu": float(probabilities[self.class_to_index_["NEU"]]),
            "neg": float(probabilities[self.class_to_index_["NEG"]]),
            "confidence": float(probabilities[best_index]),
        }


def _read_training_frame(sample_size: int) -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH, nrows=int(sample_size), low_memory=False, encoding_errors="ignore")
    frame = frame.copy()
    for column in ("Summary", "Text"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").map(clean_text)
    if "Score" in frame.columns:
        frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
        frame = frame.dropna(subset=["Score"]).copy()
        frame["Score"] = frame["Score"].astype(int)
    frame["combined_text"] = (
        frame.get("Summary", pd.Series("", index=frame.index)).fillna("")
        + " "
        + frame.get("Text", pd.Series("", index=frame.index)).fillna("")
    ).map(clean_text)
    frame["label_code"] = frame["Score"].map(score_to_label_code)
    return frame


def _stratified_split(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label_code in LABEL_ORDER:
        label_indices = frame.index[frame["label_code"] == label_code].to_numpy().copy()
        if not len(label_indices):
            continue
        rng.shuffle(label_indices)
        if len(label_indices) == 1:
            train_indices.extend(label_indices.tolist())
            continue

        test_count = max(1, int(round(len(label_indices) * test_size)))
        test_count = min(test_count, len(label_indices) - 1)
        test_indices.extend(label_indices[:test_count].tolist())
        train_indices.extend(label_indices[test_count:].tolist())

    train_frame = frame.loc[sorted(train_indices)].reset_index(drop=True)
    test_frame = frame.loc[sorted(test_indices)].reset_index(drop=True)
    return train_frame, test_frame


def _compute_classification_report(y_true: list[str], y_pred: list[str], label_order: list[str] = LABEL_ORDER) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(label_order)}
    matrix = np.zeros((len(label_order), len(label_order)), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label not in label_to_index or predicted_label not in label_to_index:
            continue
        matrix[label_to_index[true_label], label_to_index[predicted_label]] += 1

    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    report: dict[str, Any] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    supports: list[int] = []

    for label_code in label_order:
        index = label_to_index[label_code]
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        report[LABEL_CODE_TO_NAME[label_code]] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": support,
        }

        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        supports.append(support)

    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    total_support = int(sum(supports))
    weighted_precision = float(np.average(precision_values, weights=supports)) if total_support else 0.0
    weighted_recall = float(np.average(recall_values, weights=supports)) if total_support else 0.0
    weighted_f1 = float(np.average(f1_values, weights=supports)) if total_support else 0.0

    report["accuracy"] = round(accuracy, 4)
    report["macro avg"] = {
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1_score": round(macro_f1, 4),
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": round(weighted_precision, 4),
        "recall": round(weighted_recall, 4),
        "f1_score": round(weighted_f1, 4),
        "support": total_support,
    }

    return {
        "labels": _label_codes_to_names(label_order),
        "matrix": matrix.astype(int).tolist(),
        "accuracy": round(accuracy, 4),
        "report": report,
    }


def _evaluate_baseline_model(texts: list[str], y_true: list[str], model_name: str) -> dict[str, Any]:
    if model_name == "vader":
        from sentiment_analyzer import vader_scores

        y_pred = [str(vader_scores(text).get("label_code", "NEU")).upper() for text in texts]
    elif model_name == "roberta":
        from roberta_model import analyze_roberta

        y_pred = [str(analyze_roberta(text).get("roberta_label_code", "NEU")).upper() for text in texts]
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    return _compute_classification_report(y_true, y_pred)


def _build_training_artifacts(sample_size: int = 6000, test_size: float = 0.2) -> dict[str, Any]:
    frame = _read_training_frame(sample_size)
    train_frame, test_frame = _stratified_split(frame, test_size=test_size)

    vectorizer = SimpleTfidfVectorizer(max_features=7000, min_df=2, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_frame["combined_text"].tolist())
    X_test = vectorizer.transform(test_frame["combined_text"].tolist())

    model = LogisticRegressionSentimentModel()
    model.fit(X_train, train_frame["label_code"].tolist())
    y_test = test_frame["label_code"].tolist()
    y_pred = model.predict(X_test).tolist()

    logistic_report = _compute_classification_report(y_test, y_pred)

    baseline_texts = test_frame["combined_text"].tolist()
    vader_report = _evaluate_baseline_model(baseline_texts, y_test, "vader")
    roberta_report = _evaluate_baseline_model(baseline_texts, y_test, "roberta")

    comparison_chart_data = [
        {
            "model": "Logistic Regression",
            "accuracy": logistic_report["accuracy"],
            "precision": logistic_report["report"]["weighted avg"]["precision"],
            "recall": logistic_report["report"]["weighted avg"]["recall"],
            "f1": logistic_report["report"]["weighted avg"]["f1_score"],
        },
        {
            "model": "VADER",
            "accuracy": vader_report["accuracy"],
            "precision": vader_report["report"]["weighted avg"]["precision"],
            "recall": vader_report["report"]["weighted avg"]["recall"],
            "f1": vader_report["report"]["weighted avg"]["f1_score"],
        },
        {
            "model": "RoBERTa",
            "accuracy": roberta_report["accuracy"],
            "precision": roberta_report["report"]["weighted avg"]["precision"],
            "recall": roberta_report["report"]["weighted avg"]["recall"],
            "f1": roberta_report["report"]["weighted avg"]["f1_score"],
        },
    ]

    report = {
        "dataset_size": int(len(frame)),
        "training_samples": int(len(train_frame)),
        "testing_samples": int(len(test_frame)),
        "tfidf_features": int(len(vectorizer.feature_names_)),
        "label_order": _label_codes_to_names(LABEL_ORDER),
        "model_metrics": {
            "logistic_regression": {
                "accuracy": logistic_report["accuracy"],
                "precision": logistic_report["report"]["weighted avg"]["precision"],
                "recall": logistic_report["report"]["weighted avg"]["recall"],
                "f1_score": logistic_report["report"]["weighted avg"]["f1_score"],
            },
            "vader": {
                "accuracy": vader_report["accuracy"],
                "precision": vader_report["report"]["weighted avg"]["precision"],
                "recall": vader_report["report"]["weighted avg"]["recall"],
                "f1_score": vader_report["report"]["weighted avg"]["f1_score"],
            },
            "roberta": {
                "accuracy": roberta_report["accuracy"],
                "precision": roberta_report["report"]["weighted avg"]["precision"],
                "recall": roberta_report["report"]["weighted avg"]["recall"],
                "f1_score": roberta_report["report"]["weighted avg"]["f1_score"],
            },
        },
        "classification_report": logistic_report["report"],
        "confusion_matrix": {
            "labels": logistic_report["labels"],
            "matrix": logistic_report["matrix"],
        },
        "comparison_chart_data": comparison_chart_data,
        "summary": {
            "dataset_size": int(len(frame)),
            "training_samples": int(len(train_frame)),
            "testing_samples": int(len(test_frame)),
            "tfidf_features": int(len(vectorizer.feature_names_)),
        },
    }

    return {
        "vectorizer": vectorizer,
        "model": model,
        "report": report,
    }


def save_model_assets(model: LogisticRegressionSentimentModel, vectorizer: SimpleTfidfVectorizer, report: dict[str, Any]) -> None:
    _ensure_dirs()
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with EVALUATION_PATH.open("w", encoding="utf-8") as evaluation_file:
        json.dump(report, evaluation_file, indent=2)


def train_and_save_model(sample_size: int = 6000, test_size: float = 0.2) -> dict[str, Any]:
    artifacts = _build_training_artifacts(sample_size=sample_size, test_size=test_size)
    save_model_assets(artifacts["model"], artifacts["vectorizer"], artifacts["report"])
    return artifacts["report"]


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_model_bundle() -> tuple[LogisticRegressionSentimentModel, SimpleTfidfVectorizer]:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_and_save_model()

    model = _load_pickle(MODEL_PATH)
    vectorizer = _load_pickle(VECTORIZER_PATH)
    return model, vectorizer


def load_ml_evaluation_report() -> dict[str, Any]:
    if EVALUATION_PATH.exists():
        with EVALUATION_PATH.open("r", encoding="utf-8") as evaluation_file:
            return json.load(evaluation_file)

    return train_and_save_model()


@lru_cache(maxsize=8192)
def analyze_logistic_regression(text: str) -> dict[str, Any]:
    normalized = clean_text(text)
    model, vectorizer = load_model_bundle()
    return model.predict_single(vectorizer, normalized)


def create_model_comparison_payload(frame: pd.DataFrame) -> dict[str, Any]:
    from sentiment_analyzer import add_ml_columns, get_agreement_summary, pairwise_agreement_summary

    enriched = add_ml_columns(frame)
    if "Score" not in enriched.columns:
        return {"by_rating": [], "agreement": {}, "columns": []}

    grouped_rows: list[dict[str, Any]] = []
    for score, group in enriched.groupby("Score", dropna=False):
        row = {"score": int(score), "count": int(len(group))}
        for column in [
            "vader_neg",
            "vader_neu",
            "vader_pos",
            "vader_compound",
            "roberta_neg",
            "roberta_neu",
            "roberta_pos",
            "ml_neg",
            "ml_neu",
            "ml_pos",
            "ml_confidence",
        ]:
            if column in group.columns:
                row[column] = float(group[column].mean())
        grouped_rows.append(row)

    vader_roberta_agreement = get_agreement_summary(enriched)
    agreement = dict(vader_roberta_agreement)
    agreement["pairwise"] = {
        "vader_roberta": vader_roberta_agreement,
        "vader_ml": pairwise_agreement_summary(enriched, "vader", "ml"),
        "roberta_ml": pairwise_agreement_summary(enriched, "roberta", "ml"),
    }
    agreement["all_three_agree"] = int(
        (
            (enriched["vader_label_code"] == enriched["roberta_label_code"])
            & (enriched["vader_label_code"] == enriched["ml_label_code"])
        ).sum()
    )
    comparable = enriched[
        enriched["vader_label_code"].isin(LABEL_ORDER)
        & enriched["roberta_label_code"].isin(LABEL_ORDER)
        & enriched["ml_label_code"].isin(LABEL_ORDER)
    ]
    agreement["all_three_agree_percentage"] = round(
        (agreement["all_three_agree"] / len(comparable)) * 100, 2
    ) if len(comparable) else 0.0

    return {
        "by_rating": grouped_rows,
        "agreement": agreement,
        "columns": [column for column in [
            "vader_neg",
            "vader_neu",
            "vader_pos",
            "vader_compound",
            "roberta_neg",
            "roberta_neu",
            "roberta_pos",
            "ml_neg",
            "ml_neu",
            "ml_pos",
            "ml_confidence",
        ] if column in enriched.columns],
    }
