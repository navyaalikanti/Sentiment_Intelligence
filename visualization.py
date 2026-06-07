from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sentiment_analyzer import (
    add_ml_columns,
    agreement_and_distribution,
    get_agreement_summary,
    pairwise_agreement_summary,
    sentiment_distribution_counts,
)


plt.style.use("ggplot")


def rating_distribution_data(frame) -> list[dict[str, Any]]:
    if "Score" not in frame.columns:
        return []

    counts = frame["Score"].value_counts().sort_index()
    return [
        {"score": int(score), "count": int(count)}
        for score, count in counts.items()
    ]


def sentiment_distribution_data(frame) -> dict[str, list[dict[str, Any]]]:
    return sentiment_distribution_counts(frame)


def model_comparison_chart_data(frame) -> dict[str, Any]:
    enriched = add_ml_columns(frame)
    if "Score" not in enriched.columns:
        return {"by_rating": [], "agreement": {}, "columns": []}

    score_group = enriched.groupby("Score", dropna=False)
    rows: list[dict[str, Any]] = []
    for score, group in score_group:
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
        rows.append(row)

    agreement = get_agreement_summary(enriched)
    agreement["pairwise"] = {
        "vader_roberta": agreement.copy(),
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
        enriched["vader_label_code"].isin({"POS", "NEG", "NEU"})
        & enriched["roberta_label_code"].isin({"POS", "NEG", "NEU"})
        & enriched["ml_label_code"].isin({"POS", "NEG", "NEU"})
    ]
    agreement["all_three_agree_percentage"] = round(
        (agreement["all_three_agree"] / len(comparable)) * 100, 2
    ) if len(comparable) else 0.0

    return {
        "by_rating": rows,
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


def create_pairplot_image(frame, limit: int = 200) -> str | None:
    required_columns = [
        "vader_neg",
        "vader_neu",
        "vader_pos",
        "roberta_neg",
        "roberta_neu",
        "roberta_pos",
    ]
    optional_ml_columns = ["ml_neg", "ml_neu", "ml_pos"]
    vars_to_plot = required_columns + [column for column in optional_ml_columns if column in frame.columns]
    if not all(column in frame.columns for column in required_columns):
        return None

    sample = frame[vars_to_plot + [column for column in ["Score"] if column in frame.columns]].head(limit).dropna()
    if sample.empty:
        return None

    try:
        pairplot = sns.pairplot(
            data=sample,
            vars=vars_to_plot,
            hue="Score" if "Score" in sample.columns else None,
            palette="tab10",
            corner=True,
        )

        buffer = io.BytesIO()
        pairplot.fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
        plt.close(pairplot.fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception:
        return None
