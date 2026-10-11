from pathlib import Path

import pandas as pd
from transformers import pipeline


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "sentiment_topic_validation_sample.csv"
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

MODEL_NAME = "ganeshkharad/gk-hinglish-sentiment"


# ---------------------------------------------------------
# Label mapping
# ---------------------------------------------------------

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print(f"Loading validation data: {VALIDATION_FILE}")

    df = pd.read_csv(VALIDATION_FILE)

    df = df[
        df["human_sentiment"].isin(
            ["Positive", "Neutral", "Negative"]
        )
    ].copy()

    print(f"Validation rows: {len(df):,}")

    print(f"\nLoading model: {MODEL_NAME}")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=128,
    )

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    predictions = classifier(
        texts,
        batch_size=16,
    )

    df["model_label_raw"] = [
        prediction["label"]
        for prediction in predictions
    ]

    df["model_score"] = [
        prediction["score"]
        for prediction in predictions
    ]

    df["model_sentiment"] = (
        df["model_label_raw"]
        .map(LABEL_MAP)
    )

    # -----------------------------------------------------
    # Overall accuracy
    # -----------------------------------------------------

    accuracy = (
        df["model_sentiment"]
        == df["human_sentiment"]
    ).mean()

    print("\n" + "=" * 70)
    print("HINGLISH MODEL VALIDATION RESULTS")
    print("=" * 70)

    print(f"Rows evaluated: {len(df):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy %: {accuracy * 100:.2f}%")

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    confusion = pd.crosstab(
        df["human_sentiment"],
        df["model_sentiment"],
        rownames=["Human"],
        colnames=["Model"],
        dropna=False,
    )

    print("\nConfusion matrix:")
    print(confusion)

    # -----------------------------------------------------
    # Accuracy by app
    # -----------------------------------------------------

    print("\nAccuracy by app:")

    for app_key, group in df.groupby("app_key"):

        app_accuracy = (
            group["model_sentiment"]
            == group["human_sentiment"]
        ).mean()

        print(
            f"  {app_key}: "
            f"{app_accuracy * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Accuracy by star rating
    # -----------------------------------------------------

    print("\nAccuracy by star rating:")

    for rating, group in df.groupby("score"):

        rating_accuracy = (
            group["model_sentiment"]
            == group["human_sentiment"]
        ).mean()

        print(
            f"  {rating}-star: "
            f"{rating_accuracy * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Accuracy on likely Hinglish / code-mixed subset
    # -----------------------------------------------------

    hinglish = df[
        df["contains_code_mix"] == True
    ].copy()

    print("\nCode-mixed subset:")

    if len(hinglish) > 0:

        hinglish_accuracy = (
            hinglish["model_sentiment"]
            == hinglish["human_sentiment"]
        ).mean()

        print(f"  Rows: {len(hinglish):,}")
        print(
            f"  Accuracy: "
            f"{hinglish_accuracy * 100:.2f}%"
        )

    else:

        print("  No Latin + Devanagari reviews in sample.")

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "validation"
        / "hinglish_sentiment_predictions.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()