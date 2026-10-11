from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "modeling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

N_TOPICS = 15
TOP_WORDS = 12

MIN_DF = 10
MAX_DF = 0.70


# =========================================================
# TEXT PREPARATION
# =========================================================

def prepare_text(df: pd.DataFrame) -> pd.Series:
    """
    Prepare review text for TF-IDF/NMF.

    We deliberately keep the preprocessing conservative.
    The goal is to discover product-related language,
    not rewrite the original reviews.
    """

    text = (
        df["normalized_text"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # Remove URLs.
    text = text.str.replace(
        r"https?://\S+|www\.\S+",
        " ",
        regex=True,
    )

    # Keep alphabetic/numeric words.
    text = text.str.replace(
        r"[^a-z0-9\s]",
        " ",
        regex=True,
    )

    # Normalize whitespace.
    text = text.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    return text


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {
    "app",
    "apps",
    "application",
    "use",
    "using",
    "used",
    "good",
    "nice",
    "best",
    "very",
    "much",
    "really",
    "also",
    "one",
    "get",
    "got",
    "getting",
    "please",
    "pls",
    "plz",
    "sir",
    "hai",
    "ha",
    "h",
    "he",
    "hi",
    "ka",
    "ke",
    "ki",
    "ko",
    "me",
    "mein",
    "mera",
    "mere",
    "meri",
    "ye",
    "yah",
    "is",
    "iss",
    "se",
    "par",
    "per",
    "aur",
    "or",
    "to",
    "the",
    "this",
    "that",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "for",
    "from",
    "with",
    "you",
    "your",
    "my",
    "i",
    "we",
    "they",
    "they",
}


# =========================================================
# TOPIC MODEL
# =========================================================

def run_topic_model(
    df: pd.DataFrame,
    app_key: str,
) -> None:

    print("\n" + "=" * 70)
    print(f"TOPIC MODELING: {app_key.upper()}")
    print("=" * 70)

    # -----------------------------------------------------
    # Keep text eligible reviews.
    # -----------------------------------------------------

    model_df = df[
        df["text_eligible"] == True
    ].copy()

    print(
        f"Text-eligible reviews: "
        f"{len(model_df):,}"
    )

    # Keep negative reviews as the primary pain signal.
    #
    # We include 1-3 star reviews because these are the
    # reviews most likely to contain actionable product pain.
    #
    model_df = model_df[
        model_df["score"] <= 3
    ].copy()

    print(
        f"1-3 star reviews used: "
        f"{len(model_df):,}"
    )

    model_df["model_text"] = prepare_text(model_df)

    # Remove empty text after cleaning.
    model_df = model_df[
        model_df["model_text"].str.len() > 0
    ].copy()

    print(
        f"Reviews with usable text: "
        f"{len(model_df):,}"
    )

    if len(model_df) < 100:
        raise RuntimeError(
            f"Not enough reviews for {app_key}."
        )

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
    )

    X = vectorizer.fit_transform(
        model_df["model_text"]
    )

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} documents × "
        f"{X.shape[1]:,} features"
    )

    # -----------------------------------------------------
    # NMF
    # -----------------------------------------------------

    n_topics = min(
        N_TOPICS,
        max(2, X.shape[1] // 10),
    )

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=400,
    )

    W = model.fit_transform(X)

    # -----------------------------------------------------
    # Display topics
    # -----------------------------------------------------

    feature_names = vectorizer.get_feature_names_out()

    print("\nCandidate topics:")

    topic_rows = []

    for topic_idx, topic in enumerate(
        model.components_
    ):

        top_indices = topic.argsort()[
            ::-1
        ][:TOP_WORDS]

        words = [
            feature_names[i]
            for i in top_indices
        ]

        print(
            f"\nTopic {topic_idx + 1}: "
            + ", ".join(words)
        )

        topic_rows.append(
            {
                "topic_id": topic_idx + 1,
                "top_words": ", ".join(words),
            }
        )

    # -----------------------------------------------------
    # Assign dominant topic
    # -----------------------------------------------------

    model_df["topic_id"] = (
        W.argmax(axis=1) + 1
    )

    model_df["topic_strength"] = (
        W.max(axis=1)
    )

    # -----------------------------------------------------
    # Topic sizes
    # -----------------------------------------------------

    topic_counts = (
        model_df["topic_id"]
        .value_counts()
        .sort_index()
    )

    print("\nTopic sizes:")

    for topic_id, count in topic_counts.items():

        percentage = (
            count / len(model_df)
        ) * 100

        print(
            f"  Topic {topic_id:2d}: "
            f"{count:7,} "
            f"({percentage:5.2f}%)"
        )

    # -----------------------------------------------------
    # Save topic definitions
    # -----------------------------------------------------

    topics_path = (
        OUTPUT_DIR
        / f"{app_key}_topics.csv"
    )

    pd.DataFrame(topic_rows).to_csv(
        topics_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Save review-level assignments
    # -----------------------------------------------------

    assignments_path = (
        OUTPUT_DIR
        / f"{app_key}_topic_assignments.csv"
    )

    model_df[
        [
            "reviewId",
            "app_key",
            "score",
            "at",
            "normalized_text",
            "topic_id",
            "topic_strength",
        ]
    ].to_csv(
        assignments_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved topics to: {topics_path}"
    )

    print(
        f"Saved assignments to: "
        f"{assignments_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key in APPS:

        path = (
            PROCESSED_DIR
            / f"{app_key}_processed.csv"
        )

        print(
            f"\nLoading: {path}"
        )

        df = pd.read_csv(path)

        run_topic_model(
            df,
            app_key,
        )

    print("\n" + "=" * 70)
    print("TOPIC MODELING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()