from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "modeling" / "refined"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SETTINGS
# =========================================================

# We only refine the apps/categories where the
# 15-topic model was too broad.
APP_SETTINGS = {
    "paytm": {
        "n_topics": 25,
    },
    "googlepay": {
        "n_topics": 25,
    },
}

TOP_WORDS = 12

MIN_DF = 10
MAX_DF = 0.70

MAX_FEATURES = 50000


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
}


# =========================================================
# TEXT PREPARATION
# =========================================================

def prepare_text(
    df: pd.DataFrame,
) -> pd.Series:

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

    # Keep English/Romanized words and numbers.
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
# RUN REFINED MODEL
# =========================================================

def run_refined_model(
    app_key: str,
    n_topics: int,
) -> None:

    print("\n" + "=" * 80)

    print(
        f"REFINED TOPIC MODEL: "
        f"{app_key.upper()}"
    )

    print("=" * 80)

    input_path = (
        PROCESSED_DIR
        / f"{app_key}_processed.csv"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Processed file not found: "
            f"{input_path}"
        )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print(
        f"Loading: {input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    # -----------------------------------------------------
    # Restrict to topic-model population.
    # -----------------------------------------------------

    model_df = df[
        df["text_eligible"]
        == True
    ].copy()

    print(
        f"Text-eligible reviews: "
        f"{len(model_df):,}"
    )

    model_df = model_df[
        model_df["score"] <= 3
    ].copy()

    print(
        f"1–3★ reviews used: "
        f"{len(model_df):,}"
    )

    # -----------------------------------------------------
    # Prepare normalized text
    # -----------------------------------------------------

    model_df["model_text"] = (
        prepare_text(model_df)
    )

    model_df = model_df[
        model_df["model_text"]
        .str.len()
        > 0
    ].copy()

    print(
        f"Reviews with usable text: "
        f"{len(model_df):,}"
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
        max_features=MAX_FEATURES,
    )

    X = vectorizer.fit_transform(
        model_df["model_text"]
    )

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} documents × "
        f"{X.shape[1]:,} features"
    )

    if X.shape[1] < n_topics:

        raise RuntimeError(
            f"{app_key}: only "
            f"{X.shape[1]} features available "
            f"for {n_topics} topics."
        )

    # -----------------------------------------------------
    # NMF
    # -----------------------------------------------------

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=500,
    )

    W = model.fit_transform(
        X
    )

    # -----------------------------------------------------
    # Topic definitions
    # -----------------------------------------------------

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    topic_rows = []

    print(
        "\nCandidate refined topics:"
    )

    for topic_idx, topic in enumerate(
        model.components_
    ):

        top_indices = (
            topic.argsort()[::-1]
            [:TOP_WORDS]
        )

        words = [
            feature_names[i]
            for i in top_indices
        ]

        topic_id = (
            topic_idx + 1
        )

        print(
            f"\nTopic {topic_id}: "
            + ", ".join(words)
        )

        topic_rows.append(
            {
                "topic_id": topic_id,
                "top_words": ", ".join(words),
            }
        )

    # -----------------------------------------------------
    # Assign dominant topic
    # -----------------------------------------------------

    model_df["topic_id"] = (
        W.argmax(axis=1)
        + 1
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

    print(
        "\nTopic sizes:"
    )

    for topic_id, count in (
        topic_counts.items()
    ):

        percentage = (
            count
            / len(model_df)
            * 100
        )

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
        / f"{app_key}_refined_topics.csv"
    )

    pd.DataFrame(
        topic_rows
    ).to_csv(
        topics_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Save assignments
    # -----------------------------------------------------

    assignments_path = (
        OUTPUT_DIR
        / f"{app_key}_refined_assignments.csv"
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
        f"\nSaved topics to: "
        f"{topics_path}"
    )

    print(
        f"Saved assignments to: "
        f"{assignments_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key, settings in (
        APP_SETTINGS.items()
    ):

        run_refined_model(
            app_key=app_key,
            n_topics=settings[
                "n_topics"
            ],
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REFINED TOPIC MODELING COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()