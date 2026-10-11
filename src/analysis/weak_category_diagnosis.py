from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "modeling"


# =========================================================
# SETTINGS
# =========================================================

TARGETS = {
    "phonepe": {
        "category": "Updates & UI",
        "topics": [11],
    },
    "paytm": {
        "category": "App reliability & access",
        "topics": [1, 7, 8, 9],
    },
    "googlepay": {
        "category": "Transaction & account issues",
        "topics": [3],
    },
}

REVIEWS_PER_TOPIC = 15


# =========================================================
# LOAD FILES
# =========================================================

def load_topics(app_key: str) -> pd.DataFrame:

    path = (
        MODEL_DIR
        / f"{app_key}_topics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing topic file: {path}"
        )

    return pd.read_csv(path)


def load_assignments(app_key: str) -> pd.DataFrame:

    path = (
        MODEL_DIR
        / f"{app_key}_topic_assignments.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing assignments file: {path}"
        )

    return pd.read_csv(path)


# =========================================================
# PRINT TOPIC DIAGNOSIS
# =========================================================

def diagnose_topic(
    app_key: str,
    topic_id: int,
    topics: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:

    topic_row = topics[
        topics["topic_id"]
        == topic_id
    ]

    if topic_row.empty:
        print(
            f"\nTopic {topic_id} not found."
        )
        return

    topic_row = topic_row.iloc[0]

    topic_reviews = assignments[
        assignments["topic_id"]
        == topic_id
    ].copy()

    if topic_reviews.empty:
        print(
            f"\nTopic {topic_id} has no reviews."
        )
        return

    # -----------------------------------------------------
    # Rating distribution
    # -----------------------------------------------------

    ratings = (
        topic_reviews["score"]
        .value_counts()
        .sort_index()
    )

    one_star = int(
        ratings.get(1, 0)
    )

    two_star = int(
        ratings.get(2, 0)
    )

    three_star = int(
        ratings.get(3, 0)
    )

    negative_share = (
        (one_star + two_star)
        / len(topic_reviews)
        * 100
    )

    # -----------------------------------------------------
    # Topic strength
    # -----------------------------------------------------

    mean_strength = (
        topic_reviews[
            "topic_strength"
        ].mean()
    )

    median_strength = (
        topic_reviews[
            "topic_strength"
        ].median()
    )

    # -----------------------------------------------------
    # Strongest reviews
    # -----------------------------------------------------

    strongest = (
        topic_reviews
        .sort_values(
            "topic_strength",
            ascending=False,
        )
        .head(REVIEWS_PER_TOPIC)
    )

    # -----------------------------------------------------
    # Longest reviews
    #
    # Longer reviews often reveal the actual issue
    # better than one-word "good/bad" reviews.
    # -----------------------------------------------------

    longest = (
        topic_reviews.assign(
            text_length=(
                topic_reviews[
                    "normalized_text"
                ]
                .fillna("")
                .astype(str)
                .str.len()
            )
        )
        .sort_values(
            "text_length",
            ascending=False,
        )
        .head(5)
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print("\n" + "-" * 100)

    print(
        f"TOPIC {topic_id}"
    )

    print(
        f"Keywords: "
        f"{topic_row['top_words']}"
    )

    print(
        f"Topic size: "
        f"{len(topic_reviews):,}"
    )

    print(
        "Ratings: "
        f"1★={one_star:,}, "
        f"2★={two_star:,}, "
        f"3★={three_star:,}"
    )

    print(
        f"1–2★ share: "
        f"{negative_share:.2f}%"
    )

    print(
        f"Topic strength: "
        f"mean={mean_strength:.4f}, "
        f"median={median_strength:.4f}"
    )

    # -----------------------------------------------------
    # Strong examples
    # -----------------------------------------------------

    print(
        "\nStrongest reviews:"
    )

    for i, (_, review) in enumerate(
        strongest.iterrows(),
        start=1,
    ):

        text = str(
            review["normalized_text"]
        )

        if len(text) > 400:
            text = (
                text[:400]
                + "..."
            )

        print(
            f"  {i}. "
            f"[{int(review['score'])}★] "
            f"{text}"
        )

        print(
            f"     strength="
            f"{review['topic_strength']:.4f}"
        )

    # -----------------------------------------------------
    # Long examples
    # -----------------------------------------------------

    print(
        "\nLonger reviews from this topic:"
    )

    for i, (_, review) in enumerate(
        longest.iterrows(),
        start=1,
    ):

        text = str(
            review["normalized_text"]
        )

        if len(text) > 600:
            text = (
                text[:600]
                + "..."
            )

        print(
            f"  {i}. "
            f"[{int(review['score'])}★] "
            f"{text}"
        )


# =========================================================
# DIAGNOSE APP
# =========================================================

def diagnose_app(
    app_key: str,
    target: dict,
) -> None:

    print("\n" + "=" * 100)

    print(
        f"WEAK CATEGORY DIAGNOSIS: "
        f"{app_key.upper()}"
    )

    print(
        f"Category: "
        f"{target['category']}"
    )

    print(
        f"Underlying NMF topics: "
        f"{target['topics']}"
    )

    print("=" * 100)

    topics = load_topics(
        app_key
    )

    assignments = load_assignments(
        app_key
    )

    for topic_id in target["topics"]:

        diagnose_topic(
            app_key=app_key,
            topic_id=topic_id,
            topics=topics,
            assignments=assignments,
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key, target in TARGETS.items():

        diagnose_app(
            app_key=app_key,
            target=target,
        )

    print("\n" + "=" * 100)

    print(
        "WEAK CATEGORY DIAGNOSIS COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()