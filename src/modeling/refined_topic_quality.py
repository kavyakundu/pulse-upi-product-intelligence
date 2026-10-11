from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFINED_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "refined"
)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "paytm",
    "googlepay",
]

REVIEWS_PER_TOPIC = 10

MIN_TOPIC_STRENGTH = 0.05


# =========================================================
# LOAD FILES
# =========================================================

def load_topics(app_key: str) -> pd.DataFrame:

    path = (
        REFINED_DIR
        / f"{app_key}_refined_topics.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing refined topic file: "
            f"{path}"
        )

    topics = pd.read_csv(
        path
    )

    required_columns = {
        "topic_id",
        "top_words",
    }

    missing = (
        required_columns
        - set(topics.columns)
    )

    if missing:

        raise ValueError(
            f"{app_key}: topic file missing "
            f"columns: {sorted(missing)}"
        )

    return topics


def load_assignments(app_key: str) -> pd.DataFrame:

    path = (
        REFINED_DIR
        / f"{app_key}_refined_assignments.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing refined assignment file: "
            f"{path}"
        )

    assignments = pd.read_csv(
        path
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "normalized_text",
        "topic_id",
        "topic_strength",
    }

    missing = (
        required_columns
        - set(assignments.columns)
    )

    if missing:

        raise ValueError(
            f"{app_key}: assignment file missing "
            f"columns: {sorted(missing)}"
        )

    return assignments


# =========================================================
# TOPIC ANALYSIS
# =========================================================

def analyze_topic(
    topic_id: int,
    topic_row: pd.Series,
    assignments: pd.DataFrame,
) -> None:

    topic_reviews = assignments[
        assignments["topic_id"]
        == topic_id
    ].copy()

    if topic_reviews.empty:

        return

    topic_size = len(
        topic_reviews
    )

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
        / topic_size
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

    max_strength = (
        topic_reviews[
            "topic_strength"
        ].max()
    )

    strong_reviews = topic_reviews[
        topic_reviews["topic_strength"]
        >= MIN_TOPIC_STRENGTH
    ].copy()

    strong_share = (
        len(strong_reviews)
        / topic_size
        * 100
    )

    if strong_reviews.empty:

        strong_reviews = (
            topic_reviews.copy()
        )

    # -----------------------------------------------------
    # Representative reviews
    # -----------------------------------------------------

    representative = (
        strong_reviews
        .sort_values(
            "topic_strength",
            ascending=False,
        )
        .head(REVIEWS_PER_TOPIC)
    )

    # -----------------------------------------------------
    # Longer reviews
    # -----------------------------------------------------

    longer = (
        strong_reviews
        .assign(
            text_length=(
                strong_reviews[
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

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"TOPIC {topic_id}"
    )

    print(
        f"Keywords: "
        f"{topic_row['top_words']}"
    )

    print(
        f"Topic size: "
        f"{topic_size:,}"
    )

    print(
        "Rating distribution: "
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
        f"median={median_strength:.4f}, "
        f"max={max_strength:.4f}"
    )

    print(
        f"Strong-assignment coverage: "
        f"{strong_share:.2f}%"
    )

    # -----------------------------------------------------
    # Strong examples
    # -----------------------------------------------------

    print(
        "\nStrongest representative reviews:"
    )

    for i, (_, review) in enumerate(
        representative.iterrows(),
        start=1,
    ):

        text = str(
            review["normalized_text"]
        )

        if len(text) > 350:

            text = (
                text[:350]
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
        "\nLonger reviews:"
    )

    for i, (_, review) in enumerate(
        longer.iterrows(),
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
# ANALYZE ONE APP
# =========================================================

def analyze_app(
    app_key: str,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"REFINED TOPIC QUALITY: "
        f"{app_key.upper()}"
    )

    print(
        "=" * 100
    )

    topics = load_topics(
        app_key
    )

    assignments = load_assignments(
        app_key
    )

    print(
        f"1–3★ reviews: "
        f"{len(assignments):,}"
    )

    for _, topic in topics.iterrows():

        topic_id = int(
            topic["topic_id"]
        )

        analyze_topic(
            topic_id=topic_id,
            topic_row=topic,
            assignments=assignments,
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key in APPS:

        analyze_app(
            app_key
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REFINED TOPIC QUALITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()