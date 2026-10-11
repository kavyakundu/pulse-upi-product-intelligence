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

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

REVIEWS_PER_TOPIC = 10

# Ignore reviews where the topic assignment is
# extremely weak.
MIN_TOPIC_STRENGTH = 0.01


# =========================================================
# INSPECT ONE APP
# =========================================================

def inspect_app(app_key: str) -> None:

    assignments_path = (
        MODEL_DIR
        / f"{app_key}_topic_assignments.csv"
    )

    topics_path = (
        MODEL_DIR
        / f"{app_key}_topics.csv"
    )

    print("\n" + "=" * 90)
    print(
        f"TOPIC QUALITY ANALYSIS: "
        f"{app_key.upper()}"
    )
    print("=" * 90)

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Missing assignments file: "
            f"{assignments_path}"
        )

    if not topics_path.exists():
        raise FileNotFoundError(
            f"Missing topics file: "
            f"{topics_path}"
        )

    assignments = pd.read_csv(
        assignments_path
    )

    topics = pd.read_csv(
        topics_path
    )

    if assignments.empty:
        raise RuntimeError(
            f"No topic assignments found "
            f"for {app_key}."
        )

    if topics.empty:
        raise RuntimeError(
            f"No topic definitions found "
            f"for {app_key}."
        )

    total_reviews = len(assignments)

    # -----------------------------------------------------
    # Inspect every topic
    # -----------------------------------------------------

    for _, topic in topics.iterrows():

        topic_id = int(
            topic["topic_id"]
        )

        topic_reviews = assignments[
            assignments["topic_id"]
            == topic_id
        ].copy()

        topic_size = len(
            topic_reviews
        )

        if topic_size == 0:
            continue

        # -------------------------------------------------
        # Rating distribution
        # -------------------------------------------------

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

        one_to_two_share = (
            (one_star + two_star)
            / topic_size
            * 100
        )

        # -------------------------------------------------
        # Topic strength
        # -------------------------------------------------

        strength_mean = (
            topic_reviews[
                "topic_strength"
            ].mean()
        )

        strength_median = (
            topic_reviews[
                "topic_strength"
            ].median()
        )

        strength_min = (
            topic_reviews[
                "topic_strength"
            ].min()
        )

        strength_max = (
            topic_reviews[
                "topic_strength"
            ].max()
        )

        # -------------------------------------------------
        # Select meaningful reviews
        # -------------------------------------------------
        #
        # Some NMF assignments can have extremely small
        # topic strength. Those reviews are not useful
        # for interpreting the topic.
        # -------------------------------------------------

        meaningful_reviews = topic_reviews[
            topic_reviews["topic_strength"]
            >= MIN_TOPIC_STRENGTH
        ].copy()

        # If the threshold removes everything,
        # fall back to the strongest available reviews.
        if meaningful_reviews.empty:

            meaningful_reviews = (
                topic_reviews.copy()
            )

        # -------------------------------------------------
        # Strongest examples
        # -------------------------------------------------

        strongest_reviews = (
            meaningful_reviews
            .sort_values(
                "topic_strength",
                ascending=False,
            )
            .head(REVIEWS_PER_TOPIC)
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        print("\n" + "-" * 90)

        print(
            f"TOPIC {topic_id}"
        )

        print(
            f"Keywords: "
            f"{topic['top_words']}"
        )

        print(
            f"Topic size: "
            f"{topic_size:,} "
            f"("
            f"{topic_size / total_reviews * 100:.2f}%"
            f")"
        )

        print(
            "Rating distribution: "
            f"1★={one_star:,}, "
            f"2★={two_star:,}, "
            f"3★={three_star:,}"
        )

        print(
            f"1–2★ share: "
            f"{one_to_two_share:.2f}%"
        )

        print(
            f"Topic strength: "
            f"mean={strength_mean:.4f}, "
            f"median={strength_median:.4f}, "
            f"min={strength_min:.4f}, "
            f"max={strength_max:.4f}"
        )

        print(
            "\nStrongest representative reviews:"
        )

        if strongest_reviews.empty:

            print(
                "  No reviews available."
            )

            continue

        for i, (_, review) in enumerate(
            strongest_reviews.iterrows(),
            start=1,
        ):

            text = str(
                review["normalized_text"]
            )

            # Keep terminal output readable.
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
                f"     Topic strength: "
                f"{review['topic_strength']:.4f}"
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key in APPS:

        inspect_app(
            app_key
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TOPIC QUALITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()