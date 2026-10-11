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

REVIEWS_PER_TOPIC = 5


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

    print("\n" + "=" * 80)
    print(f"TOPIC INSPECTION: {app_key.upper()}")
    print("=" * 80)

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

    for _, topic_row in topics.iterrows():

        topic_id = int(
            topic_row["topic_id"]
        )

        top_words = topic_row["top_words"]

        topic_reviews = assignments[
            assignments["topic_id"]
            == topic_id
        ].copy()

        # Strongest reviews for this topic.
        topic_reviews = topic_reviews.sort_values(
            "topic_strength",
            ascending=False,
        ).head(REVIEWS_PER_TOPIC)

        print("\n" + "-" * 80)

        print(
            f"TOPIC {topic_id}"
        )

        print(
            f"Keywords: {top_words}"
        )

        print(
            f"Reviews in topic: "
            f"{len(assignments[assignments['topic_id'] == topic_id]):,}"
        )

        print("\nRepresentative reviews:")

        if topic_reviews.empty:

            print("  No reviews found.")

            continue

        for i, (_, review) in enumerate(
            topic_reviews.iterrows(),
            start=1,
        ):

            text = str(
                review["normalized_text"]
            )

            # Keep terminal output readable.
            if len(text) > 350:
                text = text[:350] + "..."

            print(
                f"\n  {i}. "
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
        inspect_app(app_key)

    print("\n" + "=" * 80)
    print("TOPIC INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()