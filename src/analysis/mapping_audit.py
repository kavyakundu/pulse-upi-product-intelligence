from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

REFINED_DIR = (
    MODEL_DIR
    / "refined"
)


# =========================================================
# SETTINGS
# =========================================================

TARGETS = {
    "phonepe": [
        "App reliability & access",
        "Transaction & payment issues",
        "Cashback & rewards",
    ],
    "paytm": [
        "Customer support",
        "Transaction & payment issues",
        "Account & login issues",
    ],
    "googlepay": [
        "AutoPay & payment controls",
        "Transaction & payment issues",
        "Customer support",
    ],
}

REVIEWS_PER_TOPIC = 8


# =========================================================
# FILE CONFIGURATION
# =========================================================

APP_CONFIG = {
    "phonepe": {
        "assignments": (
            MODEL_DIR
            / "phonepe_topic_assignments.csv"
        ),
        "mapping": (
            MODEL_DIR
            / "topic_mapping.csv"
        ),
    },
    "paytm": {
        "assignments": (
            REFINED_DIR
            / "paytm_refined_assignments.csv"
        ),
        "mapping": (
            REFINED_DIR
            / "refined_topic_mapping.csv"
        ),
    },
    "googlepay": {
        "assignments": (
            REFINED_DIR
            / "googlepay_refined_assignments.csv"
        ),
        "mapping": (
            REFINED_DIR
            / "refined_topic_mapping.csv"
        ),
    },
}


# =========================================================
# LOAD DATA
# =========================================================

def load_assignments(
    app_key: str,
) -> pd.DataFrame:

    path = APP_CONFIG[
        app_key
    ]["assignments"]

    if not path.exists():

        raise FileNotFoundError(
            f"Assignments file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "reviewId",
        "app_key",
        "score",
        "normalized_text",
        "topic_id",
        "topic_strength",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{app_key}: assignments missing "
            f"columns: {sorted(missing)}"
        )

    return df


def load_mapping(
    app_key: str,
) -> pd.DataFrame:

    path = APP_CONFIG[
        app_key
    ]["mapping"]

    if not path.exists():

        raise FileNotFoundError(
            f"Mapping file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "app_key",
        "topic_id",
        "decision",
        "final_category",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{app_key}: mapping missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# PREPARE TEXT
# =========================================================

def prepare_text(
    text: str,
) -> str:

    if pd.isna(text):

        return ""

    return " ".join(
        str(text).split()
    )


# =========================================================
# PRINT TOPIC REVIEW SUMMARY
# =========================================================

def print_topic_summary(
    app_key: str,
    topic_id: int,
    assignments: pd.DataFrame,
    topic_words: str | None = None,
) -> None:

    topic_df = assignments[
        assignments["topic_id"]
        == topic_id
    ].copy()

    if topic_df.empty:

        print(
            f"\n  Topic {topic_id}: "
            f"NO REVIEWS FOUND"
        )

        return

    topic_df["display_text"] = (
        topic_df["normalized_text"]
        .apply(prepare_text)
    )

    topic_df["word_count"] = (
        topic_df["display_text"]
        .str.split()
        .str.len()
    )

    topic_df["one_two_star"] = (
        topic_df["score"]
        .isin([1, 2])
    )

    one_two_share = (
        topic_df["one_two_star"].mean()
        * 100
    )

    mean_strength = (
        topic_df["topic_strength"]
        .mean()
    )

    median_strength = (
        topic_df["topic_strength"]
        .median()
    )

    strong_share = (
        (
            topic_df["topic_strength"]
            >= 0.05
        ).mean()
        * 100
    )

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"Topic {topic_id}"
    )

    if topic_words:

        print(
            f"Keywords: {topic_words}"
        )

    print(
        f"Reviews: {len(topic_df):,}"
    )

    print(
        f"1–2★ share: "
        f"{one_two_share:.2f}%"
    )

    print(
        f"Mean strength: "
        f"{mean_strength:.4f}"
    )

    print(
        f"Median strength: "
        f"{median_strength:.4f}"
    )

    print(
        f"Strong assignment coverage: "
        f"{strong_share:.2f}%"
    )

    # -----------------------------------------------------
    # Select informative examples.
    # -----------------------------------------------------
    #
    # We want actual explanatory reviews rather than
    # generic "good/bad" reviews.
    # -----------------------------------------------------

    informative = topic_df[
        topic_df["word_count"]
        >= 6
    ].copy()

    if len(informative) < REVIEWS_PER_TOPIC:

        informative = topic_df.copy()

    informative = informative.sort_values(
        [
            "topic_strength",
            "word_count",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print(
        "\nRepresentative reviews:"
    )

    shown = 0

    for _, row in informative.iterrows():

        text = row[
            "display_text"
        ]

        if not text:

            continue

        if len(text) > 500:

            text = (
                text[:500]
                + "..."
            )

        print(
            f"\n  {shown + 1}. "
            f"[{int(row['score'])}★] "
            f"{text}"
        )

        print(
            f"     strength="
            f"{float(row['topic_strength']):.4f}"
        )

        shown += 1

        if shown >= REVIEWS_PER_TOPIC:

            break


# =========================================================
# AUDIT ONE APP
# =========================================================

def audit_app(
    app_key: str,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"MAPPING AUDIT: "
        f"{app_key.upper()}"
    )

    print(
        "=" * 100
    )

    assignments = load_assignments(
        app_key
    )

    mapping = load_mapping(
        app_key
    )

    target_categories = TARGETS[
        app_key
    ]

    for category in target_categories:

        print(
            "\n"
            + "#" * 100
        )

        print(
            f"CATEGORY: {category}"
        )

        category_mapping = mapping[
            (
                mapping["app_key"]
                == app_key
            )
            &
            (
                mapping["final_category"]
                == category
            )
            &
            (
                mapping["decision"].isin(
                    [
                        "KEEP",
                        "MERGE",
                    ]
                )
            )
        ].copy()

        if category_mapping.empty:

            print(
                "No topics currently mapped "
                "to this category."
            )

            continue

        for _, map_row in (
            category_mapping
            .sort_values("topic_id")
            .iterrows()
        ):

            topic_id = int(
                map_row["topic_id"]
            )

            # Find keywords if available.
            if app_key == "phonepe":

                topics_path = (
                    MODEL_DIR
                    / "phonepe_topics.csv"
                )

            else:

                topics_path = (
                    REFINED_DIR
                    / f"{app_key}_refined_topics.csv"
                )

            topic_words = None

            if topics_path.exists():

                topics = pd.read_csv(
                    topics_path
                )

                topic_match = topics[
                    topics["topic_id"]
                    == topic_id
                ]

                if not topic_match.empty:

                    topic_words = str(
                        topic_match.iloc[0][
                            "top_words"
                        ]
                    )

            print_topic_summary(
                app_key=app_key,
                topic_id=topic_id,
                assignments=assignments,
                topic_words=topic_words,
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TARGETED TOPIC-TO-CATEGORY AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "\nThis audit does NOT modify any files."
    )

    print(
        "It only inspects the topics currently feeding "
        "the categories we flagged."
    )

    for app_key in [
        "phonepe",
        "paytm",
        "googlepay",
    ]:

        audit_app(
            app_key
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MAPPING AUDIT COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()