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

OUTPUT_PATH = (
    MODEL_DIR
    / "topic_mapping.csv"
)


# =========================================================
# PHONEPE TOPIC MAPPING
# =========================================================
#
# Decision meanings:
#
# KEEP:
#   Topic is sufficiently specific and actionable.
#
# MERGE:
#   Topic contains useful pain but overlaps with another
#   final category.
#
# NOISE:
#   Topic is too generic, mixed, or not sufficiently
#   actionable for final product-pain metrics.
# =========================================================

PHONEPE_TOPIC_MAP = {

    # -----------------------------------------------------
    # App reliability & access
    # -----------------------------------------------------

    2: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Topic is too generic and mixes not-good sentiment "
            "with unrelated complaints."
        ),
    },

    4: {
        "decision": "KEEP",
        "final_category": "App reliability & access",
        "reason": (
            "Clear repeated app-not-working and functional "
            "failure complaints."
        ),
    },

    15: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Topic is dominated by PhonePe-name repetition, "
            "generic sentiment, and mixed support complaints."
        ),
    },

    # -----------------------------------------------------
    # Transaction & payment issues
    # -----------------------------------------------------

    7: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Topic is highly mixed and includes app updates, "
            "generic PhonePe references, and unrelated text."
        ),
    },

    12: {
        "decision": "MERGE",
        "final_category": "Transaction & payment issues",
        "reason": (
            "Contains genuine payment and transaction failures, "
            "although some reviews also mention account/opening issues."
        ),
    },

    # -----------------------------------------------------
    # Cashback & rewards
    # -----------------------------------------------------

    14: {
        "decision": "KEEP",
        "final_category": "Cashback & rewards",
        "reason": (
            "Contains explicit cashback and reward complaints."
        ),
    },

    # -----------------------------------------------------
    # Customer support
    # -----------------------------------------------------

    8: {
        "decision": "KEEP",
        "final_category": "Customer support",
        "reason": (
            "Contains clear customer-service and unresolved "
            "support complaints."
        ),
    },

    # -----------------------------------------------------
    # Performance & speed
    # -----------------------------------------------------

    6: {
        "decision": "KEEP",
        "final_category": "Performance & speed",
        "reason": (
            "Clear repeated slow and loading complaints."
        ),
    },

    # -----------------------------------------------------
    # Updates & UI
    # -----------------------------------------------------

    11: {
        "decision": "KEEP",
        "final_category": "Updates & UI",
        "reason": (
            "Strong evidence of dissatisfaction with the new "
            "UI, navigation, dark mode removal, and update changes."
        ),
    },

    # -----------------------------------------------------
    # Generic / noise topics
    # -----------------------------------------------------

    1: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic bad-app sentiment."
        ),
    },

    3: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Broad generic language without a specific product pain."
        ),
    },

    5: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic worst-app sentiment."
        ),
    },

    9: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic experience language without sufficient "
            "product specificity."
        ),
    },

    10: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic app/UPI language with insufficiently "
            "specific product meaning."
        ),
    },

    13: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic okay/bad sentiment and mixed language."
        ),
    },
}


# =========================================================
# VALIDATION
# =========================================================

EXPECTED_TOPICS = set(
    range(1, 16)
)

VALID_DECISIONS = {
    "KEEP",
    "MERGE",
    "NOISE",
}


def validate_mapping() -> None:

    actual_topics = set(
        PHONEPE_TOPIC_MAP.keys()
    )

    missing = (
        EXPECTED_TOPICS
        - actual_topics
    )

    extra = (
        actual_topics
        - EXPECTED_TOPICS
    )

    if missing:

        raise ValueError(
            "PhonePe mapping missing topics: "
            f"{sorted(missing)}"
        )

    if extra:

        raise ValueError(
            "PhonePe mapping contains unexpected topics: "
            f"{sorted(extra)}"
        )

    for topic_id, mapping in (
        PHONEPE_TOPIC_MAP.items()
    ):

        decision = mapping[
            "decision"
        ]

        if decision not in VALID_DECISIONS:

            raise ValueError(
                f"Topic {topic_id}: invalid decision "
                f"{decision}"
            )

        if decision in {
            "KEEP",
            "MERGE",
        }:

            if not mapping[
                "final_category"
            ]:

                raise ValueError(
                    f"Topic {topic_id}: "
                    "final category is required."
                )


# =========================================================
# BUILD OUTPUT
# =========================================================

def build_mapping() -> pd.DataFrame:

    rows = []

    topics_path = (
        MODEL_DIR
        / "phonepe_topics.csv"
    )

    if not topics_path.exists():

        raise FileNotFoundError(
            f"PhonePe topics file not found:\n"
            f"{topics_path}"
        )

    topics = pd.read_csv(
        topics_path
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
            f"PhonePe topics file missing columns: "
            f"{sorted(missing)}"
        )

    for _, topic in (
        topics.iterrows()
    ):

        topic_id = int(
            topic["topic_id"]
        )

        mapping = PHONEPE_TOPIC_MAP[
            topic_id
        ]

        rows.append(
            {
                "app_key": "phonepe",
                "topic_id": topic_id,
                "top_words": topic[
                    "top_words"
                ],
                "decision": mapping[
                    "decision"
                ],
                "final_category": mapping[
                    "final_category"
                ],
                "reason": mapping[
                    "reason"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    mapping: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PHONEPE TOPIC MAPPING"
    )

    print(
        "=" * 90
    )

    for decision in [
        "KEEP",
        "MERGE",
        "NOISE",
    ]:

        print(
            f"\n{decision}:"
        )

        subset = mapping[
            mapping["decision"]
            == decision
        ]

        for _, row in (
            subset.sort_values(
                "topic_id"
            ).iterrows()
        ):

            category = (
                row["final_category"]
                if row["final_category"]
                else "-"
            )

            print(
                f"  Topic "
                f"{int(row['topic_id']):2d}: "
                f"{category}"
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    validate_mapping()

    mapping = build_mapping()

    mapping.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(
        mapping
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"Saved PhonePe mapping to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()