from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "validation"
)

INPUT_PATH = (
    ANALYSIS_DIR
    / "final_pain_category_assignments.csv"
)

OUTPUT_PATH = (
    VALIDATION_DIR
    / "complaint_filter_validation.csv"
)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

TOTAL_SAMPLES = 120

MAPPED_SAMPLES = 60
NON_MAPPED_SAMPLES = 60

RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

def load_assignments() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Assignments file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "normalized_text",
        "topic_id",
        "topic_strength",
        "decision",
        "final_category",
        "is_mapped_pain",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Assignments file is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# TEXT PREPARATION
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
# BUILD SAMPLE
# =========================================================

def build_validation_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["review_text"] = (
        df["normalized_text"]
        .apply(prepare_text)
    )

    # Remove empty reviews.
    df = df[
        df["review_text"]
        .str.len()
        > 0
    ].copy()

    mapped = df[
        df["is_mapped_pain"]
        == True
    ].copy()

    non_mapped = df[
        df["is_mapped_pain"]
        != True
    ].copy()

    # -----------------------------------------------------
    # Sample mapped pain reviews.
    #
    # We stratify across apps first so one app cannot
    # dominate the validation set.
    # -----------------------------------------------------

    mapped_parts = []

    per_app_mapped = (
        MAPPED_SAMPLES
        // len(APPS)
    )

    for app_key in APPS:

        app_df = mapped[
            mapped["app_key"]
            == app_key
        ].copy()

        sample_size = min(
            per_app_mapped,
            len(app_df),
        )

        if sample_size > 0:

            sampled = app_df.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            )

            mapped_parts.append(
                sampled
            )

    mapped_sample = pd.concat(
        mapped_parts,
        ignore_index=True,
    )

    # Fill remaining slots if division left
    # unused places.
    remaining = (
        MAPPED_SAMPLES
        - len(mapped_sample)
    )

    if remaining > 0:

        remaining_pool = mapped[
            ~mapped["reviewId"].isin(
                mapped_sample["reviewId"]
            )
        ]

        if len(remaining_pool) > 0:

            extra = remaining_pool.sample(
                n=min(
                    remaining,
                    len(remaining_pool),
                ),
                random_state=RANDOM_STATE + 1,
            )

            mapped_sample = pd.concat(
                [
                    mapped_sample,
                    extra,
                ],
                ignore_index=True,
            )

    # -----------------------------------------------------
    # Sample non-mapped controls.
    # -----------------------------------------------------

    non_mapped_sample = non_mapped.sample(
        n=min(
            NON_MAPPED_SAMPLES,
            len(non_mapped),
        ),
        random_state=RANDOM_STATE + 2,
    )

    # -----------------------------------------------------
    # Combine.
    # -----------------------------------------------------

    mapped_sample = mapped_sample.copy()

    mapped_sample[
        "sampling_group"
    ] = "mapped_pain"

    non_mapped_sample = (
        non_mapped_sample.copy()
    )

    non_mapped_sample[
        "sampling_group"
    ] = "non_mapped_control"

    sample = pd.concat(
        [
            mapped_sample,
            non_mapped_sample,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Shuffle for blind-ish manual labeling.
    # -----------------------------------------------------

    sample = sample.sample(
        frac=1.0,
        random_state=RANDOM_STATE + 3,
    ).reset_index(
        drop=True
    )

    # Create an explicit validation ID.
    sample.insert(
        0,
        "validation_id",
        range(
            1,
            len(sample) + 1,
        ),
    )

    # -----------------------------------------------------
    # Human label fields.
    #
    # These are intentionally blank.
    # -----------------------------------------------------

    sample[
        "human_label"
    ] = ""

    sample[
        "label_reason"
    ] = ""

    # -----------------------------------------------------
    # Keep only columns needed for manual review.
    # -----------------------------------------------------

    output = sample[
        [
            "validation_id",
            "reviewId",
            "app_key",
            "score",
            "sampling_group",
            "final_category",
            "topic_id",
            "topic_strength",
            "review_text",
            "human_label",
            "label_reason",
        ]
    ].copy()

    return output


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    sample: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "COMPLAINT FILTER VALIDATION SAMPLE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nTotal validation reviews: "
        f"{len(sample):,}"
    )

    print(
        "\nSampling groups:"
    )

    print(
        sample[
            "sampling_group"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nApps:"
    )

    print(
        sample[
            "app_key"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nStar ratings:"
    )

    print(
        sample[
            "score"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LABEL DEFINITIONS"
    )

    print(
        "=" * 90
    )

    print(
        "\nUse human_label = complaint when the review "
        "describes a concrete problem, failure, unmet "
        "expectation, harmful outcome, or actionable "
        "product pain."
    )

    print(
        "\nUse human_label = non_complaint when the review "
        "is positive, generic praise, meaningless text, "
        "or does not actually describe product pain."
    )

    print(
        "\nUse human_label = ambiguous when the meaning "
        "cannot be determined reliably from the review."
    )

    print(
        "\nThe label_reason column is optional but useful "
        "for explaining difficult decisions."
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_assignments()

    sample = build_validation_sample(
        df
    )

    if sample.empty:

        raise RuntimeError(
            "Validation sample is empty."
        )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        sample
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "VALIDATION FILE CREATED"
    )

    print(
        "=" * 90
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\nOpen this CSV in Excel and fill only:"
    )

    print(
        "  human_label"
    )

    print(
        "  label_reason"
    )

    print(
        "\nAllowed human_label values:"
    )

    print(
        "  complaint"
    )

    print(
        "  non_complaint"
    )

    print(
        "  ambiguous"
    )

    print(
        "\nAfter labeling, we will measure the filter's "
        "precision/recall before changing the final metrics."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()