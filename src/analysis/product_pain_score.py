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

INPUT_PATH = (
    ANALYSIS_DIR
    / "final_pain_category_metrics.csv"
)

OUTPUT_PATH = (
    ANALYSIS_DIR
    / "product_pain_scores.csv"
)

APP_SUMMARY_PATH = (
    ANALYSIS_DIR
    / "product_pain_rankings.csv"
)


# =========================================================
# PRODUCT PAIN SCORE
# =========================================================
#
# Project-defined heuristic:
#
#   50% = prevalence
#   30% = severity
#   20% = topic-assignment confidence
#
# Prevalence:
#   Share of classifier-predicted complaints that belong
#   to the category.
#
# Severity:
#   Share of category reviews rated 1–2★.
#
# Topic-assignment confidence:
#   Share of category reviews with topic strength >= 0.05.
#
# IMPORTANT:
# This is a project-defined prioritization heuristic,
# not a statistically validated industry standard.
# =========================================================

PREVALENCE_WEIGHT = 0.50
SEVERITY_WEIGHT = 0.30
CONFIDENCE_WEIGHT = 0.20

STRONG_TOPIC_THRESHOLD = 0.05


# =========================================================
# LOAD METRICS
# =========================================================

def load_metrics() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Final pain metrics file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "app_key",
        "category",
        "classifier_complaint_count",
        "mapped_pain_total",
        "review_count",
        "prevalence_among_classifier_complaints_pct",
        "share_of_mapped_pain_pct",
        "severity_1_2_within_category_pct",
        "severity_1_star_within_category_pct",
        "mean_topic_strength",
        "median_topic_strength",
        "strong_assignment_share_pct",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Final pain metrics file is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# VALIDATE INPUT DATA
# =========================================================

def validate_metrics(
    df: pd.DataFrame,
) -> None:

    if df.empty:

        raise ValueError(
            "Final pain metrics file is empty."
        )

    duplicate_categories = (
        df[
            [
                "app_key",
                "category",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_categories > 0:

        raise ValueError(
            "Duplicate app/category rows found: "
            f"{duplicate_categories}"
        )

    numeric_columns = [
        "classifier_complaint_count",
        "mapped_pain_total",
        "review_count",
        "prevalence_among_classifier_complaints_pct",
        "share_of_mapped_pain_pct",
        "severity_1_2_within_category_pct",
        "severity_1_star_within_category_pct",
        "mean_topic_strength",
        "median_topic_strength",
        "strong_assignment_share_pct",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    percentage_columns = [
        "prevalence_among_classifier_complaints_pct",
        "share_of_mapped_pain_pct",
        "severity_1_2_within_category_pct",
        "severity_1_star_within_category_pct",
        "strong_assignment_share_pct",
    ]

    for column in percentage_columns:

        invalid = (
            (df[column] < 0)
            | (df[column] > 100)
        )

        if invalid.any():

            raise ValueError(
                f"Column '{column}' contains "
                "values outside 0–100."
            )

    invalid_counts = (
        df[
            [
                "classifier_complaint_count",
                "mapped_pain_total",
                "review_count",
            ]
        ]
        < 0
    ).any().any()

    if invalid_counts:

        raise ValueError(
            "Count columns contain negative values."
        )


# =========================================================
# VALIDATE WEIGHTS
# =========================================================

def validate_weights() -> None:

    total_weight = (
        PREVALENCE_WEIGHT
        + SEVERITY_WEIGHT
        + CONFIDENCE_WEIGHT
    )

    if abs(total_weight - 1.0) > 1e-9:

        raise ValueError(
            "Product Pain Score weights must sum to 1.0."
        )


# =========================================================
# CALCULATE PRODUCT PAIN SCORE
# =========================================================

def calculate_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # -----------------------------------------------------
    # Normalize components to 0–1.
    # -----------------------------------------------------

    prevalence = (
        result[
            "prevalence_among_classifier_complaints_pct"
        ]
        .clip(
            lower=0,
            upper=100,
        )
        / 100
    )

    severity = (
        result[
            "severity_1_2_within_category_pct"
        ]
        .clip(
            lower=0,
            upper=100,
        )
        / 100
    )

    topic_confidence = (
        result[
            "strong_assignment_share_pct"
        ]
        .clip(
            lower=0,
            upper=100,
        )
        / 100
    )

    # -----------------------------------------------------
    # Weighted components.
    # -----------------------------------------------------

    result["prevalence_component"] = (
        prevalence
        * PREVALENCE_WEIGHT
        * 100
    )

    result["severity_component"] = (
        severity
        * SEVERITY_WEIGHT
        * 100
    )

    result["confidence_component"] = (
        topic_confidence
        * CONFIDENCE_WEIGHT
        * 100
    )

    # -----------------------------------------------------
    # Final Product Pain Score.
    # -----------------------------------------------------

    result["product_pain_score"] = (
        result["prevalence_component"]
        + result["severity_component"]
        + result["confidence_component"]
    )

    # -----------------------------------------------------
    # Priority label.
    # -----------------------------------------------------

    def get_priority(
        score: float,
    ) -> str:

        if score >= 60:
            return "Very High"

        if score >= 45:
            return "High"

        if score >= 30:
            return "Medium"

        return "Lower"

    result["priority"] = (
        result["product_pain_score"]
        .apply(get_priority)
    )

    # -----------------------------------------------------
    # Rank within each app.
    # -----------------------------------------------------

    result["rank_within_app"] = (
        result
        .groupby("app_key")[
            "product_pain_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Rank across all apps.
    # -----------------------------------------------------

    result["overall_rank"] = (
        result[
            "product_pain_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return result


# =========================================================
# CREATE APP RANKINGS
# =========================================================

def create_app_rankings(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for app_key, app_df in (
        scores.groupby("app_key")
    ):

        app_df = app_df.sort_values(
            [
                "product_pain_score",
                "review_count",
            ],
            ascending=[
                False,
                False,
            ],
        )

        for position, (_, row) in enumerate(
            app_df.iterrows(),
            start=1,
        ):

            rows.append(
                {
                    "app_key": app_key,

                    "rank": position,

                    "category": row[
                        "category"
                    ],

                    "product_pain_score": row[
                        "product_pain_score"
                    ],

                    "priority": row[
                        "priority"
                    ],

                    "review_count": row[
                        "review_count"
                    ],

                    "classifier_complaint_count": row[
                        "classifier_complaint_count"
                    ],

                    "mapped_pain_total": row[
                        "mapped_pain_total"
                    ],

                    "prevalence_among_classifier_complaints_pct": row[
                        "prevalence_among_classifier_complaints_pct"
                    ],

                    "share_of_mapped_pain_pct": row[
                        "share_of_mapped_pain_pct"
                    ],

                    "severity_1_2_within_category_pct": row[
                        "severity_1_2_within_category_pct"
                    ],

                    "severity_1_star_within_category_pct": row[
                        "severity_1_star_within_category_pct"
                    ],

                    "strong_assignment_share_pct": row[
                        "strong_assignment_share_pct"
                    ],

                    "mean_topic_strength": row[
                        "mean_topic_strength"
                    ],

                    "median_topic_strength": row[
                        "median_topic_strength"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    scores: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PRODUCT PAIN SCORE"
    )

    print(
        "=" * 90
    )

    print(
        "\nFormula:"
    )

    print(
        "Score = "
        "50% prevalence + "
        "30% severity + "
        "20% topic-assignment confidence"
    )

    print(
        "\nPrevalence = share of classifier-predicted "
        "complaints assigned to the category."
    )

    print(
        "Severity = share of category reviews rated 1–2★."
    )

    print(
        "Topic-assignment confidence = share with "
        f"topic strength >= {STRONG_TOPIC_THRESHOLD:.2f}."
    )

    print(
        "\nNOTE: This is a "
        "project-defined heuristic, "
        "not a statistically validated formula."
    )

    # -----------------------------------------------------
    # Per-app rankings
    # -----------------------------------------------------

    for app_key in [
        "phonepe",
        "paytm",
        "googlepay",
    ]:

        print(
            "\n"
            + "-" * 90
        )

        print(
            app_key.upper()
        )

        print(
            "-" * 90
        )

        app_df = scores[
            scores["app_key"]
            == app_key
        ].copy()

        app_df = app_df.sort_values(
            "product_pain_score",
            ascending=False,
        )

        for _, row in app_df.iterrows():

            print(
                f"\n  #{int(row['rank_within_app'])} "
                f"{row['category']}"
            )

            print(
                f"      Product Pain Score: "
                f"{row['product_pain_score']:.2f}"
            )

            print(
                f"      Priority: "
                f"{row['priority']}"
            )

            print(
                f"      Reviews: "
                f"{int(row['review_count']):,}"
            )

            print(
                f"      Prevalence among classifier complaints: "
                f"{row['prevalence_among_classifier_complaints_pct']:.2f}%"
            )

            print(
                f"      Severity (1–2★): "
                f"{row['severity_1_2_within_category_pct']:.2f}%"
            )

            print(
                f"      Topic-assignment confidence: "
                f"{row['strong_assignment_share_pct']:.2f}%"
            )

            print(
                f"      Components: "
                f"prevalence="
                f"{row['prevalence_component']:.2f}, "
                f"severity="
                f"{row['severity_component']:.2f}, "
                f"confidence="
                f"{row['confidence_component']:.2f}"
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CALCULATING FINAL PRODUCT PAIN SCORES"
    )

    print(
        "=" * 90
    )

    validate_weights()

    metrics = load_metrics()

    validate_metrics(
        metrics
    )

    print(
        f"\nLoaded categories: "
        f"{len(metrics):,}"
    )

    print(
        f"Apps: "
        f"{metrics['app_key'].nunique()}"
    )

    print(
        f"Input file:\n"
        f"{INPUT_PATH}"
    )

    # -----------------------------------------------------
    # Calculate scores
    # -----------------------------------------------------

    scores = calculate_score(
        metrics
    )

    # -----------------------------------------------------
    # Create rankings
    # -----------------------------------------------------

    rankings = create_app_rankings(
        scores
    )

    # -----------------------------------------------------
    # Sort final detailed file
    # -----------------------------------------------------

    scores = scores.sort_values(
        [
            "app_key",
            "product_pain_score",
            "review_count",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    scores.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    rankings.to_csv(
        APP_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print_results(
        scores
    )

    # -----------------------------------------------------
    # Files saved
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 90
    )

    print(
        f"\nDetailed scores:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nRankings:\n"
        f"{APP_SUMMARY_PATH}"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL PRODUCT PAIN SCORE COMPLETE"
    )

    print(
        "=" * 90
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()