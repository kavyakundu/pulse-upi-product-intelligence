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
    / "product_pain_scores.csv"
)

COMPARISON_OUTPUT = (
    ANALYSIS_DIR
    / "cross_app_comparison.csv"
)

SHARED_OUTPUT = (
    ANALYSIS_DIR
    / "shared_product_pains.csv"
)

APP_SPECIFIC_OUTPUT = (
    ANALYSIS_DIR
    / "app_specific_product_pains.csv"
)


# =========================================================
# APPS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]


# =========================================================
# LOAD DATA
# =========================================================

def load_scores() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Product Pain Scores file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "app_key",
        "category",
        "product_pain_score",
        "priority",
        "review_count",
        "prevalence_among_classifier_complaints_pct",
        "severity_1_2_within_category_pct",
        "strong_assignment_share_pct",
        "rank_within_app",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Product Pain Scores file is missing "
            f"columns: {sorted(missing)}"
        )

    # Numeric validation.
    numeric_columns = [
        "product_pain_score",
        "review_count",
        "prevalence_among_classifier_complaints_pct",
        "severity_1_2_within_category_pct",
        "strong_assignment_share_pct",
        "rank_within_app",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # Validate app coverage.
    missing_apps = (
        set(APPS)
        - set(df["app_key"].unique())
    )

    if missing_apps:

        raise ValueError(
            "Missing expected apps: "
            f"{sorted(missing_apps)}"
        )

    # Validate uniqueness.
    duplicate_count = (
        df[
            [
                "app_key",
                "category",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate app/category combinations found: "
            f"{duplicate_count}"
        )

    return df


# =========================================================
# FIND SHARED CATEGORIES
# =========================================================

def find_shared_categories(
    scores: pd.DataFrame,
) -> list[str]:

    category_sets = []

    for app_key in APPS:

        categories = set(
            scores.loc[
                scores["app_key"] == app_key,
                "category",
            ]
            .dropna()
            .tolist()
        )

        category_sets.append(
            categories
        )

    shared = set.intersection(
        *category_sets
    )

    return sorted(
        shared
    )


# =========================================================
# BUILD SHARED COMPARISON
# =========================================================

def build_shared_comparison(
    scores: pd.DataFrame,
    shared_categories: list[str],
) -> pd.DataFrame:

    rows = []

    for category in shared_categories:

        category_df = scores[
            scores["category"]
            == category
        ].copy()

        row = {
            "category": category,
        }

        score_values = []

        for app_key in APPS:

            app_row = category_df[
                category_df["app_key"]
                == app_key
            ]

            if app_row.empty:

                raise ValueError(
                    f"Expected shared category "
                    f"'{category}' missing for "
                    f"{app_key}."
                )

            app_row = app_row.iloc[0]

            row[
                f"{app_key}_product_pain_score"
            ] = app_row[
                "product_pain_score"
            ]

            row[
                f"{app_key}_priority"
            ] = app_row[
                "priority"
            ]

            row[
                f"{app_key}_review_count"
            ] = app_row[
                "review_count"
            ]

            row[
                f"{app_key}_prevalence_pct"
            ] = app_row[
                "prevalence_among_classifier_complaints_pct"
            ]

            row[
                f"{app_key}_severity_pct"
            ] = app_row[
                "severity_1_2_within_category_pct"
            ]

            row[
                f"{app_key}_topic_confidence_pct"
            ] = app_row[
                "strong_assignment_share_pct"
            ]

            score_values.append(
                app_row[
                    "product_pain_score"
                ]
            )

        # Average Product Pain Score across apps.
        row[
            "average_product_pain_score"
        ] = sum(score_values) / len(
            score_values
        )

        # Highest-scoring app.
        highest_index = score_values.index(
            max(score_values)
        )

        highest_app = APPS[
            highest_index
        ]

        row[
            "highest_pain_app"
        ] = highest_app

        row[
            "highest_pain_score"
        ] = max(
            score_values
        )

        # Score spread.
        row[
            "score_spread"
        ] = (
            max(score_values)
            - min(score_values)
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = result.sort_values(
            [
                "average_product_pain_score",
                "highest_pain_score",
            ],
            ascending=[
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )

    return result


# =========================================================
# BUILD APP-SPECIFIC COMPARISON
# =========================================================

def build_app_specific(
    scores: pd.DataFrame,
    shared_categories: list[str],
) -> pd.DataFrame:

    shared_set = set(
        shared_categories
    )

    rows = []

    for app_key in APPS:

        app_df = scores[
            scores["app_key"]
            == app_key
        ].copy()

        app_specific = app_df[
            ~app_df["category"].isin(
                shared_set
            )
        ].copy()

        app_specific = app_specific.sort_values(
            "product_pain_score",
            ascending=False,
        )

        for _, row in (
            app_specific.iterrows()
        ):

            rows.append(
                {
                    "app_key": app_key,

                    "category": row[
                        "category"
                    ],

                    "product_pain_score": row[
                        "product_pain_score"
                    ],

                    "priority": row[
                        "priority"
                    ],

                    "rank_within_app": row[
                        "rank_within_app"
                    ],

                    "review_count": row[
                        "review_count"
                    ],

                    "prevalence_pct": row[
                        "prevalence_among_classifier_complaints_pct"
                    ],

                    "severity_1_2_pct": row[
                        "severity_1_2_within_category_pct"
                    ],

                    "topic_confidence_pct": row[
                        "strong_assignment_share_pct"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# BUILD LONG-FORM COMPARISON
# =========================================================

def build_long_comparison(
    scores: pd.DataFrame,
    shared_categories: list[str],
) -> pd.DataFrame:

    result = scores.copy()

    shared_set = set(
        shared_categories
    )

    result["comparison_type"] = (
        result["category"]
        .apply(
            lambda category:
            "shared"
            if category in shared_set
            else "app_specific"
        )
    )

    result["app_name"] = (
        result["app_key"]
        .map(
            {
                "phonepe": "PhonePe",
                "paytm": "Paytm",
                "googlepay": "Google Pay",
            }
        )
    )

    result = result[
        [
            "app_key",
            "app_name",
            "category",
            "comparison_type",
            "rank_within_app",
            "product_pain_score",
            "priority",
            "review_count",
            "prevalence_among_classifier_complaints_pct",
            "severity_1_2_within_category_pct",
            "strong_assignment_share_pct",
            "mean_topic_strength",
            "median_topic_strength",
        ]
    ].copy()

    result = result.sort_values(
        [
            "comparison_type",
            "category",
            "product_pain_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return result


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(
    shared: pd.DataFrame,
    app_specific: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 95
    )

    print(
        "FINAL CROSS-APP PRODUCT PAIN COMPARISON"
    )

    print(
        "=" * 95
    )

    print(
        "\nShared categories across all three apps:"
    )

    print(
        f"  {len(shared):,}"
    )

    if not shared.empty:

        print(
            "\n"
            + "-" * 95
        )

        print(
            "SHARED PRODUCT PAINS"
        )

        print(
            "-" * 95
        )

        for _, row in (
            shared.iterrows()
        ):

            print(
                f"\n  {row['category']}"
            )

            print(
                f"      Average Product Pain Score: "
                f"{row['average_product_pain_score']:.2f}"
            )

            print(
                f"      Highest: "
                f"{row['highest_pain_app']} "
                f"({row['highest_pain_score']:.2f})"
            )

            print(
                f"      Score spread: "
                f"{row['score_spread']:.2f}"
            )

            for app_key in APPS:

                print(
                    f"      {app_key}: "
                    f"{row[f'{app_key}_product_pain_score']:.2f}"
                )

    if not app_specific.empty:

        print(
            "\n"
            + "-" * 95
        )

        print(
            "APP-SPECIFIC MAPPED PAINS"
        )

        print(
            "-" * 95
        )

        for app_key in APPS:

            print(
                f"\n  {app_key.upper()}"
            )

            app_df = app_specific[
                app_specific["app_key"]
                == app_key
            ]

            for _, row in (
                app_df.iterrows()
            ):

                print(
                    f"      "
                    f"{row['category']}: "
                    f"{row['product_pain_score']:.2f} "
                    f"({row['priority']})"
                )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n"
        + "=" * 95
    )

    print(
        "BUILDING FINAL CROSS-APP COMPARISON"
    )

    print(
        "=" * 95
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    scores = load_scores()

    print(
        f"\nLoaded Product Pain Score rows: "
        f"{len(scores):,}"
    )

    print(
        f"Apps: "
        f"{scores['app_key'].nunique()}"
    )

    # -----------------------------------------------------
    # Shared categories
    # -----------------------------------------------------

    shared_categories = find_shared_categories(
        scores
    )

    print(
        f"Shared categories: "
        f"{len(shared_categories):,}"
    )

    # -----------------------------------------------------
    # Build outputs
    # -----------------------------------------------------

    shared = build_shared_comparison(
        scores=scores,
        shared_categories=shared_categories,
    )

    app_specific = build_app_specific(
        scores=scores,
        shared_categories=shared_categories,
    )

    comparison = build_long_comparison(
        scores=scores,
        shared_categories=shared_categories,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    shared.to_csv(
        SHARED_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    app_specific.to_csv(
        APP_SPECIFIC_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print_results(
        shared=shared,
        app_specific=app_specific,
    )

    # -----------------------------------------------------
    # Confirmation
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 95
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 95
    )

    print(
        f"\nFull comparison:\n"
        f"{COMPARISON_OUTPUT}"
    )

    print(
        f"\nShared-category comparison:\n"
        f"{SHARED_OUTPUT}"
    )

    print(
        f"\nApp-specific pain comparison:\n"
        f"{APP_SPECIFIC_OUTPUT}"
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "FINAL CROSS-APP COMPARISON COMPLETE"
    )

    print(
        "=" * 95
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()