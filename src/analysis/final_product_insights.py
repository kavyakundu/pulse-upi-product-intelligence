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

OUTPUT_PATH = (
    ANALYSIS_DIR
    / "final_product_insights.csv"
)

SHARED_OUTPUT_PATH = (
    ANALYSIS_DIR
    / "shared_product_pains.csv"
)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]


# =========================================================
# LOAD SCORES
# =========================================================

def load_scores() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Product pain scores not found:\n"
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
        "prevalence_pct",
        "severity_1_2_within_category_pct",
        "strong_assignment_share_pct",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Product pain score file is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# CREATE APP-LEVEL INSIGHTS
# =========================================================

def create_app_insights(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for app_key in APPS:

        app_df = scores[
            scores["app_key"]
            == app_key
        ].copy()

        if app_df.empty:
            continue

        app_df = app_df.sort_values(
            "product_pain_score",
            ascending=False,
        )

        for rank, (_, row) in enumerate(
            app_df.iterrows(),
            start=1,
        ):

            score = float(
                row["product_pain_score"]
            )

            prevalence = float(
                row["prevalence_pct"]
            )

            severity = float(
                row[
                    "severity_1_2_within_category_pct"
                ]
            )

            confidence = float(
                row[
                    "strong_assignment_share_pct"
                ]
            )

            # -------------------------------------------------
            # Descriptive insight type
            # -------------------------------------------------

            if (
                prevalence >= 10
                and severity >= 85
            ):

                insight_type = (
                    "High-frequency severe pain"
                )

            elif (
                prevalence < 5
                and severity >= 90
                and confidence >= 65
            ):

                insight_type = (
                    "Lower-frequency high-severity pain"
                )

            elif (
                prevalence >= 5
                and confidence >= 60
            ):

                insight_type = (
                    "Meaningful recurring pain"
                )

            else:

                insight_type = (
                    "Secondary pain signal"
                )

            rows.append(
                {
                    "app_key": app_key,
                    "rank_within_app": rank,
                    "category": row[
                        "category"
                    ],
                    "product_pain_score": score,
                    "priority": row[
                        "priority"
                    ],
                    "review_count": int(
                        row["review_count"]
                    ),
                    "prevalence_pct": prevalence,
                    "severity_1_2_pct": severity,
                    "model_confidence_pct": confidence,
                    "insight_type": insight_type,
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# CREATE SHARED PAIN TABLE
# =========================================================
#
# A pain is considered "shared" when the category exists
# for at least 2 apps.
#
# We calculate this directly from product_pain_scores.csv
# instead of relying on the separate comparison file.
# =========================================================

def create_shared_pains(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = scores.groupby(
        "category"
    )

    for category, category_df in grouped:

        apps_present = (
            category_df[
                "app_key"
            ]
            .nunique()
        )

        if apps_present < 2:
            continue

        app_scores = {}

        for _, row in category_df.iterrows():

            app_scores[
                row["app_key"]
            ] = float(
                row["product_pain_score"]
            )

        highest_app = max(
            app_scores,
            key=app_scores.get,
        )

        lowest_app = min(
            app_scores,
            key=app_scores.get,
        )

        cross_app_average = sum(
            app_scores.values()
        ) / len(
            app_scores
        )

        rows.append(
            {
                "category": category,
                "apps_represented": apps_present,
                "phonepe_score": app_scores.get(
                    "phonepe"
                ),
                "paytm_score": app_scores.get(
                    "paytm"
                ),
                "googlepay_score": app_scores.get(
                    "googlepay"
                ),
                "cross_app_average": (
                    cross_app_average
                ),
                "highest_pain_app": highest_app,
                "lowest_pain_app": lowest_app,
            }
        )

    shared = pd.DataFrame(
        rows
    )

    if shared.empty:
        return shared

    shared = shared.sort_values(
        "cross_app_average",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return shared


# =========================================================
# CREATE APP-SPECIFIC PAINS
# =========================================================
#
# These are categories that appear in only one app.
# =========================================================

def create_app_specific_pains(
    scores: pd.DataFrame,
) -> pd.DataFrame:

    category_app_counts = (
        scores.groupby(
            "category"
        )[
            "app_key"
        ]
        .nunique()
        .rename(
            "apps_represented"
        )
        .reset_index()
    )

    merged = scores.merge(
        category_app_counts,
        on="category",
        how="left",
        validate="many_to_one",
    )

    specific = merged[
        merged["apps_represented"]
        == 1
    ].copy()

    specific = specific.sort_values(
        "product_pain_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return specific[
        [
            "app_key",
            "category",
            "product_pain_score",
            "priority",
            "review_count",
            "prevalence_pct",
            "severity_1_2_within_category_pct",
            "strong_assignment_share_pct",
            "apps_represented",
        ]
    ]


# =========================================================
# PRINT APP INSIGHTS
# =========================================================

def print_app_insights(
    insights: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "APP-LEVEL PRODUCT INSIGHTS"
    )

    print(
        "=" * 90
    )

    for app_key in APPS:

        app_df = insights[
            insights["app_key"]
            == app_key
        ].sort_values(
            "rank_within_app"
        )

        print(
            f"\n{app_key.upper()}"
        )

        for _, row in (
            app_df.iterrows()
        ):

            print(
                f"\n  #{int(row['rank_within_app'])} "
                f"{row['category']}"
            )

            print(
                f"    Score: "
                f"{row['product_pain_score']:.2f}"
            )

            print(
                f"    Priority: "
                f"{row['priority']}"
            )

            print(
                f"    Type: "
                f"{row['insight_type']}"
            )

            print(
                f"    Reviews: "
                f"{int(row['review_count']):,}"
            )

            print(
                f"    Prevalence: "
                f"{row['prevalence_pct']:.2f}%"
            )

            print(
                f"    Severity: "
                f"{row['severity_1_2_pct']:.2f}%"
            )

            print(
                f"    Model confidence: "
                f"{row['model_confidence_pct']:.2f}%"
            )


# =========================================================
# PRINT SHARED PAINS
# =========================================================

def print_shared_pains(
    shared: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SHARED PRODUCT PAINS"
    )

    print(
        "=" * 90
    )

    if shared.empty:

        print(
            "No shared categories found."
        )

        return

    for _, row in shared.iterrows():

        print(
            f"\n  {row['category']}"
        )

        print(
            f"    Apps represented: "
            f"{int(row['apps_represented'])}"
        )

        if pd.notna(
            row["phonepe_score"]
        ):

            print(
                f"    PhonePe: "
                f"{row['phonepe_score']:.2f}"
            )

        if pd.notna(
            row["paytm_score"]
        ):

            print(
                f"    Paytm: "
                f"{row['paytm_score']:.2f}"
            )

        if pd.notna(
            row["googlepay_score"]
        ):

            print(
                f"    Google Pay: "
                f"{row['googlepay_score']:.2f}"
            )

        print(
            f"    Cross-app average: "
            f"{row['cross_app_average']:.2f}"
        )

        print(
            f"    Highest pain: "
            f"{row['highest_pain_app']}"
        )

        print(
            f"    Lowest pain: "
            f"{row['lowest_pain_app']}"
        )


# =========================================================
# PRINT APP-SPECIFIC PAINS
# =========================================================

def print_app_specific_pains(
    specific: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "APP-SPECIFIC PRODUCT PAINS"
    )

    print(
        "=" * 90
    )

    if specific.empty:

        print(
            "No app-specific categories found."
        )

        return

    for _, row in specific.iterrows():

        print(
            f"\n  {row['app_key'].upper()} "
            f"| {row['category']}"
        )

        print(
            f"    Score: "
            f"{row['product_pain_score']:.2f}"
        )

        print(
            f"    Reviews: "
            f"{int(row['review_count']):,}"
        )

        print(
            f"    Prevalence: "
            f"{row['prevalence_pct']:.2f}%"
        )

        print(
            f"    Severity: "
            f"{row['severity_1_2_within_category_pct']:.2f}%"
        )

        print(
            f"    Confidence: "
            f"{row['strong_assignment_share_pct']:.2f}%"
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
        "FINAL PRODUCT INSIGHTS"
    )

    print(
        "=" * 90
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    scores = load_scores()

    print(
        f"\nLoaded pain categories: "
        f"{len(scores):,}"
    )

    # -----------------------------------------------------
    # Build analyses
    # -----------------------------------------------------

    insights = create_app_insights(
        scores
    )

    shared = create_shared_pains(
        scores
    )

    app_specific = (
        create_app_specific_pains(
            scores
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    insights.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    shared.to_csv(
        SHARED_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print_app_insights(
        insights
    )

    print_shared_pains(
        shared
    )

    print_app_specific_pains(
        app_specific
    )

    # -----------------------------------------------------
    # Final confirmation
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
        f"Final product insights:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nShared product pains:\n"
        f"{SHARED_OUTPUT_PATH}"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL PRODUCT INSIGHT ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()