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

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INTEGRATED_ASSIGNMENTS_FILE = (
    ANALYSIS_DIR
    / "final_pain_category_assignments.csv"
)

CLASSIFIER_PREDICTIONS_FILE = (
    ANALYSIS_DIR
    / "complaint_classifier_predictions.csv"
)

OUTPUT_METRICS_FILE = (
    ANALYSIS_DIR
    / "final_pain_category_metrics.csv"
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
# CATEGORY ORDER
# =========================================================

CATEGORY_ORDER = [
    "App reliability & access",
    "Performance & speed",
    "Transaction & payment issues",
    "Updates & UI",
    "Cashback & rewards",
    "Customer support",
    "Device & security compatibility",
    "Account & login issues",
    "Fraud, trust & security",
    "AutoPay & payment controls",
    "Bank account & linking",
    "UPI & UPI Lite",
    "Network, server & OTP issues",
]


# =========================================================
# LOAD INTEGRATED DATA
# =========================================================

def load_integrated_assignments() -> pd.DataFrame:

    if not INTEGRATED_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            "Final integrated assignments file not found:\n"
            f"{INTEGRATED_ASSIGNMENTS_FILE}"
        )

    df = pd.read_csv(
        INTEGRATED_ASSIGNMENTS_FILE
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
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
            "Integrated assignments are missing "
            f"columns: {sorted(missing)}"
        )

    df["score"] = pd.to_numeric(
        df["score"],
        errors="raise",
    )

    df["topic_id"] = pd.to_numeric(
        df["topic_id"],
        errors="raise",
    ).astype(int)

    df["topic_strength"] = pd.to_numeric(
        df["topic_strength"],
        errors="raise",
    )

    df["is_mapped_pain"] = (
        df["is_mapped_pain"]
        .fillna(False)
        .astype(bool)
    )

    return df


# =========================================================
# LOAD CLASSIFIER PREDICTIONS
# =========================================================

def load_classifier_predictions() -> pd.DataFrame:

    if not CLASSIFIER_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Classifier predictions file not found:\n"
            f"{CLASSIFIER_PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        CLASSIFIER_PREDICTIONS_FILE
    )

    required_columns = {
        "reviewId",
        "app_key",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Classifier predictions are missing "
            f"columns: {sorted(missing)}"
        )

    duplicate_count = (
        df["reviewId"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Classifier predictions contain "
            f"{duplicate_count:,} duplicate review IDs."
        )

    return df


# =========================================================
# VALIDATE FINAL POPULATION
# =========================================================

def validate_final_population(
    integrated: pd.DataFrame,
    classifier: pd.DataFrame,
) -> None:

    invalid_scores = ~integrated[
        "score"
    ].isin(
        [
            1,
            2,
            3,
        ]
    )

    invalid_count = (
        invalid_scores.sum()
    )

    if invalid_count > 0:
        raise ValueError(
            "Integrated data contains "
            f"{invalid_count:,} rows outside the "
            "expected 1–3★ population."
        )

    duplicate_integrated = (
        integrated["reviewId"]
        .duplicated()
        .sum()
    )

    if duplicate_integrated > 0:
        raise ValueError(
            "Integrated assignments contain "
            f"{duplicate_integrated:,} duplicate review IDs."
        )

    classifier_ids = set(
        classifier["reviewId"]
    )

    integrated_ids = set(
        integrated["reviewId"]
    )

    outside_classifier = (
        integrated_ids
        - classifier_ids
    )

    if outside_classifier:
        raise ValueError(
            "Integrated assignments contain "
            f"{len(outside_classifier):,} reviews that are not "
            "present in classifier predictions."
        )


# =========================================================
# CALCULATE CATEGORY METRICS
# =========================================================

def calculate_metrics(
    integrated: pd.DataFrame,
    classifier: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    app_integrated = integrated[
        integrated["app_key"]
        == app_key
    ].copy()

    app_classifier = classifier[
        classifier["app_key"]
        == app_key
    ].copy()

    classifier_complaints = len(
        app_classifier
    )

    if classifier_complaints == 0:
        raise ValueError(
            f"{app_key}: no classifier complaints found."
        )

    mapped_pain = app_integrated[
        app_integrated["is_mapped_pain"]
    ].copy()

    if mapped_pain.empty:
        raise ValueError(
            f"{app_key}: no mapped pain reviews found."
        )

    total_one_two = (
        mapped_pain["score"]
        .isin(
            [
                1,
                2,
            ]
        )
        .sum()
    )

    total_one_star = (
        mapped_pain["score"]
        .eq(1)
        .sum()
    )

    rows = []

    for category, category_df in (
        mapped_pain.groupby(
            "final_category"
        )
    ):

        review_count = len(
            category_df
        )

        # -------------------------------------------------
        # Severity within category
        # -------------------------------------------------

        one_two_count = (
            category_df["score"]
            .isin(
                [
                    1,
                    2,
                ]
            )
            .sum()
        )

        one_star_count = (
            category_df["score"]
            .eq(1)
            .sum()
        )

        severity_1_2 = (
            one_two_count
            / review_count
            * 100
            if review_count
            else 0
        )

        severity_1_star = (
            one_star_count
            / review_count
            * 100
            if review_count
            else 0
        )

        # -------------------------------------------------
        # FINAL prevalence
        # -------------------------------------------------
        # Denominator = classifier-predicted complaints,
        # NOT all 1–3★ reviews.
        # -------------------------------------------------

        prevalence_pct = (
            review_count
            / classifier_complaints
            * 100
        )

        # -------------------------------------------------
        # Share of mapped-pain reviews
        # -------------------------------------------------

        mapped_pain_total = len(
            mapped_pain
        )

        share_of_mapped_pain_pct = (
            review_count
            / mapped_pain_total
            * 100
        )

        # -------------------------------------------------
        # Share of classifier complaints that are 1–2★
        # -------------------------------------------------

        classifier_one_two = (
            app_classifier
            .get("score")
        )

        if classifier_one_two is not None:
            classifier_one_two = pd.to_numeric(
                classifier_one_two,
                errors="coerce",
            )

            total_classifier_one_two = (
                classifier_one_two
                .isin(
                    [
                        1,
                        2,
                    ]
                )
                .sum()
            )

            total_classifier_one_star = (
                classifier_one_two
                .eq(1)
                .sum()
            )
        else:
            total_classifier_one_two = 0
            total_classifier_one_star = 0

        share_of_all_1_2_pct = (
            one_two_count
            / total_classifier_one_two
            * 100
            if total_classifier_one_two
            else 0
        )

        share_of_all_1_star_pct = (
            one_star_count
            / total_classifier_one_star
            * 100
            if total_classifier_one_star
            else 0
        )

        # -------------------------------------------------
        # Topic strength
        # -------------------------------------------------

        mean_strength = (
            category_df[
                "topic_strength"
            ].mean()
        )

        median_strength = (
            category_df[
                "topic_strength"
            ].median()
        )

        strong_count = (
            category_df[
                "topic_strength"
            ]
            >= 0.05
        ).sum()

        strong_share = (
            strong_count
            / review_count
            * 100
        )

        # -------------------------------------------------
        # Average rating
        # -------------------------------------------------

        average_rating = (
            category_df["score"]
            .mean()
        )

        rows.append(
            {
                "app_key": app_key,

                "category": category,

                "classifier_complaint_count": (
                    classifier_complaints
                ),

                "mapped_pain_total": (
                    mapped_pain_total
                ),

                "review_count": (
                    review_count
                ),

                "prevalence_among_classifier_complaints_pct": (
                    prevalence_pct
                ),

                "share_of_mapped_pain_pct": (
                    share_of_mapped_pain_pct
                ),

                "one_two_star_count": (
                    one_two_count
                ),

                "severity_1_2_within_category_pct": (
                    severity_1_2
                ),

                "one_star_count": (
                    one_star_count
                ),

                "severity_1_star_within_category_pct": (
                    severity_1_star
                ),

                "share_of_all_1_2_star_classifier_complaints_pct": (
                    share_of_all_1_2_pct
                ),

                "share_of_all_1_star_classifier_complaints_pct": (
                    share_of_all_1_star_pct
                ),

                "average_rating": (
                    average_rating
                ),

                "mean_topic_strength": (
                    mean_strength
                ),

                "median_topic_strength": (
                    median_strength
                ),

                "strong_assignment_share_pct": (
                    strong_share
                ),
            }
        )

    return pd.DataFrame(
        rows
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
        "FINAL POST-CLASSIFIER PAIN CATEGORY METRICS"
    )

    print(
        "=" * 90
    )

    print(
        "\nIntegrated source:"
    )

    print(
        INTEGRATED_ASSIGNMENTS_FILE
    )

    print(
        "\nClassifier source:"
    )

    print(
        CLASSIFIER_PREDICTIONS_FILE
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    integrated = (
        load_integrated_assignments()
    )

    classifier = (
        load_classifier_predictions()
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    validate_final_population(
        integrated=integrated,
        classifier=classifier,
    )

    print(
        "\nValidation:"
    )

    print(
        f"  Integrated reviews: "
        f"{len(integrated):,}"
    )

    print(
        f"  Classifier complaints: "
        f"{len(classifier):,}"
    )

    print(
        f"  Mapped pain reviews: "
        f"{int(integrated['is_mapped_pain'].sum()):,}"
    )

    # -----------------------------------------------------
    # Process apps
    # -----------------------------------------------------

    all_metrics = []

    for app_key in APPS:

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"PROCESSING: {app_key.upper()}"
        )

        app_integrated = integrated[
            integrated["app_key"]
            == app_key
        ]

        app_classifier = classifier[
            classifier["app_key"]
            == app_key
        ]

        complaint_count = len(
            app_classifier
        )

        topic_assigned_count = len(
            app_integrated
        )

        mapped_count = int(
            app_integrated[
                "is_mapped_pain"
            ].sum()
        )

        print(
            f"Classifier complaints: "
            f"{complaint_count:,}"
        )

        print(
            f"Topic-assigned complaints: "
            f"{topic_assigned_count:,}"
        )

        print(
            f"Mapped pain reviews: "
            f"{mapped_count:,}"
        )

        print(
            f"Mapping coverage: "
            f"{mapped_count / complaint_count * 100:.2f}%"
        )

        metrics = calculate_metrics(
            integrated=integrated,
            classifier=classifier,
            app_key=app_key,
        )

        all_metrics.append(
            metrics
        )

    # =====================================================
    # COMBINE
    # =====================================================

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # =====================================================
    # ORDER
    # =====================================================

    metrics_df["category"] = pd.Categorical(
        metrics_df["category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )

    metrics_df = metrics_df.sort_values(
        [
            "app_key",
            "category",
        ]
    ).reset_index(
        drop=True
    )

    # =====================================================
    # SAVE
    # =====================================================

    metrics_df.to_csv(
        OUTPUT_METRICS_FILE,
        index=False,
        encoding="utf-8",
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL POST-CLASSIFIER RESULTS"
    )

    print(
        "=" * 90
    )

    for app_key in APPS:

        print(
            f"\n{app_key.upper()}"
        )

        app_metrics = metrics_df[
            metrics_df["app_key"]
            == app_key
        ].copy()

        app_metrics = app_metrics.sort_values(
            "prevalence_among_classifier_complaints_pct",
            ascending=False,
        )

        for _, row in (
            app_metrics.iterrows()
        ):

            print(
                f"\n  {row['category']}"
            )

            print(
                f"    Reviews: "
                f"{int(row['review_count']):,}"
            )

            print(
                f"    Prevalence among classifier complaints: "
                f"{row['prevalence_among_classifier_complaints_pct']:.2f}%"
            )

            print(
                f"    Share of mapped pain: "
                f"{row['share_of_mapped_pain_pct']:.2f}%"
            )

            print(
                f"    1–2★ within category: "
                f"{row['severity_1_2_within_category_pct']:.2f}%"
            )

            print(
                f"    1★ within category: "
                f"{row['severity_1_star_within_category_pct']:.2f}%"
            )

            print(
                f"    Mean topic strength: "
                f"{row['mean_topic_strength']:.4f}"
            )

            print(
                f"    Strong assignment coverage: "
                f"{row['strong_assignment_share_pct']:.2f}%"
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

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
        f"\nFinal metrics:\n"
        f"{OUTPUT_METRICS_FILE}"
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL POST-CLASSIFIER PAIN METRICS COMPLETE"
    )

    print(
        "=" * 90
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()