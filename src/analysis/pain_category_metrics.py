from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "modeling"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# FINAL CATEGORY DEFINITIONS
# =========================================================
#
# These are the categories produced by the topic mapping.
# The same category can contain multiple NMF topics.
# =========================================================

CATEGORY_ORDER = [
    "App reliability & access",
    "Performance & speed",
    "Transaction & payment issues",
    "Updates & UI",
    "Cashback & rewards",
    "Customer support",
    "Device & security compatibility",
    "Transaction & account issues",
    "AutoPay & payment controls",
    "Bank account & linking",
]


# =========================================================
# LOAD MAPPING
# =========================================================

def load_topic_mapping() -> pd.DataFrame:

    path = (
        MODEL_DIR
        / "topic_mapping.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Topic mapping not found: "
            f"{path}"
        )

    mapping = pd.read_csv(
        path
    )

    required_columns = {
        "app_key",
        "topic_id",
        "decision",
        "final_category",
    }

    missing = (
        required_columns
        - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            f"Topic mapping missing columns: "
            f"{sorted(missing)}"
        )

    return mapping


# =========================================================
# LOAD APP ASSIGNMENTS
# =========================================================

def load_assignments(
    app_key: str,
) -> pd.DataFrame:

    path = (
        MODEL_DIR
        / f"{app_key}_topic_assignments.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Assignments not found: "
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "topic_id",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{app_key}: assignments missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# APPLY TOPIC MAPPING
# =========================================================

def apply_mapping(
    assignments: pd.DataFrame,
    mapping: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    app_mapping = mapping[
        mapping["app_key"]
        == app_key
    ].copy()

    if app_mapping.empty:
        raise ValueError(
            f"No topic mapping found "
            f"for {app_key}."
        )

    # Keep only topics that are explicitly mapped
    # to a final category.
    app_mapping = app_mapping[
        app_mapping["decision"]
        .isin(["KEEP", "MERGE"])
    ].copy()

    if app_mapping.empty:
        raise ValueError(
            f"No usable mapped topics found "
            f"for {app_key}."
        )

    app_mapping = app_mapping[
        [
            "topic_id",
            "decision",
            "final_category",
        ]
    ].copy()

    merged = assignments.merge(
        app_mapping,
        on="topic_id",
        how="left",
    )

    # Reviews belonging to NOISE topics have no
    # final_category and are excluded from pain metrics.
    merged["is_mapped_pain"] = (
        merged["final_category"]
        .notna()
        & merged["final_category"]
        .ne("")
    )

    return merged


# =========================================================
# CALCULATE METRICS FOR ONE APP
# =========================================================

def calculate_app_metrics(
    df: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    # -----------------------------------------------------
    # Only mapped product-pain reviews.
    # -----------------------------------------------------

    pain_df = df[
        df["is_mapped_pain"]
    ].copy()

    if pain_df.empty:
        raise ValueError(
            f"{app_key}: no mapped pain reviews."
        )

    # -----------------------------------------------------
    # Overall denominator.
    #
    # This is ALL 1–3 star reviews used by the topic model.
    # -----------------------------------------------------

    total_1_3_reviews = len(df)

    total_1_2_reviews = (
        df["score"]
        .isin([1, 2])
        .sum()
    )

    total_1_star_reviews = (
        df["score"]
        .eq(1)
        .sum()
    )

    # -----------------------------------------------------
    # Group by final category.
    # -----------------------------------------------------

    rows = []

    grouped = pain_df.groupby(
        "final_category"
    )

    for category, category_df in grouped:

        review_count = len(
            category_df
        )

        one_to_two_count = (
            category_df["score"]
            .isin([1, 2])
            .sum()
        )

        one_star_count = (
            category_df["score"]
            .eq(1)
            .sum()
        )

        average_rating = (
            category_df["score"]
            .mean()
        )

        share_of_1_3 = (
            review_count
            / total_1_3_reviews
            * 100
        )

        share_of_1_2 = (
            one_to_two_count
            / total_1_2_reviews
            * 100
            if total_1_2_reviews
            else 0
        )

        share_of_1_star = (
            one_star_count
            / total_1_star_reviews
            * 100
            if total_1_star_reviews
            else 0
        )

        rows.append(
            {
                "app_key": app_key,
                "category": category,
                "review_count": review_count,
                "share_of_1_3_reviews_pct": (
                    share_of_1_3
                ),
                "one_to_two_star_count": (
                    one_to_two_count
                ),
                "share_of_1_2_reviews_pct": (
                    share_of_1_2
                ),
                "one_star_count": (
                    one_star_count
                ),
                "share_of_1_star_reviews_pct": (
                    share_of_1_star
                ),
                "average_rating": (
                    average_rating
                ),
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    return metrics


# =========================================================
# CALCULATE OVERALL SUMMARY
# =========================================================

def calculate_summary(
    all_metrics: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        all_metrics
        .groupby("category", as_index=False)
        .agg(
            total_review_count=(
                "review_count",
                "sum",
            ),
            apps_with_category=(
                "app_key",
                "nunique",
            ),
        )
    )

    summary = summary.sort_values(
        "total_review_count",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return summary


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PAIN CATEGORY METRICS"
    )

    print(
        "=" * 80
    )

    # -----------------------------------------------------
    # Load mapping.
    # -----------------------------------------------------

    mapping = load_topic_mapping()

    all_metrics = []
    all_assignments = []

    # -----------------------------------------------------
    # Process each app.
    # -----------------------------------------------------

    for app_key in APPS:

        print(
            f"\nLoading assignments: "
            f"{app_key}"
        )

        assignments = load_assignments(
            app_key
        )

        print(
            f"1–3★ review assignments: "
            f"{len(assignments):,}"
        )

        mapped = apply_mapping(
            assignments=assignments,
            mapping=mapping,
            app_key=app_key,
        )

        mapped_count = (
            mapped["is_mapped_pain"]
            .sum()
        )

        noise_count = (
            len(mapped)
            - mapped_count
        )

        print(
            f"Mapped pain reviews: "
            f"{mapped_count:,}"
        )

        print(
            f"Noise reviews excluded: "
            f"{noise_count:,}"
        )

        metrics = calculate_app_metrics(
            mapped,
            app_key,
        )

        all_metrics.append(
            metrics
        )

        # Save review-level final categories.
        mapped_output = mapped[
            [
                "reviewId",
                "app_key",
                "score",
                "at",
                "topic_id",
                "decision",
                "final_category",
                "is_mapped_pain",
            ]
        ].copy()

        all_assignments.append(
            mapped_output
        )

    # -----------------------------------------------------
    # Combine metrics.
    # -----------------------------------------------------

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # Put categories in a consistent order
    # where possible.
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

    # -----------------------------------------------------
    # Combined review-level assignments.
    # -----------------------------------------------------

    assignments_df = pd.concat(
        all_assignments,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Overall category summary.
    # -----------------------------------------------------

    summary_df = calculate_summary(
        metrics_df
    )

    # -----------------------------------------------------
    # Save files.
    # -----------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "pain_category_metrics.csv"
    )

    assignments_path = (
        OUTPUT_DIR
        / "pain_category_assignments.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "pain_category_summary.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8",
    )

    assignments_df.to_csv(
        assignments_path,
        index=False,
        encoding="utf-8",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print final metrics.
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL PAIN CATEGORY METRICS"
    )

    print(
        "=" * 80
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
            "review_count",
            ascending=False,
        )

        for _, row in app_metrics.iterrows():

            print(
                f"  {row['category']}: "
                f"{int(row['review_count']):,} reviews "
                f"("
                f"{row['share_of_1_3_reviews_pct']:.2f}% of 1–3★"
                f")"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FILES SAVED"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Review assignments: "
        f"{assignments_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()