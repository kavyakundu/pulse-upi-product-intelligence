from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "modeling"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

OUTPUT_DIR = ANALYSIS_DIR


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

# A review is considered a strong topic assignment when
# the NMF topic strength is at or above this threshold.
MIN_TOPIC_STRENGTH = 0.05


# =========================================================
# LOAD TOPIC MAPPING
# =========================================================

def load_topic_mapping() -> pd.DataFrame:

    path = (
        MODEL_DIR
        / "topic_mapping.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Topic mapping not found: {path}"
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
            "Topic mapping missing columns: "
            f"{sorted(missing)}"
        )

    return mapping


# =========================================================
# LOAD ASSIGNMENTS
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
            f"Assignments not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "topic_id",
        "topic_strength",
        "normalized_text",
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
# APPLY FINAL CATEGORY MAPPING
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

    app_mapping = app_mapping[
        app_mapping["decision"]
        .isin(["KEEP", "MERGE"])
    ].copy()

    app_mapping = app_mapping[
        [
            "topic_id",
            "decision",
            "final_category",
        ]
    ].copy()

    result = assignments.merge(
        app_mapping,
        on="topic_id",
        how="left",
    )

    result["is_mapped"] = (
        result["final_category"]
        .notna()
        & result["final_category"]
        .ne("")
    )

    return result


# =========================================================
# CATEGORY QUALITY METRICS
# =========================================================

def calculate_category_quality(
    df: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    mapped = df[
        df["is_mapped"]
    ].copy()

    if mapped.empty:
        raise ValueError(
            f"{app_key}: no mapped reviews found."
        )

    rows = []

    for category, category_df in (
        mapped.groupby(
            "final_category"
        )
    ):

        total = len(
            category_df
        )

        # ---------------------------------------------
        # Topic strength
        # ---------------------------------------------

        strong_count = (
            category_df[
                "topic_strength"
            ]
            >= MIN_TOPIC_STRENGTH
        ).sum()

        strong_share = (
            strong_count
            / total
            * 100
        )

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

        # ---------------------------------------------
        # Very short reviews
        # ---------------------------------------------

        text = (
            category_df[
                "normalized_text"
            ]
            .fillna("")
            .astype(str)
        )

        word_counts = (
            text.str.split()
            .str.len()
        )

        short_count = (
            word_counts <= 2
        ).sum()

        short_share = (
            short_count
            / total
            * 100
        )

        # ---------------------------------------------
        # Rating mix
        # ---------------------------------------------

        one_star = (
            category_df["score"]
            .eq(1)
            .sum()
        )

        two_star = (
            category_df["score"]
            .eq(2)
            .sum()
        )

        three_star = (
            category_df["score"]
            .eq(3)
            .sum()
        )

        negative_share = (
            (one_star + two_star)
            / total
            * 100
        )

        # ---------------------------------------------
        # Unique review count
        # ---------------------------------------------

        unique_reviews = (
            category_df[
                "reviewId"
            ].nunique()
        )

        # ---------------------------------------------
        # Quality interpretation
        # ---------------------------------------------

        if strong_share >= 70:
            strength_quality = "High"

        elif strong_share >= 40:
            strength_quality = "Medium"

        else:
            strength_quality = "Low"

        rows.append(
            {
                "app_key": app_key,
                "category": category,
                "review_count": total,
                "unique_review_count": (
                    unique_reviews
                ),
                "mean_topic_strength": (
                    mean_strength
                ),
                "median_topic_strength": (
                    median_strength
                ),
                "strong_assignment_count": (
                    strong_count
                ),
                "strong_assignment_share_pct": (
                    strong_share
                ),
                "short_review_count": (
                    short_count
                ),
                "short_review_share_pct": (
                    short_share
                ),
                "one_star_count": (
                    one_star
                ),
                "two_star_count": (
                    two_star
                ),
                "three_star_count": (
                    three_star
                ),
                "negative_rating_share_pct": (
                    negative_share
                ),
                "strength_quality": (
                    strength_quality
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# FIND POTENTIAL WEAK CATEGORIES
# =========================================================

def identify_warnings(
    quality_df: pd.DataFrame,
) -> pd.DataFrame:

    warnings = []

    for _, row in quality_df.iterrows():

        issues = []

        if (
            row[
                "strong_assignment_share_pct"
            ]
            < 40
        ):

            issues.append(
                "Low topic-strength coverage"
            )

        if (
            row[
                "short_review_share_pct"
            ]
            > 60
        ):

            issues.append(
                "Mostly very short reviews"
            )

        if (
            row[
                "negative_rating_share_pct"
            ]
            < 60
        ):

            issues.append(
                "Weak negative-rating concentration"
            )

        if issues:

            warnings.append(
                {
                    "app_key": row["app_key"],
                    "category": row["category"],
                    "warnings": "; ".join(
                        issues
                    ),
                }
            )

    return pd.DataFrame(
        warnings
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
        "CATEGORY VALIDATION"
    )

    print(
        "=" * 90
    )

    mapping = load_topic_mapping()

    quality_results = []

    for app_key in APPS:

        print(
            f"\nChecking {app_key}..."
        )

        assignments = (
            load_assignments(
                app_key
            )
        )

        mapped = apply_mapping(
            assignments=assignments,
            mapping=mapping,
            app_key=app_key,
        )

        quality = (
            calculate_category_quality(
                mapped,
                app_key,
            )
        )

        quality_results.append(
            quality
        )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    quality_df = pd.concat(
        quality_results,
        ignore_index=True,
    )

    quality_df = quality_df.sort_values(
        [
            "app_key",
            "review_count",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Warnings
    # -----------------------------------------------------

    warnings_df = (
        identify_warnings(
            quality_df
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    quality_path = (
        OUTPUT_DIR
        / "category_quality.csv"
    )

    warnings_path = (
        OUTPUT_DIR
        / "category_quality_warnings.csv"
    )

    quality_df.to_csv(
        quality_path,
        index=False,
        encoding="utf-8",
    )

    warnings_df.to_csv(
        warnings_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CATEGORY QUALITY RESULTS"
    )

    print(
        "=" * 90
    )

    for app_key in APPS:

        print(
            f"\n{app_key.upper()}"
        )

        app_quality = quality_df[
            quality_df["app_key"]
            == app_key
        ]

        for _, row in app_quality.iterrows():

            print(
                f"  {row['category']}:"
            )

            print(
                f"    Reviews: "
                f"{int(row['review_count']):,}"
            )

            print(
                f"    Strong assignment: "
                f"{row['strong_assignment_share_pct']:.2f}%"
            )

            print(
                f"    Mean topic strength: "
                f"{row['mean_topic_strength']:.4f}"
            )

            print(
                f"    Median topic strength: "
                f"{row['median_topic_strength']:.4f}"
            )

            print(
                f"    Short reviews: "
                f"{row['short_review_share_pct']:.2f}%"
            )

            print(
                f"    1–2★ share: "
                f"{row['negative_rating_share_pct']:.2f}%"
            )

            print(
                f"    Quality: "
                f"{row['strength_quality']}"
            )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "WARNINGS"
    )

    print(
        "=" * 90
    )

    if warnings_df.empty:

        print(
            "No quality warnings detected."
        )

    else:

        for _, row in warnings_df.iterrows():

            print(
                f"  {row['app_key']} | "
                f"{row['category']} | "
                f"{row['warnings']}"
            )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"Saved quality report: "
        f"{quality_path}"
    )

    print(
        f"Saved warnings: "
        f"{warnings_path}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()