from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

REFINED_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "refined"
)

MODELING_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

CLASSIFIER_FILE = (
    ANALYSIS_DIR
    / "complaint_classifier_predictions.csv"
)

REFINED_MAPPING_FILE = (
    REFINED_DIR
    / "refined_topic_mapping.csv"
)

# Existing PhonePe topic-model output.
PHONEPE_ASSIGNMENTS_FILE = (
    MODELING_DIR
    / "phonepe_topic_assignments.csv"
)

PHONEPE_MAPPING_FILE = (
    MODELING_DIR
    / "topic_mapping.csv"
)

OUTPUT_ASSIGNMENTS = (
    ANALYSIS_DIR
    / "final_pain_category_assignments.csv"
)

OUTPUT_COVERAGE = (
    ANALYSIS_DIR
    / "final_pain_category_coverage.csv"
)


# ============================================================
# SETTINGS
# ============================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

COMPLAINT_THRESHOLD = 0.75


# ============================================================
# LOAD CLASSIFIER PREDICTIONS
# ============================================================

def load_classifier_predictions() -> pd.DataFrame:

    if not CLASSIFIER_FILE.exists():
        raise FileNotFoundError(
            f"Classifier predictions not found:\n"
            f"{CLASSIFIER_FILE}\n\n"
            "Run complaint_classifier.py first."
        )

    df = pd.read_csv(
        CLASSIFIER_FILE
    )

    required = {
        "reviewId",
        "app_key",
        "score",
        "clean_text",
        "complaint_label",
        "complaint_probability",
        "complaint_confidence",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Classifier predictions are missing columns: "
            f"{sorted(missing)}"
        )

    df["reviewId"] = (
        df["reviewId"]
        .astype(str)
        .str.strip()
    )

    df["app_key"] = (
        df["app_key"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce",
    )

    df["complaint_probability"] = pd.to_numeric(
        df["complaint_probability"],
        errors="coerce",
    )

    # Final complaint pool.
    complaints = df[
        df["complaint_label"]
        == "complaint"
    ].copy()

    return complaints.reset_index(
        drop=True
    )


# ============================================================
# LOAD REFINED PAYTM + GOOGLE PAY ASSIGNMENTS
# ============================================================

def load_refined_assignments(
    app_key: str,
) -> pd.DataFrame:

    path = (
        REFINED_DIR
        / f"{app_key}_refined_assignments.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Refined assignments not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "reviewId",
        "app_key",
        "score",
        "at",
        "normalized_text",
        "topic_id",
        "topic_strength",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{app_key} assignments missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# LOAD PHONEPE ASSIGNMENTS
# ============================================================

def load_phonepe_assignments() -> pd.DataFrame:

    if not PHONEPE_ASSIGNMENTS_FILE.exists():

        raise FileNotFoundError(
            f"PhonePe topic assignments not found:\n"
            f"{PHONEPE_ASSIGNMENTS_FILE}\n\n"
            "Expected the existing PhonePe NMF output."
        )

    df = pd.read_csv(
        PHONEPE_ASSIGNMENTS_FILE
    )

    required = {
        "reviewId",
        "topic_id",
        "topic_strength",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:

        raise ValueError(
            "PhonePe topic assignments are missing columns: "
            f"{sorted(missing)}"
        )

    # Add stable app identifier.
    df["app_key"] = "phonepe"

    return df


# ============================================================
# LOAD REFINED MAPPING
# ============================================================

def load_refined_mapping() -> pd.DataFrame:

    if not REFINED_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Refined topic mapping not found:\n"
            f"{REFINED_MAPPING_FILE}"
        )

    mapping = pd.read_csv(
        REFINED_MAPPING_FILE
    )

    required = {
        "app_key",
        "topic_id",
        "decision",
        "final_category",
    }

    missing = (
        required - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "Refined mapping missing columns: "
            f"{sorted(missing)}"
        )

    return mapping


# ============================================================
# LOAD PHONEPE MAPPING
# ============================================================

def load_phonepe_mapping() -> pd.DataFrame:

    if not PHONEPE_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"PhonePe topic mapping not found:\n"
            f"{PHONEPE_MAPPING_FILE}"
        )

    mapping = pd.read_csv(
        PHONEPE_MAPPING_FILE
    )

    required = {
        "topic_id",
        "decision",
        "final_category",
    }

    missing = (
        required - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "PhonePe mapping missing columns: "
            f"{sorted(missing)}"
        )

    mapping["app_key"] = "phonepe"

    return mapping


# ============================================================
# NORMALIZE ASSIGNMENTS
# ============================================================

def normalize_assignment_columns(
    df: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    result = df.copy()

    result["reviewId"] = (
        result["reviewId"]
        .astype(str)
        .str.strip()
    )

    result["app_key"] = app_key

    result["topic_id"] = pd.to_numeric(
        result["topic_id"],
        errors="coerce",
    )

    result["topic_strength"] = pd.to_numeric(
        result["topic_strength"],
        errors="coerce",
    )

    return result


# ============================================================
# APPLY MAPPING
# ============================================================

def apply_mapping(
    assignments: pd.DataFrame,
    mapping: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    app_mapping = mapping[
        mapping["app_key"]
        == app_key
    ].copy()

    # Only audited KEEP / MERGE topics are valid
    # final pain categories.
    app_mapping = app_mapping[
        app_mapping["decision"].isin(
            [
                "KEEP",
                "MERGE",
            ]
        )
    ].copy()

    app_mapping = app_mapping[
        [
            "topic_id",
            "decision",
            "final_category",
        ]
    ]

    result = assignments.merge(
        app_mapping,
        on="topic_id",
        how="left",
    )

    result["is_mapped_pain"] = (
        result["final_category"]
        .notna()
        & result["final_category"].ne("")
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("FINAL COMPLAINT → PAIN CATEGORY INTEGRATION")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load final classifier output.
    # --------------------------------------------------------

    complaints = (
        load_classifier_predictions()
    )

    print(
        f"\nFinal complaint pool: "
        f"{len(complaints):,}"
    )

    print("\nComplaints by app:")

    print(
        complaints["app_key"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 2. Load mappings.
    # --------------------------------------------------------

    refined_mapping = (
        load_refined_mapping()
    )

    phonepe_mapping = (
        load_phonepe_mapping()
    )

    all_mapping = pd.concat(
        [
            phonepe_mapping,
            refined_mapping,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate no duplicate mapping keys.
    # --------------------------------------------------------

    duplicate_mapping_keys = (
        all_mapping[
            all_mapping.duplicated(
                subset=[
                    "app_key",
                    "topic_id",
                ],
                keep=False,
            )
        ]
    )

    if not duplicate_mapping_keys.empty:

        raise ValueError(
            "Duplicate app/topic mapping keys detected:\n"
            + duplicate_mapping_keys[
                [
                    "app_key",
                    "topic_id",
                    "decision",
                    "final_category",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 3. Process app-specific topic assignments.
    # --------------------------------------------------------

    mapped_frames = []

    for app_key in APPS:

        print(
            f"\nProcessing {app_key.upper()}..."
        )

        # ----------------------------------------------
        # Load appropriate topic assignments.
        # ----------------------------------------------

        if app_key == "phonepe":

            assignments = (
                load_phonepe_assignments()
            )

        else:

            assignments = (
                load_refined_assignments(
                    app_key
                )
            )

        assignments = (
            normalize_assignment_columns(
                assignments,
                app_key,
            )
        )

        print(
            f"Topic assignments available: "
            f"{len(assignments):,}"
        )

        # ----------------------------------------------
        # Apply audited mapping.
        # ----------------------------------------------

        mapped = apply_mapping(
            assignments=assignments,
            mapping=all_mapping,
            app_key=app_key,
        )

        # ----------------------------------------------
        # Keep ONLY classifier-confirmed complaints.
        # ----------------------------------------------

        app_complaints = complaints[
            complaints["app_key"]
            == app_key
        ][
            [
                "reviewId",
                "complaint_probability",
                "complaint_confidence",
            ]
        ].copy()

        before_merge = len(
            app_complaints
        )

        # Merge classifier decisions onto topic assignments.
        merged = mapped.merge(
            app_complaints,
            on="reviewId",
            how="inner",
        )

        # ----------------------------------------------
        # Validate join coverage.
        # ----------------------------------------------

        matched = len(
            merged
        )

        unmatched = (
            before_merge - matched
        )

        print(
            f"Classifier complaints: "
            f"{before_merge:,}"
        )

        print(
            f"Complaints with topic assignment: "
            f"{matched:,}"
        )

        print(
            f"Complaints without topic assignment: "
            f"{unmatched:,}"
        )

        if before_merge > 0:

            coverage = (
                matched
                / before_merge
                * 100
            )

            print(
                f"Topic coverage of complaints: "
                f"{coverage:.2f}%"
            )

        # ----------------------------------------------
        # Add to final output.
        # ----------------------------------------------

        merged["classifier_threshold"] = (
            COMPLAINT_THRESHOLD
        )

        mapped_frames.append(
            merged
        )

    # --------------------------------------------------------
    # 4. Combine apps.
    # --------------------------------------------------------

    final_df = pd.concat(
        mapped_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 5. Remove duplicate review IDs.
    # --------------------------------------------------------

    duplicate_reviews = (
        final_df[
            final_df.duplicated(
                subset=["reviewId"],
                keep=False,
            )
        ]
    )

    if not duplicate_reviews.empty:

        raise ValueError(
            "Duplicate review IDs found after integration:\n"
            + duplicate_reviews[
                [
                    "reviewId",
                    "app_key",
                    "topic_id",
                ]
            ].head(20).to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. Final category status.
    # --------------------------------------------------------

    final_df["is_mapped_pain"] = (
        final_df["final_category"]
        .notna()
        & final_df["final_category"].ne("")
    )

    # Only mapped complaint reviews are final pain reviews.
    final_df["final_pain_review"] = (
        final_df["is_mapped_pain"]
    )

    # --------------------------------------------------------
    # 7. Save full integrated assignment file.
    # --------------------------------------------------------

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_ASSIGNMENTS,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 8. Coverage summary.
    # --------------------------------------------------------

    coverage_rows = []

    for app_key in APPS:

        app_complaints = complaints[
            complaints["app_key"]
            == app_key
        ]

        app_final = final_df[
            final_df["app_key"]
            == app_key
        ]

        complaint_count = len(
            app_complaints
        )

        topic_assigned_count = len(
            app_final
        )

        mapped_pain_count = int(
            app_final[
                "is_mapped_pain"
            ].sum()
        )

        if complaint_count:

            topic_coverage = (
                topic_assigned_count
                / complaint_count
                * 100
            )

        else:

            topic_coverage = 0

        if topic_assigned_count:

            mapping_coverage = (
                mapped_pain_count
                / topic_assigned_count
                * 100
            )

        else:

            mapping_coverage = 0

        coverage_rows.append(
            {
                "app_key": app_key,
                "classifier_complaints": (
                    complaint_count
                ),
                "topic_assigned_complaints": (
                    topic_assigned_count
                ),
                "mapped_pain_reviews": (
                    mapped_pain_count
                ),
                "topic_assignment_coverage_pct": (
                    topic_coverage
                ),
                "mapping_coverage_pct": (
                    mapping_coverage
                ),
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    coverage_df.to_csv(
        OUTPUT_COVERAGE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 9. Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL INTEGRATION SUMMARY")
    print("=" * 90)

    print(
        f"\nClassifier complaints: "
        f"{len(complaints):,}"
    )

    print(
        f"Topic-assigned complaints: "
        f"{len(final_df):,}"
    )

    print(
        f"Mapped pain reviews: "
        f"{int(final_df['is_mapped_pain'].sum()):,}"
    )

    print(
        f"Unmapped topic reviews: "
        f"{int((~final_df['is_mapped_pain']).sum()):,}"
    )

    print("\nCoverage by app:")

    display_df = coverage_df.copy()

    display_df[
        "topic_assignment_coverage_pct"
    ] = (
        display_df[
            "topic_assignment_coverage_pct"
        ].round(2)
    )

    display_df[
        "mapping_coverage_pct"
    ] = (
        display_df[
            "mapping_coverage_pct"
        ].round(2)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print("\n" + "-" * 90)
    print("FILES SAVED")
    print("-" * 90)

    print(
        f"\nIntegrated assignments:\n"
        f"{OUTPUT_ASSIGNMENTS}"
    )

    print(
        f"\nCoverage report:\n"
        f"{OUTPUT_COVERAGE}"
    )

    print("\n" + "=" * 90)
    print("INTEGRATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()