from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# App datasets
# ---------------------------------------------------------

DATASETS = {
    "PhonePe": RAW_DATA_DIR / "phonepe_reviews.csv",
    "Paytm": RAW_DATA_DIR / "paytm_reviews.csv",
    "Google Pay": RAW_DATA_DIR / "googlepay_reviews.csv",
}


# ---------------------------------------------------------
# Audit function
# ---------------------------------------------------------

def audit_dataset(app_name: str, file_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"{app_name} DATA AUDIT")
    print("=" * 70)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Parse timestamp
    df["at"] = pd.to_datetime(df["at"], errors="coerce")

    print(f"\nFile: {file_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "reviewId",
        "content",
        "score",
        "thumbsUpCount",
        "at",
        "reviewCreatedVersion",
        "appVersion",
        "app_key",
        "app_name",
        "package_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    print("\nRequired columns:")
    if missing_columns:
        print(f"  Missing: {missing_columns}")
    else:
        print("  All present")

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print("\nMissing values:")
    missing = df.isna().sum()

    for column, count in missing.items():
        percentage = (count / len(df)) * 100
        print(
            f"  {column}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    # -----------------------------------------------------
    # Duplicate checks
    # -----------------------------------------------------

    duplicate_ids = df["reviewId"].duplicated().sum()

    print("\nDuplicate review IDs:")
    print(f"  {duplicate_ids:,}")

    # -----------------------------------------------------
    # Rating validation
    # -----------------------------------------------------

    valid_scores = df["score"].isin([1, 2, 3, 4, 5]).all()

    print("\nRating validation:")
    print(f"  All ratings valid: {valid_scores}")

    print("\nRating distribution:")

    rating_counts = (
        df["score"]
        .value_counts()
        .sort_index()
    )

    for rating, count in rating_counts.items():
        percentage = (count / len(df)) * 100
        print(
            f"  {rating} star: "
            f"{count:,} ({percentage:.2f}%)"
        )

    # -----------------------------------------------------
    # Date validation
    # -----------------------------------------------------

    invalid_dates = df["at"].isna().sum()

    print("\nTimestamp validation:")
    print(f"  Invalid/missing timestamps: {invalid_dates:,}")

    if invalid_dates == 0:
        print(f"  Earliest: {df['at'].min()}")
        print(f"  Latest:   {df['at'].max()}")

    # -----------------------------------------------------
    # Text statistics
    # -----------------------------------------------------

    text_lengths = df["content"].dropna().astype(str).str.len()

    if len(text_lengths) > 0:
        print("\nReview text statistics:")
        print(f"  Empty/missing text: {df['content'].isna().sum():,}")
        print(f"  Mean length: {text_lengths.mean():.1f}")
        print(f"  Median length: {text_lengths.median():.1f}")
        print(f"  Max length: {text_lengths.max():,}")

    # -----------------------------------------------------
    # App-version coverage
    # -----------------------------------------------------

    version_missing = df["appVersion"].isna().sum()
    version_present = len(df) - version_missing

    print("\nApp-version coverage:")
    print(
        f"  Present: "
        f"{version_present:,} "
        f"({version_present / len(df) * 100:.2f}%)"
    )
    print(
        f"  Missing: "
        f"{version_missing:,} "
        f"({version_missing / len(df) * 100:.2f}%)"
    )

    # -----------------------------------------------------
    # Review ID uniqueness
    # -----------------------------------------------------

    unique_ids = df["reviewId"].nunique()

    print("\nReview ID uniqueness:")
    print(f"  Unique IDs: {unique_ids:,}")
    print(f"  Total rows: {len(df):,}")
    print(f"  IDs fully unique: {unique_ids == len(df)}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print("UPI APP REVIEW DATA AUDIT")

    for app_name, file_path in DATASETS.items():
        audit_dataset(app_name, file_path)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()