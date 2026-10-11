
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

for path in sorted(DATA_DIR.glob("*_processed.csv")):
    df = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(f"APP: {path.stem}")
    print(f"TOTAL REVIEWS: {len(df):,}")

    for column in [
        "language_detected",
        "script",
    ]:
        if column in df.columns:
            print(f"\n{column} distribution:")
            print(
                df[column]
                .fillna("Missing")
                .value_counts()
                .head(15)
                .to_string()
            )

    for column in [
        "likely_romanized_indian",
        "contains_code_mix",
        "text_eligible",
    ]:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).to_string())

    if {"raw_text", "normalized_text"}.issubset(df.columns):
        raw = df["raw_text"].fillna("").astype(str)
        normalized = (
            df["normalized_text"].fillna("").astype(str)
        )

        changed = raw.str.lower().str.strip() != normalized.str.strip()

        print(f"\nReviews changed by normalization: {changed.sum():,}")
        print("\nSample comparisons:")

        cols = [
            c for c in [
                "raw_text",
                "normalized_text",
                "language_detected",
                "likely_romanized_indian",
            ]
            if c in df.columns
        ]

        print(
            df.loc[changed, cols]
            .head(10)
            .to_string(index=False)
        )
