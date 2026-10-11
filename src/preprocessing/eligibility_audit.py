
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

for path in sorted(DATA.glob("*_processed.csv")):
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}\n{path.stem}: {len(df):,} reviews")

    # Identify each eligibility condition separately.
    short_words = df["word_count"].fillna(0) < 2
    emoji_only = df["emoji_only"].fillna(False).astype(bool)
    short_text = df["normalized_text"].fillna("").str.len() < 3

    excluded = ~df["text_eligible"].fillna(False).astype(bool)

    print(f"Excluded total: {excluded.sum():,}")
    print(f"Fewer than 2 words: {short_words.sum():,}")
    print(f"Emoji-only: {emoji_only.sum():,}")
    print(f"Normalized text under 3 characters: {short_text.sum():,}")
    print(
        "Excluded by all three conditions: "
        f"{(excluded & short_words & emoji_only & short_text).sum():,}"
    )

    print("\nExcluded review examples:")
    cols = [
        "raw_text", "normalized_text", "word_count",
        "emoji_only", "language_detected", "script"
    ]
    print(
        df.loc[excluded, cols]
        .head(15)
        .to_string(index=False)
    )

    print("\nPotential Romanized Indian-language examples:")
    romanized = df["likely_romanized_indian"].fillna(False).astype(bool)
    cols = ["raw_text", "normalized_text", "language_detected"]
    print(
        df.loc[romanized, cols]
        .sample(n=min(10, romanized.sum()), random_state=42)
        .to_string(index=False)
    )
