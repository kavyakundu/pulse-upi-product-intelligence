from pathlib import Path
import re

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Romanized Indian-language indicators
# ---------------------------------------------------------
# These are deliberately broad lexical indicators.
# This is a STRESS TEST, not a language classifier.
# ---------------------------------------------------------

HINGLISH_TERMS = [
    "paisa",
    "paise",
    "nahi",
    "nahin",
    "acha",
    "achha",
    "accha",
    "bahut",
    "bhot",
    "kya",
    "kyu",
    "kyon",
    "hai",
    "ha",
    "hota",
    "ho raha",
    "nahi aa",
    "kat gaya",
    "kat gya",
    "bheja",
    "bhej",
    "milta",
    "mil gaya",
    "nahi mila",
    "chahiye",
    "problem ha",
    "problem hai",
    "kaise",
    "krna",
    "karna",
    "chal",
    "chalta",
    "nhi",
    "app achha",
    "bahut badhiya",
    "bakwas",
    "bekar",
    "kharaab",
    "kharab",
]


def looks_romanized_indian(text: str) -> bool:
    """
    Conservative lexical flag for likely Romanized
    Indian-language / Hinglish-style reviews.

    This is not a language detector.
    """

    text = str(text).lower()

    matched_terms = 0

    for term in HINGLISH_TERMS:

        if re.search(
            rf"\b{re.escape(term)}\b",
            text,
        ):
            matched_terms += 1

    # Require at least one distinctive Romanized term
    # and at least two words to avoid isolated fragments.
    word_count = len(text.split())

    return (
        matched_terms >= 1
        and word_count >= 2
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    datasets = []

    for app_key in [
        "phonepe",
        "paytm",
        "googlepay",
    ]:

        path = (
            PROCESSED_DATA_DIR
            / f"{app_key}_processed.csv"
        )

        print(f"Loading {app_key}...")

        df = pd.read_csv(path)

        # Only meaningful text.
        df = df[
            df["text_eligible"]
        ].copy()

        df["likely_romanized_indian"] = (
            df["raw_text"]
            .apply(looks_romanized_indian)
        )

        candidates = df[
            df["likely_romanized_indian"]
        ].copy()

        # Keep the sample manageable and balanced by app.
        sample_size = min(20, len(candidates))

        if sample_size > 0:

            sample = candidates.sample(
                n=sample_size,
                random_state=42,
            )

            sample["app_key"] = app_key

            datasets.append(sample)

        print(
            f"  Candidate reviews: "
            f"{len(candidates):,}"
        )

    if not datasets:
        raise RuntimeError(
            "No Romanized/Hinglish-style candidates found."
        )

    stress_test = pd.concat(
        datasets,
        ignore_index=True,
    )

    stress_test = stress_test.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Human labels
    # -----------------------------------------------------

    stress_test["human_sentiment"] = ""

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        VALIDATION_DIR
        / "hinglish_stress_test.csv"
    )

    stress_test.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("HINGLISH STRESS TEST CREATED")
    print("=" * 70)

    print(f"Rows: {len(stress_test):,}")
    print(
        "App distribution:"
    )
    print(
        stress_test["app_key"]
        .value_counts()
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()