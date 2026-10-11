from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "complaint_classifier_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "complaint_deployment_audit_sample.csv"
)

RANDOM_STATE = 42

# 40 reviews per app = 120 total
SAMPLE_PER_APP = 40


# ============================================================
# LOAD
# ============================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}\n\n"
        "Run complaint_classifier.py first."
    )

df = pd.read_csv(PREDICTIONS_FILE)

required = {
    "reviewId",
    "app_key",
    "score",
    "clean_text",
    "complaint_label",
    "complaint_probability",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# NORMALIZE
# ============================================================

df["score"] = pd.to_numeric(
    df["score"],
    errors="coerce",
)

df["complaint_probability"] = pd.to_numeric(
    df["complaint_probability"],
    errors="coerce",
)


# ============================================================
# CREATE PROBABILITY STRATA
# ============================================================

def probability_band(p):

    if p < 0.40:
        return "low"

    if p < 0.60:
        return "borderline"

    if p < 0.80:
        return "moderate"

    return "high"


df["probability_band"] = (
    df["complaint_probability"]
    .apply(probability_band)
)


# ============================================================
# STRATIFIED SAMPLE
# ============================================================
#
# Per app:
#
# 10 low/borderline
# 10 moderate
# 10 high-confidence complaint
# 10 additional random reviews
#
# This deliberately exposes the model to difficult cases,
# not just easy high-confidence complaints.
# ============================================================

samples = []

for app_key, app_df in df.groupby(
    "app_key",
    sort=True,
):

    app_df = app_df.copy()

    # --------------------------------------------------------
    # Borderline / low
    # --------------------------------------------------------

    difficult = app_df[
        app_df["probability_band"].isin(
            ["low", "borderline"]
        )
    ]

    n_difficult = min(
        10,
        len(difficult),
    )

    difficult_sample = difficult.sample(
        n=n_difficult,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Moderate
    # --------------------------------------------------------

    moderate = app_df[
        app_df["probability_band"]
        == "moderate"
    ]

    n_moderate = min(
        10,
        len(moderate),
    )

    moderate_sample = moderate.sample(
        n=n_moderate,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # High-confidence predicted complaints
    # --------------------------------------------------------

    high_complaints = app_df[
        (app_df["probability_band"] == "high")
        & (
            app_df["complaint_label"]
            == "complaint"
        )
    ]

    n_high = min(
        10,
        len(high_complaints),
    )

    high_sample = high_complaints.sample(
        n=n_high,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Additional random sample
    # --------------------------------------------------------

    already_selected = set(
        difficult_sample["reviewId"]
    ) | set(
        moderate_sample["reviewId"]
    ) | set(
        high_sample["reviewId"]
    )

    remaining = app_df[
        ~app_df["reviewId"].isin(
            already_selected
        )
    ]

    n_remaining = min(
        10,
        len(remaining),
    )

    random_sample = remaining.sample(
        n=n_remaining,
        random_state=RANDOM_STATE,
    )

    samples.extend(
        [
            difficult_sample,
            moderate_sample,
            high_sample,
            random_sample,
        ]
    )


# ============================================================
# COMBINE
# ============================================================

audit = pd.concat(
    samples,
    ignore_index=True,
)


# Remove accidental duplicates.
audit = (
    audit
    .drop_duplicates(
        subset=["reviewId"]
    )
    .reset_index(drop=True)
)


# ============================================================
# ADD HUMAN-LABEL COLUMNS
# ============================================================

audit["human_label"] = ""

audit["label_reason"] = ""


# ============================================================
# SHUFFLE
# ============================================================

audit = audit.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(
    drop=True
)

audit["audit_id"] = (
    audit.index + 1
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "audit_id",
    "reviewId",
    "app_key",
    "score",
    "clean_text",
    "complaint_label",
    "complaint_probability",
    "probability_band",
    "human_label",
    "label_reason",
]

audit = audit[
    output_columns
]


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("DEPLOYMENT AUDIT SAMPLE")
print("=" * 80)

print(
    f"\nAudit sample size: {len(audit):,}"
)

print("\nBy app:")

print(
    audit["app_key"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nBy probability band:")

print(
    audit["probability_band"]
    .value_counts()
    .to_string()
)

print("\nBy rating:")

print(
    audit["score"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "-" * 80)
print("OUTPUT")
print("-" * 80)

print(OUTPUT_FILE)

print("\n" + "=" * 80)
print("AUDIT SAMPLE CREATED")
print("=" * 80)

print(
    "\nOpen the CSV and fill only these two columns:"
)

print("  human_label")
print("  label_reason")

print(
    "\nAllowed human_label values:"
)

print("  complaint")
print("  non_complaint")
print("  ambiguous")