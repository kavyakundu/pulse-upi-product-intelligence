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
    / "complaint_classifier_final_audit.csv"
)

RANDOM_STATE = 42

# 40 per app = 120 final-audit reviews
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
    "complaint_confidence",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# CLEAN
# ============================================================

df["score"] = pd.to_numeric(
    df["score"],
    errors="coerce",
)

df["complaint_probability"] = pd.to_numeric(
    df["complaint_probability"],
    errors="coerce",
)

df["complaint_confidence"] = pd.to_numeric(
    df["complaint_confidence"],
    errors="coerce",
)


# ============================================================
# CONFIDENCE STRATA
# ============================================================

def confidence_band(row):

    probability = row["complaint_probability"]
    label = row["complaint_label"]

    # For predicted complaints:
    # borderline / moderate / high confidence
    if label == "complaint":

        if probability < 0.80:
            return "complaint_moderate"

        if probability < 0.90:
            return "complaint_high"

        return "complaint_very_high"

    # For predicted non-complaints:
    # difficult vs clear non-complaints
    if probability >= 0.50:
        return "noncomplaint_borderline"

    if probability >= 0.25:
        return "noncomplaint_moderate"

    return "noncomplaint_clear"


df["audit_band"] = df.apply(
    confidence_band,
    axis=1,
)


# ============================================================
# STRATIFIED FINAL AUDIT
# ============================================================
#
# Each app contributes approximately:
#
# 10 predicted complaints with moderate probability
# 10 predicted complaints with high probability
# 10 predicted complaints with very high probability
# 10 predicted non-complaints, prioritizing difficult cases
#
# This tests BOTH sides of the classifier.
# ============================================================

samples = []

for app_key, app_df in df.groupby(
    "app_key",
    sort=True,
):

    app_df = app_df.copy()

    # --------------------------------------------------------
    # 1. Predicted complaints: moderate
    # --------------------------------------------------------

    moderate_complaints = app_df[
        app_df["audit_band"]
        == "complaint_moderate"
    ]

    n = min(
        10,
        len(moderate_complaints),
    )

    if n > 0:
        samples.append(
            moderate_complaints.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    # --------------------------------------------------------
    # 2. Predicted complaints: high
    # --------------------------------------------------------

    high_complaints = app_df[
        app_df["audit_band"]
        == "complaint_high"
    ]

    n = min(
        10,
        len(high_complaints),
    )

    if n > 0:
        samples.append(
            high_complaints.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    # --------------------------------------------------------
    # 3. Predicted complaints: very high
    # --------------------------------------------------------

    very_high_complaints = app_df[
        app_df["audit_band"]
        == "complaint_very_high"
    ]

    n = min(
        10,
        len(very_high_complaints),
    )

    if n > 0:
        samples.append(
            very_high_complaints.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    # --------------------------------------------------------
    # 4. Predicted non-complaints
    #
    # Prefer borderline/moderate non-complaints because
    # these are the most difficult cases.
    # --------------------------------------------------------

    difficult_noncomplaints = app_df[
        app_df["audit_band"].isin(
            [
                "noncomplaint_borderline",
                "noncomplaint_moderate",
            ]
        )
    ]

    clear_noncomplaints = app_df[
        app_df["audit_band"]
        == "noncomplaint_clear"
    ]

    # Take up to 8 difficult non-complaints.
    n_difficult = min(
        8,
        len(difficult_noncomplaints),
    )

    selected_difficult = (
        difficult_noncomplaints.sample(
            n=n_difficult,
            random_state=RANDOM_STATE,
        )
        if n_difficult > 0
        else difficult_noncomplaints.head(0)
    )

    # Fill remaining slots with clear non-complaints.
    n_remaining = min(
        2,
        len(
            clear_noncomplaints[
                ~clear_noncomplaints["reviewId"].isin(
                    selected_difficult["reviewId"]
                )
            ]
        ),
    )

    if n_remaining > 0:

        selected_clear = clear_noncomplaints.sample(
            n=n_remaining,
            random_state=RANDOM_STATE,
        )

    else:

        selected_clear = clear_noncomplaints.head(0)

    samples.append(selected_difficult)
    samples.append(selected_clear)


# ============================================================
# COMBINE AND DEDUPLICATE
# ============================================================

if not samples:
    raise RuntimeError(
        "No audit samples could be created."
    )

audit = pd.concat(
    samples,
    ignore_index=True,
)

audit = (
    audit
    .drop_duplicates(
        subset=["reviewId"],
    )
    .reset_index(drop=True)
)


# ============================================================
# ADD HUMAN AUDIT COLUMNS
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
    "complaint_confidence",
    "audit_band",
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
print("FINAL POST-DEPLOYMENT AUDIT")
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

print("\nBy predicted label:")

print(
    audit["complaint_label"]
    .value_counts()
    .to_string()
)

print("\nBy audit band:")

print(
    audit["audit_band"]
    .value_counts()
    .sort_index()
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
print("FINAL AUDIT SAMPLE CREATED")
print("=" * 80)

print(
    "\nOpen the CSV and fill only:"
)

print("  human_label")
print("  label_reason")

print(
    "\nAllowed human_label values:"
)

print("  complaint")
print("  non_complaint")
print("  ambiguous")