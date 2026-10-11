from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "complaint_deployment_audit_sample.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "complaint_classifier_threshold_analysis.csv"
)


# Thresholds to evaluate.
THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# LOAD
# ============================================================

if not AUDIT_FILE.exists():
    raise FileNotFoundError(
        f"Audit file not found:\n{AUDIT_FILE}"
    )

df = pd.read_csv(AUDIT_FILE)

required = {
    "human_label",
    "complaint_probability",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# REMOVE AMBIGUOUS
# ============================================================

df["human_label"] = (
    df["human_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["complaint_probability"] = pd.to_numeric(
    df["complaint_probability"],
    errors="coerce",
)

df = df[
    df["human_label"].isin(
        [
            "complaint",
            "non_complaint",
        ]
    )
].copy()

df = df[
    df["complaint_probability"].notna()
].copy()


# ============================================================
# TRUE LABEL
# ============================================================

y_true = (
    df["human_label"]
    == "complaint"
).astype(int)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

rows = []

for threshold in THRESHOLDS:

    y_pred = (
        df["complaint_probability"]
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    predicted_complaints = int(
        y_pred.sum()
    )

    predicted_non_complaints = (
        len(y_pred)
        - predicted_complaints
    )

    rows.append(
        {
            "threshold": threshold,
            "reviews": len(df),
            "predicted_complaints": predicted_complaints,
            "predicted_non_complaints": predicted_non_complaints,
            "complaint_precision": precision,
            "complaint_recall": recall,
            "complaint_f1": f1,
        }
    )


results = pd.DataFrame(rows)


# ============================================================
# FORMAT
# ============================================================

for column in [
    "complaint_precision",
    "complaint_recall",
    "complaint_f1",
]:

    results[column] = (
        results[column] * 100
    ).round(2)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("COMPLAINT CLASSIFIER THRESHOLD ANALYSIS")
print("=" * 80)

print(
    f"\nUsable audit reviews: {len(df):,}"
)

print("\nThreshold comparison:")
print(
    results.to_string(
        index=False
    )
)

# Best F1 threshold
best_f1 = results.loc[
    results["complaint_f1"].idxmax()
]

print("\n" + "-" * 80)
print("BEST F1 THRESHOLD")
print("-" * 80)

print(
    best_f1.to_string()
)

print("\n" + "-" * 80)
print("FILES SAVED")
print("-" * 80)

print(OUTPUT_FILE)

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 80)