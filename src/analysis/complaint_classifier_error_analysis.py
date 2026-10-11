from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "complaint_classifier_oof_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "complaint_classifier_error_analysis.csv"
)

BEST_MODEL = "char_tfidf_logistic"


# ============================================================
# LOAD DATA
# ============================================================

if not OOF_FILE.exists():
    raise FileNotFoundError(
        f"OOF prediction file not found:\n{OOF_FILE}\n\n"
        "Run complaint_classifier_evaluation.py first."
    )

df = pd.read_csv(OOF_FILE)


required_columns = {
    "reviewId",
    "app_key",
    "score",
    "review_text",
    "human_label",
    "model",
    "fold",
    "predicted_label",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# SELECT WINNING MODEL
# ============================================================

best = df[df["model"] == BEST_MODEL].copy()

if best.empty:
    raise ValueError(
        f"No rows found for model: {BEST_MODEL}"
    )


# ============================================================
# NORMALIZE LABELS
# ============================================================

best["human_label"] = (
    best["human_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

best["predicted_label"] = (
    best["predicted_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# IDENTIFY ERRORS
# ============================================================

best["error_type"] = "correct"

# False positive:
# predicted complaint, human says non_complaint
best.loc[
    (best["predicted_label"] == "complaint")
    & (best["human_label"] == "non_complaint"),
    "error_type",
] = "false_positive"

# False negative:
# predicted non_complaint, human says complaint
best.loc[
    (best["predicted_label"] == "non_complaint")
    & (best["human_label"] == "complaint"),
    "error_type",
] = "false_negative"


errors = best[best["error_type"] != "correct"].copy()


# ============================================================
# ADD METADATA
# ============================================================

errors["error_priority"] = errors["error_type"].map(
    {
        "false_negative": "HIGH",
        "false_positive": "MEDIUM",
    }
)

errors["model_name"] = BEST_MODEL


# ============================================================
# RENAME FOR READABILITY
# ============================================================

errors = errors.rename(
    columns={
        "reviewId": "review_id",
        "app_key": "app",
    }
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

preferred_columns = [
    "model_name",
    "error_type",
    "error_priority",
    "review_id",
    "app",
    "score",
    "human_label",
    "predicted_label",
    "review_text",
    "fold",
]

remaining_columns = [
    c for c in errors.columns
    if c not in preferred_columns
]

errors = errors[
    preferred_columns + remaining_columns
]


# ============================================================
# SORT
# ============================================================

error_order = {
    "false_negative": 0,
    "false_positive": 1,
}

errors["_sort_order"] = errors["error_type"].map(error_order)

errors = (
    errors
    .sort_values(
        by=["_sort_order", "app", "review_id"],
        kind="stable"
    )
    .drop(columns="_sort_order")
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

errors.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# REPORT
# ============================================================

print("=" * 70)
print("COMPLAINT CLASSIFIER ERROR ANALYSIS")
print("=" * 70)

print(f"\nBest model: {BEST_MODEL}")
print(f"Total validation rows: {len(best):,}")
print(f"Total errors: {len(errors):,}")

fp_count = (
    errors["error_type"] == "false_positive"
).sum()

fn_count = (
    errors["error_type"] == "false_negative"
).sum()

print(f"False positives: {fp_count:,}")
print(f"False negatives: {fn_count:,}")


# ============================================================
# ERRORS BY APP
# ============================================================

print("\n" + "-" * 70)
print("ERRORS BY APP")
print("-" * 70)

app_summary = (
    errors
    .groupby(["app", "error_type"])
    .size()
    .unstack(fill_value=0)
)

print(app_summary.to_string())


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "-" * 70)
print("CONFUSION COUNTS")
print("-" * 70)

confusion = pd.crosstab(
    best["human_label"],
    best["predicted_label"]
)

print(confusion.to_string())


# ============================================================
# PRINT ERRORS
# ============================================================

print("\n" + "-" * 70)
print("ERROR DETAILS")
print("-" * 70)

for _, row in errors.iterrows():

    print(
        f"\n[{row['error_type'].upper()}] "
        f"{row['app']} | "
        f"Rating={row['score']} | "
        f"Fold={row['fold']} | "
        f"ID={row['review_id']}"
    )

    print(
        f"Human: {row['human_label']} | "
        f"Predicted: {row['predicted_label']}"
    )

    print(
        f"Text: {row['review_text']}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("Saved error-analysis file:")
print(OUTPUT_FILE)
print("=" * 70)