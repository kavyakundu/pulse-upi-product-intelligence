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
    / "analysis"
    / "complaint_classifier_deployment_diagnostics.csv"
)


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
        f"Missing columns: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# BASIC CLEANUP
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

df["text_length"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["word_count"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

total = len(df)

complaints = int(
    (
        df["complaint_label"]
        == "complaint"
    ).sum()
)

non_complaints = total - complaints


print("=" * 80)
print("COMPLAINT CLASSIFIER DEPLOYMENT DIAGNOSTICS")
print("=" * 80)

print("\nOVERALL")
print("-" * 80)

print(f"Total candidates:       {total:,}")
print(f"Predicted complaints:   {complaints:,}")
print(f"Predicted non-complaints: {non_complaints:,}")

if total:
    print(
        f"Complaint rate:         "
        f"{complaints / total * 100:.2f}%"
    )


# ============================================================
# BY RATING
# ============================================================

print("\n" + "-" * 80)
print("PREDICTIONS BY RATING")
print("-" * 80)

rating_summary = (
    df.groupby("score")
    .agg(
        reviews=("reviewId", "count"),
        complaints=(
            "complaint_label",
            lambda x: (x == "complaint").sum()
        ),
        avg_complaint_probability=(
            "complaint_probability",
            "mean"
        ),
    )
    .reset_index()
)

rating_summary["complaint_rate"] = (
    rating_summary["complaints"]
    / rating_summary["reviews"]
    * 100
)

rating_summary["complaint_rate"] = (
    rating_summary["complaint_rate"]
    .round(2)
)

rating_summary[
    "avg_complaint_probability"
] = (
    rating_summary[
        "avg_complaint_probability"
    ]
    .round(4)
)

print(
    rating_summary.to_string(
        index=False
    )
)


# ============================================================
# BY APP
# ============================================================

print("\n" + "-" * 80)
print("PREDICTIONS BY APP")
print("-" * 80)

app_summary = (
    df.groupby("app_key")
    .agg(
        reviews=("reviewId", "count"),
        complaints=(
            "complaint_label",
            lambda x: (x == "complaint").sum()
        ),
        avg_probability=(
            "complaint_probability",
            "mean"
        ),
        median_probability=(
            "complaint_probability",
            "median"
        ),
    )
    .reset_index()
)

app_summary["complaint_rate"] = (
    app_summary["complaints"]
    / app_summary["reviews"]
    * 100
)

app_summary["complaint_rate"] = (
    app_summary["complaint_rate"]
    .round(2)
)

app_summary["avg_probability"] = (
    app_summary["avg_probability"]
    .round(4)
)

app_summary["median_probability"] = (
    app_summary["median_probability"]
    .round(4)
)

print(
    app_summary.to_string(
        index=False
    )
)


# ============================================================
# PROBABILITY BANDS
# ============================================================

print("\n" + "-" * 80)
print("COMPLAINT-PROBABILITY BANDS")
print("-" * 80)

bins = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

labels = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]

df["probability_band"] = pd.cut(
    df["complaint_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

probability_summary = (
    df.groupby(
        "probability_band",
        observed=False,
    )
    .agg(
        reviews=("reviewId", "count"),
        complaints=(
            "complaint_label",
            lambda x: (x == "complaint").sum()
        ),
    )
    .reset_index()
)

print(
    probability_summary.to_string(
        index=False
    )
)


# ============================================================
# HIGH- AND LOW-CONFIDENCE PREDICTIONS
# ============================================================

print("\n" + "-" * 80)
print("CONFIDENCE SUMMARY")
print("-" * 80)

confidence_bands = {
    "Very low confidence": (
        df["complaint_confidence"] < 0.60
    ),
    "Moderate confidence": (
        (df["complaint_confidence"] >= 0.60)
        & (df["complaint_confidence"] < 0.80)
    ),
    "High confidence": (
        (df["complaint_confidence"] >= 0.80)
        & (df["complaint_confidence"] < 0.95)
    ),
    "Very high confidence": (
        df["complaint_confidence"] >= 0.95
    ),
}

for name, mask in confidence_bands.items():

    subset = df.loc[mask]

    count = len(subset)

    if count == 0:
        complaint_rate = 0
    else:
        complaint_rate = (
            (
                subset["complaint_label"]
                == "complaint"
            ).mean()
            * 100
        )

    print(
        f"{name:22s}: "
        f"{count:7,} reviews | "
        f"{complaint_rate:6.2f}% complaints"
    )


# ============================================================
# BORDERLINE REVIEWS
# ============================================================

print("\n" + "-" * 80)
print("BORDERLINE PREDICTIONS")
print("-" * 80)

borderline = (
    df[
        df["complaint_probability"].between(
            0.40,
            0.60,
            inclusive="both",
        )
    ]
    .sort_values(
        "complaint_probability"
    )
)

print(
    f"Reviews with complaint probability "
    f"between 0.40 and 0.60: {len(borderline):,}"
)

if not borderline.empty:

    print("\nSample of borderline reviews:")

    sample = borderline.head(30)[
        [
            "reviewId",
            "app_key",
            "score",
            "complaint_label",
            "complaint_probability",
            "complaint_confidence",
            "clean_text",
        ]
    ]

    print(
        sample.to_string(
            index=False
        )
    )


# ============================================================
# PREDICTED NON-COMPLAINT SAMPLE
# ============================================================

print("\n" + "-" * 80)
print("SAMPLE OF PREDICTED NON-COMPLAINTS")
print("-" * 80)

non_complaint_sample = (
    df[
        df["complaint_label"]
        == "non_complaint"
    ]
    .sort_values(
        "complaint_probability"
    )
    .head(30)
)

print(
    non_complaint_sample[
        [
            "reviewId",
            "app_key",
            "score",
            "complaint_probability",
            "complaint_confidence",
            "clean_text",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# MOST CERTAIN COMPLAINTS
# ============================================================

print("\n" + "-" * 80)
print("SAMPLE OF VERY HIGH-CONFIDENCE COMPLAINTS")
print("-" * 80)

complaint_sample = (
    df[
        df["complaint_label"]
        == "complaint"
    ]
    .sort_values(
        "complaint_probability",
        ascending=False,
    )
    .head(20)
)

print(
    complaint_sample[
        [
            "reviewId",
            "app_key",
            "score",
            "complaint_probability",
            "complaint_confidence",
            "clean_text",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE DIAGNOSTIC DATA
# ============================================================

diagnostic_columns = [
    "reviewId",
    "app_key",
    "score",
    "clean_text",
    "text_length",
    "word_count",
    "complaint_label",
    "complaint_probability",
    "complaint_confidence",
    "probability_band",
]

diagnostics = df[
    diagnostic_columns
].copy()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

diagnostics.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSTICS SAVED")
print("=" * 80)

print(OUTPUT_FILE)