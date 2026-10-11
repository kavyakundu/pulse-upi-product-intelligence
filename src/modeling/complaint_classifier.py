from pathlib import Path
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_FILE = (
    ANALYSIS_DIR
    / "complaint_classifier_predictions.csv"
)

SUMMARY_FILE = (
    ANALYSIS_DIR
    / "complaint_classifier_deployment_summary.csv"
)

MODEL_METADATA_FILE = (
    MODEL_DIR
    / "complaint_classifier_metadata.json"
)


# ============================================================
# FINAL MODEL
# ============================================================

BEST_MODEL = "char_tfidf_logistic"

# Final deployment threshold selected from the
# independent deployment-audit sample.
COMPLAINT_THRESHOLD = 0.75


VECTORIZER_CONFIG = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 50000,
    "sublinear_tf": True,
    "lowercase": True,
}


CLASSIFIER_CONFIG = {
    "C": 1.0,
    "max_iter": 2000,
    "random_state": 42,
}


# ============================================================
# PROCESSED FILES
# ============================================================

PROCESSED_FILES = {
    "phonepe": PROCESSED_DIR / "phonepe_processed.csv",
    "paytm": PROCESSED_DIR / "paytm_processed.csv",
    "googlepay": PROCESSED_DIR / "googlepay_processed.csv",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# LOAD EXACT FINAL GOLD SET
# ============================================================

def load_final_gold_set():

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"OOF prediction file not found:\n{OOF_FILE}\n\n"
            "Run complaint_classifier_evaluation.py first."
        )

    df = pd.read_csv(OOF_FILE)

    required = {
        "reviewId",
        "review_text",
        "human_label",
        "model",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"OOF file is missing columns: {sorted(missing)}"
        )

    # Use ONLY the winning model's OOF rows.
    gold = df[
        df["model"] == BEST_MODEL
    ].copy()

    if gold.empty:
        raise ValueError(
            f"No rows found for model: {BEST_MODEL}"
        )

    gold["reviewId"] = (
        gold["reviewId"]
        .astype(str)
        .str.strip()
    )

    gold["review_text"] = normalize_text(
        gold["review_text"]
    )

    gold["human_label"] = (
        gold["human_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Only final binary labels are allowed.
    gold = gold[
        gold["human_label"].isin(
            [
                "complaint",
                "non_complaint",
            ]
        )
    ].copy()

    # Remove duplicate review IDs.
    before = len(gold)

    gold = (
        gold
        .drop_duplicates(
            subset=["reviewId"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before - len(gold)
    )

    # This must exactly match the final evaluation set.
    if len(gold) != 368:
        raise ValueError(
            "Gold-set size mismatch.\n"
            f"Expected 368 reviews from the final evaluation, "
            f"but found {len(gold)}.\n\n"
            "Do not deploy until the evaluation and deployment "
            "gold sets are identical."
        )

    return gold, duplicates_removed


# ============================================================
# LOAD PROCESSED REVIEWS
# ============================================================

def load_processed_reviews():

    frames = []

    for app_key, file_path in PROCESSED_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed file not found for {app_key}:\n"
                f"{file_path}"
            )

        df = pd.read_csv(file_path)

        required = {
            "reviewId",
            "score",
            "clean_text",
            "text_eligible",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing)}"
            )

        keep_columns = [
            "reviewId",
            "score",
            "clean_text",
            "text_eligible",
        ]

        optional_columns = [
            "normalized_text",
            "word_count",
            "char_count",
            "emoji_count",
            "is_emoji_only",
            "language",
            "script",
            "is_code_mixed",
            "is_romanized_indian",
            "week_start",
            "rating_valid",
        ]

        for column in optional_columns:

            if column in df.columns:
                keep_columns.append(column)

        df = df[
            keep_columns
        ].copy()

        df["app_key"] = app_key

        frames.append(df)

    reviews = pd.concat(
        frames,
        ignore_index=True,
    )

    reviews["reviewId"] = (
        reviews["reviewId"]
        .astype(str)
        .str.strip()
    )

    reviews["score"] = pd.to_numeric(
        reviews["score"],
        errors="coerce",
    )

    # Robust text_eligible conversion.
    if reviews["text_eligible"].dtype == object:

        reviews["text_eligible"] = (
            reviews["text_eligible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(False)
        )

    else:

        reviews["text_eligible"] = (
            reviews["text_eligible"]
            .fillna(False)
            .astype(bool)
        )

    reviews["clean_text"] = normalize_text(
        reviews["clean_text"]
    )

    # Candidate pool = 1–3 star + usable text.
    candidate_mask = (
        reviews["score"].between(
            1,
            3,
            inclusive="both",
        )
        & reviews["text_eligible"]
        & reviews["clean_text"].str.len().gt(0)
    )

    candidates = reviews.loc[
        candidate_mask
    ].copy()

    return candidates.reset_index(
        drop=True
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FINAL COMPLAINT CLASSIFIER DEPLOYMENT")
print("=" * 80)


# ============================================================
# LOAD GOLD SET
# ============================================================

gold, duplicates_removed = (
    load_final_gold_set()
)

print("\nFinal gold set:")

print(
    f"  Unique labeled reviews: "
    f"{len(gold):,}"
)

print(
    f"  Duplicate rows removed: "
    f"{duplicates_removed:,}"
)

print("\nGold-set labels:")

print(
    gold["human_label"]
    .value_counts()
    .to_string()
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "-" * 80)
print("TRAINING FINAL MODEL")
print("-" * 80)

print(
    f"\nModel: {BEST_MODEL}"
)

print(
    "Features: character TF-IDF "
    f"{VECTORIZER_CONFIG['ngram_range'][0]}–"
    f"{VECTORIZER_CONFIG['ngram_range'][1]} grams"
)

print(
    f"Deployment threshold: "
    f"{COMPLAINT_THRESHOLD:.2f}"
)

X_train = normalize_text(
    gold["review_text"]
)

y_train = gold["human_label"]


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    analyzer=VECTORIZER_CONFIG["analyzer"],
    ngram_range=VECTORIZER_CONFIG["ngram_range"],
    min_df=VECTORIZER_CONFIG["min_df"],
    max_features=VECTORIZER_CONFIG["max_features"],
    sublinear_tf=VECTORIZER_CONFIG["sublinear_tf"],
    lowercase=VECTORIZER_CONFIG["lowercase"],
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

print(
    f"Training rows: "
    f"{X_train_tfidf.shape[0]:,}"
)

print(
    f"TF-IDF features: "
    f"{X_train_tfidf.shape[1]:,}"
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

classifier = LogisticRegression(
    C=CLASSIFIER_CONFIG["C"],
    max_iter=CLASSIFIER_CONFIG["max_iter"],
    random_state=CLASSIFIER_CONFIG["random_state"],
)

classifier.fit(
    X_train_tfidf,
    y_train,
)

print("Model training complete.")


# ============================================================
# LOAD FULL CANDIDATE POOL
# ============================================================

print("\n" + "-" * 80)
print("LOADING FULL 1–3 STAR TEXT-ELIGIBLE POOL")
print("-" * 80)

candidates = load_processed_reviews()

print(
    f"\nCandidate reviews: "
    f"{len(candidates):,}"
)

print("\nCandidates by app:")

print(
    candidates["app_key"]
    .value_counts()
    .to_string()
)


# ============================================================
# CLASSIFY USING VALIDATED THRESHOLD
# ============================================================

print("\n" + "-" * 80)
print("CLASSIFYING CANDIDATE REVIEWS")
print("-" * 80)

X_candidates = normalize_text(
    candidates["clean_text"]
)

X_candidates_tfidf = vectorizer.transform(
    X_candidates
)

# Get class probabilities.
probabilities = classifier.predict_proba(
    X_candidates_tfidf
)

classes = list(
    classifier.classes_
)

complaint_index = classes.index(
    "complaint"
)

complaint_probability = probabilities[
    :,
    complaint_index
]

confidence = probabilities.max(
    axis=1
)


# ------------------------------------------------------------
# FINAL DECISION
# ------------------------------------------------------------

# A review is classified as a complaint ONLY when its
# complaint probability reaches the validated 0.75 threshold.
predicted_labels = (
    complaint_probability
    >= COMPLAINT_THRESHOLD
).astype(int)

predicted_labels = pd.Series(
    predicted_labels,
    index=candidates.index,
).map(
    {
        1: "complaint",
        0: "non_complaint",
    }
)


candidates["complaint_label"] = (
    predicted_labels
)

candidates["complaint_probability"] = (
    complaint_probability
)

candidates["complaint_confidence"] = (
    confidence
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

preferred_columns = [
    "reviewId",
    "app_key",
    "score",
    "clean_text",
    "complaint_label",
    "complaint_probability",
    "complaint_confidence",
]

remaining_columns = [
    column
    for column in candidates.columns
    if column not in preferred_columns
]

candidates = candidates[
    preferred_columns + remaining_columns
]

candidates.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

total = len(candidates)

complaints = int(
    (
        candidates["complaint_label"]
        == "complaint"
    ).sum()
)

non_complaints = int(
    (
        candidates["complaint_label"]
        == "non_complaint"
    ).sum()
)


summary_rows = []

for app_key, group in candidates.groupby(
    "app_key",
    sort=True,
):

    app_total = len(group)

    app_complaints = int(
        (
            group["complaint_label"]
            == "complaint"
        ).sum()
    )

    app_non_complaints = int(
        (
            group["complaint_label"]
            == "non_complaint"
        ).sum()
    )

    summary_rows.append(
        {
            "app_key": app_key,
            "candidate_reviews": app_total,
            "complaints": app_complaints,
            "non_complaints": app_non_complaints,
            "complaint_rate": (
                app_complaints / app_total
                if app_total
                else 0
            ),
        }
    )


summary_rows.append(
    {
        "app_key": "ALL",
        "candidate_reviews": total,
        "complaints": complaints,
        "non_complaints": non_complaints,
        "complaint_rate": (
            complaints / total
            if total
            else 0
        ),
    }
)


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE MODEL METADATA
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

metadata = {
    "model_name": BEST_MODEL,
    "complaint_threshold": COMPLAINT_THRESHOLD,
    "training_reviews": int(len(gold)),
    "training_complaints": int(
        (
            y_train
            == "complaint"
        ).sum()
    ),
    "training_non_complaints": int(
        (
            y_train
            == "non_complaint"
        ).sum()
    ),
    "validation_accuracy": 0.9266,
    "validation_complaint_precision": 0.9258,
    "validation_complaint_recall": 0.9863,
    "validation_complaint_f1": 0.9551,
    "validation_macro_f1": 0.8775,
    "deployment_audit_usable_reviews": 112,
    "deployment_audit_precision": 0.8966,
    "deployment_audit_recall": 0.9123,
    "deployment_audit_f1": 0.9043,
    "candidate_reviews": int(total),
    "predicted_complaints": int(
        complaints
    ),
    "predicted_non_complaints": int(
        non_complaints
    ),
    "vectorizer": VECTORIZER_CONFIG,
    "classifier": CLASSIFIER_CONFIG,
    "candidate_filter": (
        "score 1-3 AND text_eligible=True "
        "AND clean_text non-empty"
    ),
    "threshold_selection": (
        "Selected from deployment audit; "
        "0.75 produced the highest tested F1 "
        "of 90.43%"
    ),
}


with open(
    MODEL_METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("DEPLOYMENT RESULTS")
print("=" * 80)

print(
    f"\nTraining reviews: "
    f"{len(gold):,}"
)

print(
    f"Deployment threshold: "
    f"{COMPLAINT_THRESHOLD:.2f}"
)

print(
    f"Total candidate reviews: "
    f"{total:,}"
)

print(
    f"Predicted complaints:     "
    f"{complaints:,}"
)

print(
    f"Predicted non-complaints: "
    f"{non_complaints:,}"
)

if total:

    print(
        f"Overall complaint rate:   "
        f"{complaints / total * 100:.2f}%"
    )


print("\nBy app:")

display_summary = summary_df[
    summary_df["app_key"] != "ALL"
].copy()

display_summary["complaint_rate"] = (
    display_summary["complaint_rate"]
    * 100
).round(2)

print(
    display_summary.to_string(
        index=False
    )
)


# ============================================================
# FILES
# ============================================================

print("\n" + "-" * 80)
print("FILES SAVED")
print("-" * 80)

print("\nPredictions:")
print(OUTPUT_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("\nModel metadata:")
print(MODEL_METADATA_FILE)

print("\n" + "=" * 80)
print("COMPLAINT CLASSIFIER DEPLOYMENT COMPLETE")
print("=" * 80)