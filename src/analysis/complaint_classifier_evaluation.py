from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "validation"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

BATCH1_PATH = (
    VALIDATION_DIR
    / "complaint_filter_validation.csv"
)

BATCH2_PATH = (
    VALIDATION_DIR
    / "complaint_gold_set_batch2.csv"
)

SUMMARY_PATH = (
    ANALYSIS_DIR
    / "complaint_classifier_model_comparison.csv"
)

OOF_PATH = (
    ANALYSIS_DIR
    / "complaint_classifier_oof_predictions.csv"
)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_STATE = 42

N_SPLITS = 5

LABELS = [
    "non_complaint",
    "complaint",
]


# =========================================================
# LOAD ONE BATCH
# =========================================================

def load_batch(
    path: Path,
    batch_name: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{batch_name} file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "reviewId",
        "app_key",
        "review_text",
        "human_label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{batch_name} is missing columns: "
            f"{sorted(missing)}"
        )

    df = df[
        [
            "reviewId",
            "app_key",
            "score",
            "review_text",
            "human_label",
        ]
    ].copy()

    df[
        "batch"
    ] = batch_name

    return df


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(
    text: str,
) -> str:

    if pd.isna(text):

        return ""

    return " ".join(
        str(text)
        .lower()
        .split()
    )


# =========================================================
# LOAD + CLEAN GOLD DATA
# =========================================================

def load_gold_data() -> pd.DataFrame:

    batch1 = load_batch(
        BATCH1_PATH,
        "batch1",
    )

    batch2 = load_batch(
        BATCH2_PATH,
        "batch2",
    )

    df = pd.concat(
        [
            batch1,
            batch2,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Normalize labels.
    # -----------------------------------------------------

    df[
        "human_label"
    ] = (
        df["human_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_labels = {
        "complaint",
        "non_complaint",
        "ambiguous",
    }

    invalid = df[
        ~df["human_label"].isin(
            allowed_labels
        )
    ].copy()

    if not invalid.empty:

        raise ValueError(
            "Invalid human_label values found:\n"
            + str(
                invalid[
                    "human_label"
                ]
                .value_counts()
            )
        )

    # -----------------------------------------------------
    # Normalize text.
    # -----------------------------------------------------

    df[
        "text_key"
    ] = (
        df["review_text"]
        .apply(normalize_text)
    )

    # Remove empty text.
    df = df[
        df["text_key"]
        != ""
    ].copy()

    # -----------------------------------------------------
    # Exclude ambiguous labels from supervised learning.
    # -----------------------------------------------------

    df = df[
        df["human_label"]
        != "ambiguous"
    ].copy()

    # -----------------------------------------------------
    # Detect conflicting labels for identical text.
    #
    # Example:
    # same review text labeled complaint in one batch
    # and non_complaint in another.
    #
    # We do NOT silently choose one.
    # -----------------------------------------------------

    label_counts = (
        df.groupby(
            "text_key"
        )["human_label"]
        .nunique()
    )

    conflicting_texts = set(
        label_counts[
            label_counts > 1
        ].index
    )

    if conflicting_texts:

        print(
            "\nWARNING:"
        )

        print(
            f"Found "
            f"{len(conflicting_texts):,} "
            f"exact-text conflicts."
        )

        print(
            "Those conflicting texts will be excluded "
            "from classifier evaluation."
        )

        df = df[
            ~df["text_key"].isin(
                conflicting_texts
            )
        ].copy()

    # -----------------------------------------------------
    # Deduplicate exact text.
    #
    # Keep one human label per unique review text.
    # -----------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "text_key"
        ],
        keep="first",
    ).copy()

    # -----------------------------------------------------
    # Create leakage-resistant group signature.
    #
    # Exact duplicates are already removed.
    # This secondary signature groups reviews sharing
    # their first several normalized tokens.
    #
    # It is intentionally conservative so that obvious
    # template-like duplicates stay together.
    # -----------------------------------------------------

    def group_signature(
        text: str,
    ) -> str:

        tokens = text.split()

        # For short reviews, use the full text.
        if len(tokens) <= 5:

            return text

        # For longer reviews, group by the first 5 tokens.
        return " ".join(
            tokens[:5]
        )

    df[
        "group_id"
    ] = (
        df["text_key"]
        .apply(
            group_signature
        )
    )

    # -----------------------------------------------------
    # Sort for reproducibility.
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "group_id",
            "text_key",
        ]
    ).reset_index(
        drop=True
    )

    return df


# =========================================================
# VECTORISERS
# =========================================================

def build_word_vectorizer():

    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        lowercase=True,
        max_features=50000,
    )


def build_char_vectorizer():

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        lowercase=True,
        max_features=60000,
    )


def build_word_char_features():

    return FeatureUnion(
        [
            (
                "word",
                build_word_vectorizer(),
            ),
            (
                "char",
                build_char_vectorizer(),
            ),
        ]
    )


# =========================================================
# MODEL FACTORIES
# =========================================================

def build_word_logistic():

    return (
        build_word_vectorizer(),
        LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def build_char_logistic():

    return (
        build_char_vectorizer(),
        LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def build_word_char_logistic():

    return (
        build_word_char_features(),
        LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def build_word_char_svm():

    return (
        build_word_char_features(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


# =========================================================
# MODEL DEFINITIONS
# =========================================================

MODEL_BUILDERS = {
    "word_tfidf_logistic": (
        build_word_logistic
    ),
    "char_tfidf_logistic": (
        build_char_logistic
    ),
    "word_char_tfidf_logistic": (
        build_word_char_logistic
    ),
    "word_char_tfidf_svm": (
        build_word_char_svm
    ),
}


# =========================================================
# METRIC HELPER
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    # Matrix layout:
    #
    # [[TN, FP],
    #  [FN, TP]]
    #
    # because LABELS = [non_complaint, complaint]

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "complaint_precision": (
            precision_score(
                y_true,
                y_pred,
                pos_label="complaint",
                zero_division=0,
            )
        ),
        "complaint_recall": (
            recall_score(
                y_true,
                y_pred,
                pos_label="complaint",
                zero_division=0,
            )
        ),
        "complaint_f1": (
            f1_score(
                y_true,
                y_pred,
                pos_label="complaint",
                zero_division=0,
            )
        ),
        "macro_f1": (
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# =========================================================
# GROUP-AWARE FOLD BUILDER
# =========================================================

def build_folds(
    df: pd.DataFrame,
):

    # -----------------------------------------------------
    # We use a manual group-aware assignment.
    #
    # Each group contains near-template reviews based on
    # their first few normalized tokens.
    #
    # Groups are assigned to folds while trying to maintain
    # label balance.
    # -----------------------------------------------------

    group_df = (
        df.groupby(
            "group_id"
        )
        .agg(
            label=(
                "human_label",
                lambda x: x.iloc[0],
            ),
            size=(
                "human_label",
                "size",
            ),
        )
        .reset_index()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    fold_groups = [
        [] for _ in range(
            N_SPLITS
        )
    ]

    fold_sizes = np.zeros(
        N_SPLITS,
        dtype=int,
    )

    # Prioritize largest groups first.
    group_df = group_df.sort_values(
        "size",
        ascending=False,
    ).reset_index(
        drop=True
    )

    for _, row in group_df.iterrows():

        candidate_folds = list(
            range(N_SPLITS)
        )

        # Small random tie-breaker.
        rng.shuffle(
            candidate_folds
        )

        # Prefer the smallest current fold
        # among folds containing the same class.
        same_label_sizes = []

        for fold_idx in candidate_folds:

            current_size = sum(
                group_df.loc[
                    group_df["group_id"].isin(
                        fold_groups[
                            fold_idx
                        ]
                    )
                    &
                    (
                        group_df["label"]
                        == row["label"]
                    ),
                    "size",
                ]
            )

            same_label_sizes.append(
                (
                    current_size,
                    fold_sizes[
                        fold_idx
                    ],
                    fold_idx,
                )
            )

        same_label_sizes.sort()

        selected_fold = (
            same_label_sizes[0][2]
        )

        fold_groups[
            selected_fold
        ].append(
            row["group_id"]
        )

        fold_sizes[
            selected_fold
        ] += int(
            row["size"]
        )

    # -----------------------------------------------------
    # Convert group assignment into row indices.
    # -----------------------------------------------------

    folds = []

    for fold_idx in range(
        N_SPLITS
    ):

        test_groups = set(
            fold_groups[
                fold_idx
            ]
        )

        test_idx = np.where(
            df["group_id"]
            .isin(test_groups)
            .to_numpy()
        )[0]

        train_idx = np.where(
            ~df["group_id"]
            .isin(test_groups)
            .to_numpy()
        )[0]

        folds.append(
            (
                train_idx,
                test_idx,
            )
        )

    return folds


# =========================================================
# VALIDATE FOLD LABEL DISTRIBUTION
# =========================================================

def validate_folds(
    df: pd.DataFrame,
    folds,
) -> None:

    print(
        "\n"
        + "=" * 95
    )

    print(
        "GROUPED CROSS-VALIDATION FOLDS"
    )

    print(
        "=" * 95
    )

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        train_labels = (
            df.iloc[
                train_idx
            ]["human_label"]
            .value_counts()
        )

        test_labels = (
            df.iloc[
                test_idx
            ]["human_label"]
            .value_counts()
        )

        print(
            f"\nFold {fold_number}:"
        )

        print(
            f"  Train: "
            f"{len(train_idx):,}"
        )

        print(
            f"    complaint="
            f"{int(train_labels.get('complaint', 0)):,}"
            f", non_complaint="
            f"{int(train_labels.get('non_complaint', 0)):,}"
        )

        print(
            f"  Test: "
            f"{len(test_idx):,}"
        )

        print(
            f"    complaint="
            f"{int(test_labels.get('complaint', 0)):,}"
            f", non_complaint="
            f"{int(test_labels.get('non_complaint', 0)):,}"
        )

        if (
            len(
                test_labels
            )
            < 2
        ):

            raise RuntimeError(
                f"Fold {fold_number} "
                "contains only one class. "
                "Cannot perform binary evaluation."
            )


# =========================================================
# RUN ONE MODEL
# =========================================================

def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    folds,
) -> tuple[
    dict,
    pd.DataFrame,
]:

    builder = MODEL_BUILDERS[
        model_name
    ]

    fold_predictions = []

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        train_df = df.iloc[
            train_idx
        ]

        test_df = df.iloc[
            test_idx
        ]

        y_train = (
            train_df[
                "human_label"
            ]
        )

        y_test = (
            test_df[
                "human_label"
            ]
        )

        train_text = (
            train_df[
                "review_text"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        test_text = (
            test_df[
                "review_text"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        vectorizer, classifier = (
            builder()
        )

        # -------------------------------------------------
        # Fit vectorizer only on training fold.
        # -------------------------------------------------

        X_train = (
            vectorizer.fit_transform(
                train_text
            )
        )

        X_test = (
            vectorizer.transform(
                test_text
            )
        )

        # -------------------------------------------------
        # Train classifier.
        # -------------------------------------------------

        classifier.fit(
            X_train,
            y_train,
        )

        # -------------------------------------------------
        # Predict.
        # -------------------------------------------------

        y_pred = (
            classifier.predict(
                X_test
            )
        )

        fold_result = test_df[
            [
                "reviewId",
                "app_key",
                "score",
                "review_text",
                "human_label",
            ]
        ].copy()

        fold_result[
            "model"
        ] = model_name

        fold_result[
            "fold"
        ] = fold_number

        fold_result[
            "predicted_label"
        ] = y_pred

        fold_predictions.append(
            fold_result
        )

    predictions = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    metrics = calculate_metrics(
        predictions[
            "human_label"
        ],
        predictions[
            "predicted_label"
        ],
    )

    metrics[
        "model"
    ] = model_name

    metrics[
        "n_reviews"
    ] = len(
        predictions
    )

    metrics[
        "n_folds"
    ] = N_SPLITS

    return (
        metrics,
        predictions,
    )


# =========================================================
# PER-APP METRICS
# =========================================================

def calculate_app_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for app_key in sorted(
        predictions[
            "app_key"
        ].unique()
    ):

        app_df = predictions[
            predictions["app_key"]
            == app_key
        ]

        metrics = calculate_metrics(
            app_df[
                "human_label"
            ],
            app_df[
                "predicted_label"
            ],
        )

        rows.append(
            {
                "app_key": app_key,
                **metrics,
                "n_reviews": len(
                    app_df
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PRINT MODEL RESULTS
# =========================================================

def print_model_results(
    summary: pd.DataFrame,
    app_metrics: dict[str, pd.DataFrame],
) -> None:

    print(
        "\n"
        + "=" * 95
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 95
    )

    display_columns = [
        "model",
        "accuracy",
        "complaint_precision",
        "complaint_recall",
        "complaint_f1",
        "macro_f1",
        "false_positive",
        "false_negative",
    ]

    display = summary[
        display_columns
    ].copy()

    for column in [
        "accuracy",
        "complaint_precision",
        "complaint_recall",
        "complaint_f1",
        "macro_f1",
    ]:

        display[
            column
        ] = (
            display[
                column
            ]
            * 100
        ).round(
            2
        )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "PER-APP PERFORMANCE"
    )

    print(
        "=" * 95
    )

    for model_name, metrics_df in (
        app_metrics.items()
    ):

        print(
            f"\n{model_name}"
        )

        display_columns = [
            "app_key",
            "accuracy",
            "complaint_precision",
            "complaint_recall",
            "complaint_f1",
            "macro_f1",
            "n_reviews",
        ]

        display = metrics_df[
            display_columns
        ].copy()

        for column in [
            "accuracy",
            "complaint_precision",
            "complaint_recall",
            "complaint_f1",
            "macro_f1",
        ]:

            display[
                column
            ] = (
                display[
                    column
                ]
                * 100
            ).round(
                2
            )

        print(
            display.to_string(
                index=False
            )
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "COMPLAINT CLASSIFIER EVALUATION"
    )

    print(
        "=" * 95
    )

    # -----------------------------------------------------
    # Load gold data.
    # -----------------------------------------------------

    df = load_gold_data()

    print(
        f"\nUnique labeled reviews: "
        f"{len(df):,}"
    )

    print(
        "\nLabels:"
    )

    print(
        df[
            "human_label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nApps:"
    )

    print(
        df[
            "app_key"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -----------------------------------------------------
    # Build folds.
    # -----------------------------------------------------

    folds = build_folds(
        df
    )

    validate_folds(
        df,
        folds,
    )

    # -----------------------------------------------------
    # Evaluate each model.
    # -----------------------------------------------------

    all_metrics = []

    all_predictions = []

    app_metrics = {}

    for model_name in MODEL_BUILDERS:

        print(
            "\n"
            + "-" * 95
        )

        print(
            f"Evaluating: {model_name}"
        )

        print(
            "-" * 95
        )

        metrics, predictions = (
            evaluate_model(
                model_name,
                df,
                folds,
            )
        )

        all_metrics.append(
            metrics
        )

        all_predictions.append(
            predictions
        )

        app_metrics[
            model_name
        ] = calculate_app_metrics(
            predictions
        )

    # -----------------------------------------------------
    # Combine results.
    # -----------------------------------------------------

    summary = pd.DataFrame(
        all_metrics
    )

    # Rank primarily by complaint F1,
    # then macro F1, then complaint precision.
    summary = summary.sort_values(
        [
            "complaint_f1",
            "macro_f1",
            "complaint_precision",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "rank",
        range(
            1,
            len(summary) + 1,
        ),
    )

    oof_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Save.
    # -----------------------------------------------------

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    oof_predictions.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print.
    # -----------------------------------------------------

    print_model_results(
        summary,
        app_metrics,
    )

    # -----------------------------------------------------
    # Winner.
    # -----------------------------------------------------

    winner = summary.iloc[
        0
    ]

    print(
        "\n"
        + "=" * 95
    )

    print(
        "CURRENT WINNER"
    )

    print(
        "=" * 95
    )

    print(
        f"\nModel: "
        f"{winner['model']}"
    )

    print(
        f"Complaint precision: "
        f"{winner['complaint_precision'] * 100:.2f}%"
    )

    print(
        f"Complaint recall: "
        f"{winner['complaint_recall'] * 100:.2f}%"
    )

    print(
        f"Complaint F1: "
        f"{winner['complaint_f1'] * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{winner['macro_f1'] * 100:.2f}%"
    )

    print(
        f"False positives: "
        f"{int(winner['false_positive'])}"
    )

    print(
        f"False negatives: "
        f"{int(winner['false_negative'])}"
    )

    # -----------------------------------------------------
    # Files.
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 95
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 95
    )

    print(
        f"\nModel comparison:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nOut-of-fold predictions:\n"
        f"{OOF_PATH}"
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "COMPLAINT CLASSIFIER EVALUATION COMPLETE"
    )

    print(
        "=" * 95
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()