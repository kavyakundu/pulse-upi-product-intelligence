from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "validation"
)

INPUT_PATH = (
    ANALYSIS_DIR
    / "final_pain_category_assignments.csv"
)

OLD_VALIDATION_PATH = (
    VALIDATION_DIR
    / "complaint_filter_validation.csv"
)

OUTPUT_PATH = (
    VALIDATION_DIR
    / "complaint_gold_set_batch2.csv"
)


# =========================================================
# SETTINGS
# =========================================================

TARGET_TOTAL = 300

TARGET_PER_APP = 100

RANDOM_STATE = 42


# We deliberately create several groups.
#
# 1. clear mapped pain:
#    reviews that already belong to a mapped category
#
# 2. clear non-mapped controls:
#    reviews excluded from mapped pain categories
#
# 3. hard mapped:
#    mapped reviews with weaker topic assignment
#
# 4. hard control:
#    non-mapped reviews that nevertheless contain
#    complaint-like language
#
# This is much more useful than a purely random sample.
# =========================================================

GROUP_TARGETS_PER_APP = {
    "mapped_clear": 35,
    "nonmapped_clear": 25,
    "mapped_hard": 20,
    "nonmapped_hard": 20,
}


# =========================================================
# LOAD DATA
# =========================================================

def load_assignments() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Assignments file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "normalized_text",
        "topic_id",
        "topic_strength",
        "decision",
        "final_category",
        "is_mapped_pain",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Assignments file is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# =========================================================
# TEXT HELPERS
# =========================================================

def prepare_text(
    text: str,
) -> str:

    if pd.isna(text):

        return ""

    return " ".join(
        str(text).split()
    )


def word_count(
    text: str,
) -> int:

    text = prepare_text(
        text
    )

    if not text:

        return 0

    return len(
        text.split()
    )


# =========================================================
# DUPLICATE HANDLING
# =========================================================

def create_unique_review_key(
    text: str,
) -> str:

    text = prepare_text(
        text
    )

    return (
        text
        .strip()
        .lower()
    )


def remove_duplicate_text(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df[
        "review_key"
    ] = (
        df["normalized_text"]
        .apply(
            create_unique_review_key
        )
    )

    df = df[
        df["review_key"]
        != ""
    ].copy()

    # Keep only one copy of identical text.
    df = df.drop_duplicates(
        subset=[
            "review_key"
        ],
        keep="first",
    )

    return df


# =========================================================
# COMPLAINT-LIKE SIGNALS
# =========================================================

COMPLAINT_SIGNALS = [
    "not working",
    "does not work",
    "doesn't work",
    "not open",
    "does not open",
    "doesn't open",
    "unable",
    "can't",
    "cannot",
    "problem",
    "issue",
    "error",
    "failed",
    "failure",
    "pending",
    "stuck",
    "deducted",
    "debited",
    "refund",
    "cashback",
    "cash back",
    "reward",
    "support",
    "customer care",
    "customer service",
    "no response",
    "not received",
    "didn't receive",
    "not getting",
    "slow",
    "crash",
    "crashing",
    "privacy",
    "fraud",
    "scam",
    "security",
    "update",
    "interface",
    "dark mode",
    "upi lite",
    "bank account",
    "verification",
    "otp",
    "network",
    "server",
    "recharge",
    "login",
    "account",
    "money",
    "payment",
    "transaction",
    "पैसा",
    "पेमेंट",
    "भुगतान",
    "समस्या",
    "शिकायत",
    "कैशबैक",
    "रिवॉर्ड",
    "अपडेट",
]


POSITIVE_SIGNALS = [
    "good",
    "very good",
    "best app",
    "best application",
    "great app",
    "excellent app",
    "nice app",
    "wonderful app",
    "amazing app",
    "happy",
    "super",
    "excellent service",
    "good service",
    "very good service",
    "best service",
    "बहुत अच्छा",
    "अच्छा ऐप",
]


# =========================================================
# SIGNAL COUNTS
# =========================================================

def count_signals(
    text: str,
    signals: list[str],
) -> int:

    text_lower = text.lower()

    count = 0

    for signal in signals:

        if signal.lower() in text_lower:

            count += 1

    return count


# =========================================================
# BUILD SAMPLING FEATURES
# =========================================================

def add_sampling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["review_text"] = (
        df["normalized_text"]
        .apply(prepare_text)
    )

    df["word_count"] = (
        df["review_text"]
        .apply(word_count)
    )

    df["complaint_signal_count"] = (
        df["review_text"]
        .apply(
            lambda x: count_signals(
                x,
                COMPLAINT_SIGNALS,
            )
        )
    )

    df["positive_signal_count"] = (
        df["review_text"]
        .apply(
            lambda x: count_signals(
                x,
                POSITIVE_SIGNALS,
            )
        )
    )

    return df


# =========================================================
# SAMPLE GROUPS
# =========================================================

def make_sampling_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # -----------------------------------------------------
    # Clear mapped pain:
    # strong topic assignment + meaningful text
    # -----------------------------------------------------

    clear_mapped = df[
        (df["is_mapped_pain"] == True)
        &
        (df["topic_strength"] >= 0.08)
        &
        (df["word_count"] >= 7)
    ].copy()

    # -----------------------------------------------------
    # Hard mapped:
    # weaker topic assignment and/or shorter text
    # -----------------------------------------------------

    hard_mapped = df[
        (df["is_mapped_pain"] == True)
        &
        ~df["reviewId"].isin(
            clear_mapped["reviewId"]
        )
    ].copy()

    # -----------------------------------------------------
    # Clear non-mapped:
    # no complaint-like language
    # -----------------------------------------------------

    clear_nonmapped = df[
        (df["is_mapped_pain"] != True)
        &
        (df["complaint_signal_count"] == 0)
        &
        (df["positive_signal_count"] >= 1)
    ].copy()

    # -----------------------------------------------------
    # Hard non-mapped:
    # excluded by mapping but contains complaint-like
    # language.
    # -----------------------------------------------------

    hard_nonmapped = df[
        (df["is_mapped_pain"] != True)
        &
        ~df["reviewId"].isin(
            clear_nonmapped["reviewId"]
        )
    ].copy()

    # Make sampling group explicit.
    parts = []

    for part, group_name in [
        (clear_mapped, "mapped_clear"),
        (clear_nonmapped, "nonmapped_clear"),
        (hard_mapped, "mapped_hard"),
        (hard_nonmapped, "nonmapped_hard"),
    ]:

        part = part.copy()

        part[
            "sampling_group"
        ] = group_name

        parts.append(
            part
        )

    grouped = pd.concat(
        parts,
        ignore_index=True,
    )

    # Because a row should belong to one group only,
    # remove accidental duplicate review IDs.
    grouped = grouped.drop_duplicates(
        subset=[
            "reviewId"
        ],
        keep="first",
    )

    return grouped


# =========================================================
# SAMPLE PER APP
# =========================================================

def sample_app(
    app_df: pd.DataFrame,
    app_key: str,
) -> pd.DataFrame:

    sampled_parts = []

    used_ids = set()

    for group_name, target in (
        GROUP_TARGETS_PER_APP.items()
    ):

        group_df = app_df[
            app_df["sampling_group"]
            == group_name
        ].copy()

        group_df = group_df[
            ~group_df["reviewId"].isin(
                used_ids
            )
        ]

        if group_df.empty:

            continue

        # Prefer a spread of star ratings.
        rating_parts = []

        ratings = sorted(
            group_df[
                "score"
            ].dropna()
            .unique()
            .tolist()
        )

        if ratings:

            per_rating = max(
                1,
                target // len(ratings),
            )

            for rating in ratings:

                rating_df = group_df[
                    group_df["score"]
                    == rating
                ]

                if rating_df.empty:

                    continue

                take = min(
                    per_rating,
                    len(rating_df),
                )

                rating_sample = (
                    rating_df.sample(
                        n=take,
                        random_state=(
                            RANDOM_STATE
                            + hash(
                                (
                                    app_key,
                                    group_name,
                                    rating,
                                )
                            )
                            % 10000
                        ),
                    )
                )

                rating_parts.append(
                    rating_sample
                )

            candidate = pd.concat(
                rating_parts,
                ignore_index=True,
            ) if rating_parts else group_df

        else:

            candidate = group_df

        # Fill remaining target places.
        if len(candidate) < target:

            remaining_pool = group_df[
                ~group_df["reviewId"].isin(
                    candidate["reviewId"]
                )
            ]

            remaining_n = min(
                target - len(candidate),
                len(remaining_pool),
            )

            if remaining_n > 0:

                extra = (
                    remaining_pool.sample(
                        n=remaining_n,
                        random_state=(
                            RANDOM_STATE
                            + 100
                            + len(candidate)
                        ),
                    )
                )

                candidate = pd.concat(
                    [
                        candidate,
                        extra,
                    ],
                    ignore_index=True,
                )

        # Never exceed target.
        candidate = candidate.head(
            target
        )

        sampled_parts.append(
            candidate
        )

        used_ids.update(
            candidate[
                "reviewId"
            ].tolist()
        )

    if not sampled_parts:

        return pd.DataFrame()

    result = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    # If group availability prevented reaching 100,
    # fill from any remaining unique reviews in this app.
    remaining_target = (
        TARGET_PER_APP
        - len(result)
    )

    if remaining_target > 0:

        remaining_pool = app_df[
            ~app_df["reviewId"].isin(
                result["reviewId"]
            )
        ].copy()

        if not remaining_pool.empty:

            extra = remaining_pool.sample(
                n=min(
                    remaining_target,
                    len(remaining_pool),
                ),
                random_state=(
                    RANDOM_STATE
                    + 500
                ),
            )

            extra = extra.copy()

            extra[
                "sampling_group"
            ] = "supplemental"

            result = pd.concat(
                [
                    result,
                    extra,
                ],
                ignore_index=True,
            )

    return result.head(
        TARGET_PER_APP
    )


# =========================================================
# REMOVE REVIEWS ALREADY LABELED
# =========================================================

def remove_old_validation_reviews(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if not OLD_VALIDATION_PATH.exists():

        return df.copy()

    old = pd.read_csv(
        OLD_VALIDATION_PATH
    )

    if "reviewId" in old.columns:

        old_ids = set(
            old["reviewId"]
            .astype(str)
            .tolist()
        )

        df = df[
            ~df["reviewId"]
            .astype(str)
            .isin(old_ids)
        ].copy()

    return df


# =========================================================
# MAIN SAMPLE BUILDER
# =========================================================

def build_gold_set(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # Normalize IDs to strings.
    df = df.copy()

    df[
        "reviewId"
    ] = df[
        "reviewId"
    ].astype(str)

    # Remove text duplicates.
    df = remove_duplicate_text(
        df
    )

    # Remove all reviews already in batch 1.
    df = remove_old_validation_reviews(
        df
    )

    # Add sampling features.
    df = add_sampling_features(
        df
    )

    # Construct deliberate sampling groups.
    grouped = make_sampling_groups(
        df
    )

    app_results = []

    for app_key in [
        "phonepe",
        "paytm",
        "googlepay",
    ]:

        app_df = grouped[
            grouped["app_key"]
            == app_key
        ].copy()

        sampled = sample_app(
            app_df,
            app_key,
        )

        if not sampled.empty:

            app_results.append(
                sampled
            )

    if not app_results:

        raise RuntimeError(
            "Could not construct a gold-set sample."
        )

    result = pd.concat(
        app_results,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Final shuffle.
    # -----------------------------------------------------

    result = result.sample(
        frac=1.0,
        random_state=(
            RANDOM_STATE
            + 999
        ),
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Validation ID.
    # -----------------------------------------------------

    result.insert(
        0,
        "validation_id",
        range(
            121,
            121 + len(result),
        ),
    )

    # -----------------------------------------------------
    # Empty human labels.
    # -----------------------------------------------------

    result[
        "human_label"
    ] = ""

    result[
        "label_reason"
    ] = ""

    # -----------------------------------------------------
    # Select manual-review columns.
    # -----------------------------------------------------

    output = result[
        [
            "validation_id",
            "reviewId",
            "app_key",
            "score",
            "sampling_group",
            "final_category",
            "topic_id",
            "topic_strength",
            "review_text",
            "human_label",
            "label_reason",
        ]
    ].copy()

    return output


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    sample: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 95
    )

    print(
        "COMPLAINT GOLD SET — BATCH 2"
    )

    print(
        "=" * 95
    )

    print(
        f"\nTotal reviews: "
        f"{len(sample):,}"
    )

    print(
        "\nReviews by app:"
    )

    print(
        sample[
            "app_key"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nReviews by sampling group:"
    )

    print(
        sample[
            "sampling_group"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nReviews by star rating:"
    )

    print(
        sample[
            "score"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "MANUAL LABELS"
    )

    print(
        "=" * 95
    )

    print(
        "\nUse exactly one of:"
    )

    print(
        "  complaint"
    )

    print(
        "  non_complaint"
    )

    print(
        "  ambiguous"
    )

    print(
        "\nImportant:"
    )

    print(
        "Judge the REVIEW TEXT, not the star rating."
    )

    print(
        "\nA negative review without a concrete "
        "product problem can still be non_complaint."
    )

    print(
        "\nA review that is too unclear to understand "
        "should be ambiguous."
    )

    print(
        "\nDo not change any other columns."
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_assignments()

    sample = build_gold_set(
        df
    )

    if sample.empty:

        raise RuntimeError(
            "Gold-set sample is empty."
        )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        sample
    )

    print(
        "\n"
        + "=" * 95
    )

    print(
        "GOLD SET BATCH 2 CREATED"
    )

    print(
        "=" * 95
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\nNext:"
    )

    print(
        "Open the CSV in Excel and fill only "
        "'human_label' and 'label_reason'."
    )

    print(
        "\nBatch 2 is designed to contain "
        "hard examples specifically chosen to expose "
        "semantic classification failures."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()