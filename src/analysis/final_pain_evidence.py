from pathlib import Path
import re

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

INPUT_PATH = (
    ANALYSIS_DIR
    / "final_pain_category_assignments.csv"
)

SCORES_PATH = (
    ANALYSIS_DIR
    / "product_pain_scores.csv"
)

OUTPUT_PATH = (
    ANALYSIS_DIR
    / "final_pain_evidence.csv"
)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "phonepe",
    "paytm",
    "googlepay",
]

REVIEWS_PER_CATEGORY = 5

MIN_TOPIC_STRENGTH = 0.05

MIN_EVIDENCE_WORDS = 5

# Minimum number of alphabetic characters.
# Helps reject badly corrupted / punctuation-heavy reviews.
MIN_ALPHA_CHARS = 8

# Minimum alphabetic-character ratio.
# Very low ratios are usually malformed or non-textual.
MIN_ALPHA_RATIO = 0.45


# =========================================================
# CATEGORY COMPLAINT SIGNALS
# =========================================================
#
# A review must contain at least one category-relevant
# complaint signal to qualify as primary evidence.
#
# These are intentionally broad because the dataset contains
# English, Hinglish, Romanized Indian languages, and
# Indian-script reviews.
# =========================================================

CATEGORY_SIGNALS = {

    "App reliability & access": [
        "not working",
        "does not work",
        "doesn't work",
        "not open",
        "does not open",
        "doesn't open",
        "unable to open",
        "can't open",
        "cannot open",
        "unable to use",
        "can't use",
        "cannot use",
        "crash",
        "crashing",
        "crashed",
        "hang",
        "hanging",
        "stuck",
        "login",
        "log in",
        "sign in",
        "access",
        "blocked",
        "error",
        "issue",
        "problem",
        "not opening",
        "open nahi",
        "chal nahi",
        "nahi chal",
        "khul nahi",
        "login nahi",
        "काम नहीं",
        "नहीं चल",
        "नहीं खुल",
        "लॉगिन",
        "एरर",
        "समस्या",
    ],

    "Customer support": [
        "customer care",
        "customer support",
        "customer service",
        "support team",
        "support",
        "complaint",
        "complaints",
        "ticket",
        "tickets",
        "response",
        "respond",
        "reply",
        "reponse",
        "call",
        "helpline",
        "chat",
        "live agent",
        "refund",
        "resolve",
        "resolution",
        "help",
        "no response",
        "no reply",
        "care service",
        "customer care",
        "कस्टमर",
        "ग्राहक सेवा",
        "शिकायत",
        "टिकट",
        "जवाब",
        "रिफंड",
        "समाधान",
    ],

    "Performance & speed": [
        "slow",
        "slowness",
        "very slow",
        "too slow",
        "loading",
        "load",
        "lag",
        "lagging",
        "hang",
        "hanging",
        "freeze",
        "freezing",
        "delay",
        "delayed",
        "takes too long",
        "time to open",
        "slow app",
        "slowly",
        "धीमा",
        "स्लो",
        "लोड",
        "लेट",
        "देरी",
    ],

    "Transaction & payment issues": [
        "payment",
        "payments",
        "transaction",
        "transactions",
        "failed",
        "failure",
        "pending",
        "stuck",
        "debited",
        "debit",
        "credited",
        "credit",
        "money",
        "amount",
        "transfer",
        "refund",
        "recharge",
        "balance",
        "merchant",
        "upi payment",
        "payment issue",
        "payment problem",
        "transaction issue",
        "transaction problem",
        "पेमेन्ट",
        "पेमेंट",
        "ट्रांजेक्शन",
        "लेन देन",
        "पैसा",
        "रिफंड",
        "रिचार्ज",
        "भुगतान",
    ],

    "Cashback & rewards": [
        "cashback",
        "cash back",
        "reward",
        "rewards",
        "reward points",
        "cash back offer",
        "cashback offer",
        "offer",
        "offers",
        "coupon",
        "coupons",
        "redeem",
        "redeems",
        "scratch card",
        "scratchcard",
        "bonus",
        "points",
        "cashback not",
        "no cashback",
        "didn't get cashback",
        "not receive cashback",
        "reward not",
        "cashback nahi",
        "cash back nahi",
        "रिवॉर्ड",
        "कैशबैक",
        "ऑफर",
        "कूपन",
    ],

    "Updates & UI": [
        "update",
        "updated",
        "new update",
        "latest update",
        "ui",
        "interface",
        "user interface",
        "design",
        "dark mode",
        "dark theme",
        "navigation",
        "icon",
        "screen",
        "layout",
        "version",
        "ads",
        "advertisement",
        "new ui",
        "old ui",
        "old version",
        "अपडेट",
        "इंटरफेस",
        "डार्क मोड",
        "डिजाइन",
        "नेविगेशन",
    ],

    "Device & security compatibility": [
        "61007",
        "device integrity",
        "integrity check",
        "root access",
        "system modification",
        "root detected",
        "device detected",
        "security warning",
        "device warning",
        "compatibility",
        "phone software update",
        "software update",
        "system update",
        "android",
        "root",
        "warning",
        "device",
        "डिवाइस",
        "रूट",
        "चेतावनी",
    ],

    "Fraud, trust & security": [
        "fraud",
        "scam",
        "scammed",
        "cheat",
        "cheated",
        "cheating",
        "fraud app",
        "scam app",
        "fake",
        "fraudulent",
        "stolen",
        "money gone",
        "lost money",
        "suspicious",
        "trust",
        "unsafe",
        "risk",
        "धोखा",
        "फ्रॉड",
        "स्कैम",
        "चोरी",
    ],

    "Bank account & linking": [
        "bank account",
        "bank",
        "account",
        "link bank",
        "linking",
        "linked",
        "relink",
        "add bank",
        "remove bank",
        "bank account link",
        "bank account linking",
        "account linked",
        "account not linked",
        "bank not linked",
        "bank balance",
        "bank account issue",
        "bank account problem",
        "बैंक",
        "खाता",
        "बैंक अकाउंट",
        "लिंक",
    ],

    "UPI & UPI Lite": [
        "upi",
        "upi lite",
        "upi pin",
        "pin",
        "upi id",
        "upi account",
        "receive money",
        "send money",
        "upi payment",
        "upi problem",
        "upi issue",
        "यूपीआई",
        "यूपीआई लाइट",
        "पिन",
        "पैसा प्राप्त",
    ],

    "Network, server & OTP issues": [
        "network",
        "server",
        "server issue",
        "server problem",
        "network issue",
        "network problem",
        "connection",
        "connectivity",
        "offline",
        "otp",
        "otp problem",
        "otp issue",
        "sms",
        "sms problem",
        "verification",
        "internet",
        "नेटवर्क",
        "सर्वर",
        "ओटीपी",
        "एसएमएस",
        "कनेक्शन",
    ],
}


# =========================================================
# POSITIVE / GENERIC PHRASES
# =========================================================

POSITIVE_PHRASES = [
    "very good app",
    "good app",
    "best app",
    "best application",
    "very good",
    "good application",
    "great app",
    "great application",
    "excellent app",
    "excellent application",
    "nice app",
    "nice application",
    "very nice",
    "good service",
    "very good service",
    "best service",
    "excellent service",
    "awesome app",
    "amazing app",
    "love this app",
    "i love this app",
    "good payment app",
    "best payment app",
    "best app for payment",
    "good app for payment",
    "best upi app",
    "good upi app",
    "सबसे अच्छा",
    "बहुत अच्छा",
    "अच्छा ऐप",
    "बेस्ट ऐप",
    "अच्छी सर्विस",
]


GENERIC_NEGATIVE_PHRASES = [
    "worst app",
    "worst application",
    "bad app",
    "bad application",
    "poor app",
    "poor application",
    "very bad",
    "very poor",
    "third class app",
    "bekar app",
    "bekaar app",
    "ghatiya app",
    "घटिया ऐप",
    "बेकार ऐप",
    "बकवास ऐप",
    "worst",
]


# =========================================================
# LOAD DATA
# =========================================================

def load_assignments() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Pain-category assignments not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "reviewId",
        "app_key",
        "score",
        "at",
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
            f"Assignments file is missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_scores() -> pd.DataFrame:

    if not SCORES_PATH.exists():

        raise FileNotFoundError(
            f"Product pain scores not found:\n"
            f"{SCORES_PATH}"
        )

    df = pd.read_csv(
        SCORES_PATH
    )

    required_columns = {
        "app_key",
        "category",
        "product_pain_score",
        "prevalence_pct",
        "severity_1_2_within_category_pct",
        "strong_assignment_share_pct",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Product pain score file is missing columns: "
            f"{sorted(missing)}"
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


def count_words(
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


def clean_display_text(
    text: str,
    max_length: int = 700,
) -> str:

    text = prepare_text(
        text
    )

    if len(text) > max_length:

        return (
            text[:max_length]
            + "..."
        )

    return text


# =========================================================
# TEXT QUALITY CHECKS
# =========================================================

def alphabetic_character_ratio(
    text: str,
) -> float:

    if not text:

        return 0.0

    alpha_chars = sum(
        char.isalpha()
        for char in text
    )

    return (
        alpha_chars
        / max(len(text), 1)
    )


def has_meaningful_text(
    text: str,
) -> bool:

    if not text:

        return False

    alpha_chars = sum(
        char.isalpha()
        for char in text
    )

    alpha_ratio = (
        alphabetic_character_ratio(
            text
        )
    )

    if alpha_chars < MIN_ALPHA_CHARS:

        return False

    if alpha_ratio < MIN_ALPHA_RATIO:

        return False

    # Reject obvious repeated-gibberish patterns.
    compact = re.sub(
        r"[^a-zA-Z]+",
        "",
        text.lower(),
    )

    if len(compact) >= 8:

        unique_chars = len(
            set(compact)
        )

        if (
            unique_chars <= 2
            and len(compact) >= 8
        ):

            return False

    return True


# =========================================================
# CATEGORY SIGNAL CHECK
# =========================================================

def contains_category_signal(
    text: str,
    category: str,
) -> bool:

    text_lower = text.lower()

    signals = (
        CATEGORY_SIGNALS
        .get(category, [])
    )

    for signal in signals:

        if signal.lower() in text_lower:

            return True

    return False


# =========================================================
# POSITIVE / GENERIC CHECK
# =========================================================

def looks_like_positive_generic(
    text: str,
) -> bool:

    text_lower = text.lower()

    # A very short generic positive review should
    # never be used as pain evidence.
    for phrase in POSITIVE_PHRASES:

        if (
            phrase.lower()
            == text_lower.strip()
        ):

            return True

    positive_matches = 0

    for phrase in POSITIVE_PHRASES:

        if phrase.lower() in text_lower:

            positive_matches += 1

    negative_matches = 0

    for phrase in GENERIC_NEGATIVE_PHRASES:

        if phrase.lower() in text_lower:

            negative_matches += 1

    # Strong positive language with no explicit
    # negative signal is not useful pain evidence.
    if (
        positive_matches > 0
        and negative_matches == 0
        and not any(
            word in text_lower
            for word in [
                "not",
                "no ",
                "can't",
                "cannot",
                "unable",
                "problem",
                "issue",
                "failed",
                "failure",
                "pending",
                "slow",
                "missing",
                "didn't",
                "dont",
                "don't",
                "never",
            ]
        )
    ):

        return True

    return False


# =========================================================
# EVIDENCE ELIGIBILITY
# =========================================================

def is_evidence_candidate(
    row: pd.Series,
    category: str,
) -> bool:

    text = prepare_text(
        row["normalized_text"]
    )

    if not has_meaningful_text(
        text
    ):

        return False

    word_count = count_words(
        text
    )

    if word_count < MIN_EVIDENCE_WORDS:

        return False

    if looks_like_positive_generic(
        text
    ):

        return False

    if not contains_category_signal(
        text,
        category,
    ):

        return False

    return True


# =========================================================
# DETERMINE EVIDENCE QUALITY
# =========================================================

def evidence_quality(
    row: pd.Series,
) -> float:
    """
    Higher score = better evidence.

    We reward:
    - stronger topic assignment
    - informative review length
    - severity
    - category-specific complaint signal
    - overall text quality
    """

    strength = float(
        row["topic_strength"]
    )

    score = float(
        row["score"]
    )

    text_length = float(
        row["word_count"]
    )

    length_component = min(
        text_length,
        100,
    ) / 100

    strength_component = min(
        strength,
        0.35,
    ) / 0.35

    if score == 1:

        severity_component = 1.0

    elif score == 2:

        severity_component = 0.8

    else:

        severity_component = 0.5

    # Prefer substantive reviews without
    # making long reviews automatically win.
    if text_length >= 40:

        informativeness_component = 1.0

    elif text_length >= 20:

        informativeness_component = 0.8

    elif text_length >= 10:

        informativeness_component = 0.6

    else:

        informativeness_component = 0.4

    return (
        0.45 * strength_component
        + 0.20 * length_component
        + 0.15 * severity_component
        + 0.20 * informativeness_component
    )


# =========================================================
# SELECT EVIDENCE
# =========================================================

def select_evidence_reviews(
    category_df: pd.DataFrame,
    category: str,
) -> pd.DataFrame:

    category_df = category_df[
        category_df["is_mapped_pain"]
        == True
    ].copy()

    if category_df.empty:

        return category_df

    # -----------------------------------------------------
    # Prepare display text / word counts.
    # -----------------------------------------------------

    category_df["display_text"] = (
        category_df["normalized_text"]
        .apply(prepare_text)
    )

    category_df["word_count"] = (
        category_df["display_text"]
        .str.split()
        .str.len()
    )

    # -----------------------------------------------------
    # Apply evidence-specific quality filters.
    # -----------------------------------------------------

    candidate_mask = category_df.apply(
        lambda row: is_evidence_candidate(
            row,
            category,
        ),
        axis=1,
    )

    candidate_df = category_df[
        candidate_mask
    ].copy()

    # -----------------------------------------------------
    # Prefer strong topic assignments if
    # enough evidence remains.
    # -----------------------------------------------------

    strong = candidate_df[
        candidate_df["topic_strength"]
        >= MIN_TOPIC_STRENGTH
    ].copy()

    if len(strong) >= REVIEWS_PER_CATEGORY:

        candidate_df = strong

    # -----------------------------------------------------
    # Calculate evidence quality.
    # -----------------------------------------------------

    if candidate_df.empty:

        return candidate_df

    candidate_df["evidence_quality"] = (
        candidate_df.apply(
            evidence_quality,
            axis=1,
        )
    )

    # -----------------------------------------------------
    # Sort by evidence quality.
    # -----------------------------------------------------

    candidate_df = candidate_df.sort_values(
        [
            "evidence_quality",
            "topic_strength",
            "word_count",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    # -----------------------------------------------------
    # Avoid duplicate review text.
    # -----------------------------------------------------

    selected_rows = []

    seen_text = set()

    for _, row in candidate_df.iterrows():

        normalized_key = (
            re.sub(
                r"\s+",
                " ",
                row["display_text"]
                .strip()
                .lower(),
            )
        )

        if (
            normalized_key
            and normalized_key
            in seen_text
        ):

            continue

        selected_rows.append(
            row
        )

        seen_text.add(
            normalized_key
        )

        if len(selected_rows) >= (
            REVIEWS_PER_CATEGORY
        ):

            break

    return pd.DataFrame(
        selected_rows
    )


# =========================================================
# BUILD EVIDENCE TABLE
# =========================================================

def build_evidence_table(
    assignments: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    score_lookup = scores[
        [
            "app_key",
            "category",
            "product_pain_score",
            "prevalence_pct",
            "severity_1_2_within_category_pct",
            "strong_assignment_share_pct",
        ]
    ].copy()

    score_lookup = score_lookup.rename(
        columns={
            "prevalence_pct":
                "category_prevalence_pct",
            "severity_1_2_within_category_pct":
                "category_severity_1_2_pct",
            "strong_assignment_share_pct":
                "category_confidence_pct",
        }
    )

    # -----------------------------------------------------
    # Process every scored category.
    # -----------------------------------------------------

    for _, score_row in (
        score_lookup.iterrows()
    ):

        app_key = score_row[
            "app_key"
        ]

        category = score_row[
            "category"
        ]

        category_reviews = assignments[
            (
                assignments["app_key"]
                == app_key
            )
            &
            (
                assignments["final_category"]
                == category
            )
            &
            (
                assignments["is_mapped_pain"]
                == True
            )
        ].copy()

        if category_reviews.empty:

            continue

        evidence_reviews = (
            select_evidence_reviews(
                category_reviews,
                category,
            )
        )

        if evidence_reviews.empty:

            print(
                f"\nWARNING: No strong evidence "
                f"found for "
                f"{app_key} | {category}"
            )

            continue

        for evidence_rank, (
            _,
            review,
        ) in enumerate(
            evidence_reviews.iterrows(),
            start=1,
        ):

            rows.append(
                {
                    "app_key":
                        app_key,

                    "category":
                        category,

                    "product_pain_score": (
                        score_row[
                            "product_pain_score"
                        ]
                    ),

                    "category_prevalence_pct": (
                        score_row[
                            "category_prevalence_pct"
                        ]
                    ),

                    "category_severity_1_2_pct": (
                        score_row[
                            "category_severity_1_2_pct"
                        ]
                    ),

                    "category_confidence_pct": (
                        score_row[
                            "category_confidence_pct"
                        ]
                    ),

                    "evidence_rank":
                        evidence_rank,

                    "reviewId":
                        review["reviewId"],

                    "score":
                        int(review["score"]),

                    "topic_id":
                        int(review["topic_id"]),

                    "topic_strength":
                        float(
                            review[
                                "topic_strength"
                            ]
                        ),

                    "evidence_quality":
                        float(
                            review[
                                "evidence_quality"
                            ]
                        ),

                    "review_word_count":
                        int(
                            review["word_count"]
                        ),

                    "review_text":
                        clean_display_text(
                            review[
                                "display_text"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PRINT EVIDENCE
# =========================================================

def print_evidence(
    evidence: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL PAIN EVIDENCE"
    )

    print(
        "=" * 100
    )

    for app_key in APPS:

        app_df = evidence[
            evidence["app_key"]
            == app_key
        ].copy()

        if app_df.empty:

            continue

        print(
            f"\n{app_key.upper()}"
        )

        categories = (
            app_df["category"]
            .drop_duplicates()
            .tolist()
        )

        for category in categories:

            category_df = app_df[
                app_df["category"]
                == category
            ].sort_values(
                "evidence_rank"
            )

            first = (
                category_df.iloc[0]
            )

            print(
                "\n"
                + "-" * 95
            )

            print(
                f"{category}"
            )

            print(
                f"Product Pain Score: "
                f"{first['product_pain_score']:.2f}"
            )

            print(
                f"Prevalence: "
                f"{first['category_prevalence_pct']:.2f}%"
            )

            print(
                f"Severity: "
                f"{first['category_severity_1_2_pct']:.2f}%"
            )

            print(
                f"Model confidence: "
                f"{first['category_confidence_pct']:.2f}%"
            )

            print(
                "\nEvidence reviews:"
            )

            for _, row in (
                category_df.iterrows()
            ):

                print(
                    f"\n  "
                    f"{int(row['evidence_rank'])}. "
                    f"[{int(row['score'])}★] "
                    f"{row['review_text']}"
                )

                print(
                    f"     topic="
                    f"{int(row['topic_id'])}, "
                    f"strength="
                    f"{row['topic_strength']:.4f}, "
                    f"words="
                    f"{int(row['review_word_count'])}"
                )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BUILDING FINAL PAIN EVIDENCE"
    )

    print(
        "=" * 100
    )

    assignments = (
        load_assignments()
    )

    scores = (
        load_scores()
    )

    print(
        f"\nLoaded review assignments: "
        f"{len(assignments):,}"
    )

    print(
        f"Loaded scored categories: "
        f"{len(scores):,}"
    )

    evidence = build_evidence_table(
        assignments=assignments,
        scores=scores,
    )

    if evidence.empty:

        raise RuntimeError(
            "No evidence reviews were generated."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    evidence.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print_evidence(
        evidence
    )

    # -----------------------------------------------------
    # Coverage summary
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    coverage = (
        evidence.groupby(
            [
                "app_key",
                "category",
            ]
        )
        .size()
        .reset_index(
            name="evidence_count"
        )
    )

    print(
        "EVIDENCE COVERAGE"
    )

    print(
        "=" * 100
    )

    for _, row in coverage.iterrows():

        print(
            f"{row['app_key'].upper():12s} | "
            f"{row['category']}: "
            f"{int(row['evidence_count'])} reviews"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Saved evidence table:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "=" * 100
    )

    print(
        "FINAL PAIN EVIDENCE COMPLETE"
    )

    print(
        "=" * 100
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()