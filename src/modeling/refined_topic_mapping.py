from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

REFINED_DIR = (
    MODEL_DIR
    / "refined"
)

OUTPUT_PATH = (
    REFINED_DIR
    / "refined_topic_mapping.csv"
)


# =========================================================
# SETTINGS
# =========================================================

APPS = [
    "paytm",
    "googlepay",
]

EXPECTED_TOPICS = set(
    range(1, 26)
)

VALID_DECISIONS = {
    "KEEP",
    "MERGE",
    "NOISE",
}


# =========================================================
# PAYTM REFINED TOPIC MAP
# =========================================================

PAYTM_TOPIC_MAP = {

    # -----------------------------------------------------
    # App reliability & access
    # -----------------------------------------------------

    1: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Broad working/not-working language with mixed "
            "update and error complaints."
        ),
    },

    8: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Broad not/opening language with insufficient "
            "topic specificity."
        ),
    },

    9: {
        "decision": "KEEP",
        "final_category": "App reliability & access",
        "reason": (
            "Clear Romanized Indian-language complaints about "
            "login, download, installation, and opening failures."
        ),
    },

    24: {
        "decision": "KEEP",
        "final_category": "App reliability & access",
        "reason": (
            "Clear repeated complaints about being unable to "
            "open or access the Paytm app."
        ),
    },

    # -----------------------------------------------------
    # Device & security compatibility
    # -----------------------------------------------------

    5: {
        "decision": "KEEP",
        "final_category": "Device & security compatibility",
        "reason": (
            "Highly coherent device-environment compatibility "
            "and false integrity-warning complaints."
        ),
    },

    16: {
        "decision": "MERGE",
        "final_category": "Device & security compatibility",
        "reason": (
            "Device-integrity and root-detection failures are "
            "the same underlying compatibility/security problem."
        ),
    },

    # -----------------------------------------------------
    # Customer support
    # -----------------------------------------------------

    6: {
        "decision": "KEEP",
        "final_category": "Customer support",
        "reason": (
            "Direct customer-service and service-quality complaints."
        ),
    },

    14: {
        "decision": "KEEP",
        "final_category": "Customer support",
        "reason": (
            "Clear complaints about unavailable customer support, "
            "complaint channels, and lack of response."
        ),
    },

    18: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Too generic; many reviews only say 'poor application' "
            "without establishing a specific support problem."
        ),
    },

    # -----------------------------------------------------
    # Transaction & payment issues
    # -----------------------------------------------------

    10: {
        "decision": "KEEP",
        "final_category": "Transaction & payment issues",
        "reason": (
            "Topic contains payment pending, failed, and payment "
            "processing complaints, despite some generic positives."
        ),
    },

    # -----------------------------------------------------
    # Cashback & rewards
    # -----------------------------------------------------

    13: {
        "decision": "KEEP",
        "final_category": "Cashback & rewards",
        "reason": (
            "Clear cashback, offers, and missing-reward complaints."
        ),
    },

    15: {
        "decision": "MERGE",
        "final_category": "Cashback & rewards",
        "reason": (
            "Direct overlap with cashback and reward complaints."
        ),
    },

    # -----------------------------------------------------
    # Performance & speed
    # -----------------------------------------------------

    20: {
        "decision": "KEEP",
        "final_category": "Performance & speed",
        "reason": (
            "Specific slow-app and slow-processing complaints."
        ),
    },

    # -----------------------------------------------------
    # Fraud, trust & security
    # -----------------------------------------------------

    22: {
        "decision": "KEEP",
        "final_category": "Fraud, trust & security",
        "reason": (
            "Repeated fraud, scam, and trust-related complaints."
        ),
    },

    # -----------------------------------------------------
    # Account & login issues
    # -----------------------------------------------------

    25: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Topic is too broad and mixes bank linking, payment "
            "pending, device warnings, and generic problem language."
        ),
    },

    # -----------------------------------------------------
    # Remaining topics
    # -----------------------------------------------------

    2: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic bad-app and bad-service sentiment."
        ),
    },

    3: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Broad generic language with weak topical coherence."
        ),
    },

    4: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic worst-app sentiment."
        ),
    },

    7: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mixed Paytm references, Gold, fraud, and unrelated content."
        ),
    },

    11: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic experience language without a specific product problem."
        ),
    },

    12: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mixed generic app sentiment and unrelated complaints."
        ),
    },

    17: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic work/not-work vocabulary with insufficient specificity."
        ),
    },

    19: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic dislike/download language with mixed issues."
        ),
    },

    21: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic okay/thanks/positive language."
        ),
    },

    23: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic positive payment language."
        ),
    },
}


# =========================================================
# GOOGLE PAY REFINED TOPIC MAP
# =========================================================

GOOGLEPAY_TOPIC_MAP = {

    # -----------------------------------------------------
    # App reliability & access
    # -----------------------------------------------------

    2: {
        "decision": "KEEP",
        "final_category": "App reliability & access",
        "reason": (
            "Clear app-not-working complaints and functional failures."
        ),
    },

    9: {
        "decision": "MERGE",
        "final_category": "App reliability & access",
        "reason": (
            "Romanized Indian-language complaints about download, "
            "installation, and app-opening failures."
        ),
    },

    19: {
        "decision": "KEEP",
        "final_category": "App reliability & access",
        "reason": (
            "Clear complaints about GPay opening, accessing, or "
            "launching the app."
        ),
    },

    22: {
        "decision": "MERGE",
        "final_category": "App reliability & access",
        "reason": (
            "Functional not-working/opening complaints that overlap "
            "with the broader access problem."
        ),
    },

    # -----------------------------------------------------
    # Cashback & rewards
    # -----------------------------------------------------

    7: {
        "decision": "KEEP",
        "final_category": "Cashback & rewards",
        "reason": (
            "Clear missing cashback and reward complaints."
        ),
    },

    15: {
        "decision": "MERGE",
        "final_category": "Cashback & rewards",
        "reason": (
            "Direct overlap with cashback and reward complaints."
        ),
    },

    21: {
        "decision": "MERGE",
        "final_category": "Cashback & rewards",
        "reason": (
            "Direct overlap with reward and cashback complaints."
        ),
    },

    # -----------------------------------------------------
    # Customer support
    # -----------------------------------------------------

    8: {
        "decision": "KEEP",
        "final_category": "Customer support",
        "reason": (
            "Service and customer-support complaints form a "
            "recognizable support-related topic."
        ),
    },

    # -----------------------------------------------------
    # Bank account & linking
    # -----------------------------------------------------

    12: {
        "decision": "KEEP",
        "final_category": "Bank account & linking",
        "reason": (
            "Clear bank-account linking, activation, and balance issues."
        ),
    },

    # -----------------------------------------------------
    # Transaction & payment issues
    # -----------------------------------------------------

    13: {
        "decision": "KEEP",
        "final_category": "Transaction & payment issues",
        "reason": (
            "Contains payment processing, stuck, failed, and "
            "payment-status complaints."
        ),
    },

    # -----------------------------------------------------
    # Performance & speed
    # -----------------------------------------------------

    16: {
        "decision": "KEEP",
        "final_category": "Performance & speed",
        "reason": (
            "Specific slow-app and slow-processing complaints."
        ),
    },

    # -----------------------------------------------------
    # UPI & UPI Lite
    # -----------------------------------------------------

    20: {
        "decision": "KEEP",
        "final_category": "UPI & UPI Lite",
        "reason": (
            "Distinct UPI, UPI Lite, and UPI-control language."
        ),
    },

    # -----------------------------------------------------
    # Network, server & OTP issues
    # -----------------------------------------------------

    23: {
        "decision": "KEEP",
        "final_category": "Network, server & OTP issues",
        "reason": (
            "Contains server, network, OTP, and connection-related "
            "failure language."
        ),
    },

    # -----------------------------------------------------
    # AutoPay
    # -----------------------------------------------------
    #
    # Topic 6 was audited and found to be heavily contaminated
    # with cashback, advertising, trust, and unrelated content.
    # We therefore remove it instead of forcing an AutoPay category.
    # -----------------------------------------------------

    6: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Topic is contaminated with cashback, advertisements, "
            "trust, and unrelated complaints; insufficient evidence "
            "for a reliable AutoPay category."
        ),
    },

    # -----------------------------------------------------
    # Generic / noise topics
    # -----------------------------------------------------

    1: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic bad-app sentiment."
        ),
    },

    3: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Very broad topic mixing money, transactions, and unrelated text."
        ),
    },

    4: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic worst-app sentiment."
        ),
    },

    5: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic not-good language with mixed complaints."
        ),
    },

    10: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic poor-service language without a clear issue."
        ),
    },

    11: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic experience wording."
        ),
    },

    14: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic app-name and sentiment language."
        ),
    },

    17: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic time/reward language."
        ),
    },

    18: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Mostly generic okay/thanks language."
        ),
    },

    24: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic work/not-work language without sufficient specificity."
        ),
    },

    25: {
        "decision": "NOISE",
        "final_category": "",
        "reason": (
            "Generic don't-like/download language with mixed complaints."
        ),
    },
}


# =========================================================
# GET MAP
# =========================================================

TOPIC_MAP = {
    "paytm": PAYTM_TOPIC_MAP,
    "googlepay": GOOGLEPAY_TOPIC_MAP,
}


# =========================================================
# VALIDATION
# =========================================================

def validate_mapping() -> None:

    for app_key in APPS:

        if app_key not in TOPIC_MAP:

            raise ValueError(
                f"Missing topic map for {app_key}."
            )

        app_map = TOPIC_MAP[
            app_key
        ]

        actual_topics = set(
            app_map.keys()
        )

        missing = (
            EXPECTED_TOPICS
            - actual_topics
        )

        extra = (
            actual_topics
            - EXPECTED_TOPICS
        )

        if missing:

            raise ValueError(
                f"{app_key}: missing topics "
                f"{sorted(missing)}"
            )

        if extra:

            raise ValueError(
                f"{app_key}: unexpected topics "
                f"{sorted(extra)}"
            )

        for topic_id, mapping in (
            app_map.items()
        ):

            decision = mapping[
                "decision"
            ]

            if decision not in VALID_DECISIONS:

                raise ValueError(
                    f"{app_key} Topic "
                    f"{topic_id}: invalid decision "
                    f"{decision}"
                )

            if decision in {
                "KEEP",
                "MERGE",
            }:

                if not mapping[
                    "final_category"
                ]:

                    raise ValueError(
                        f"{app_key} Topic "
                        f"{topic_id}: final_category "
                        f"is required."
                    )


# =========================================================
# BUILD OUTPUT
# =========================================================

def build_mapping() -> pd.DataFrame:

    rows = []

    for app_key in APPS:

        topics_path = (
            REFINED_DIR
            / f"{app_key}_refined_topics.csv"
        )

        if not topics_path.exists():

            raise FileNotFoundError(
                f"Refined topics file not found:\n"
                f"{topics_path}"
            )

        topics = pd.read_csv(
            topics_path
        )

        required_columns = {
            "topic_id",
            "top_words",
        }

        missing = (
            required_columns
            - set(topics.columns)
        )

        if missing:

            raise ValueError(
                f"{app_key}: refined topics file "
                f"is missing columns: "
                f"{sorted(missing)}"
            )

        for _, topic in topics.iterrows():

            topic_id = int(
                topic["topic_id"]
            )

            mapping = TOPIC_MAP[
                app_key
            ][topic_id]

            rows.append(
                {
                    "app_key": app_key,
                    "topic_id": topic_id,
                    "top_words": topic[
                        "top_words"
                    ],
                    "decision": mapping[
                        "decision"
                    ],
                    "final_category": mapping[
                        "final_category"
                    ],
                    "reason": mapping[
                        "reason"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    mapping: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 90
    )

    print(
        "REFINED TOPIC MAPPING SUMMARY"
    )

    print(
        "=" * 90
    )

    for app_key in APPS:

        print(
            f"\n{app_key.upper()}"
        )

        app_mapping = mapping[
            mapping["app_key"]
            == app_key
        ]

        for decision in [
            "KEEP",
            "MERGE",
            "NOISE",
        ]:

            print(
                f"\n{decision}:"
            )

            subset = app_mapping[
                app_mapping["decision"]
                == decision
            ].sort_values(
                "topic_id"
            )

            for _, row in subset.iterrows():

                category = (
                    row["final_category"]
                    if row["final_category"]
                    else "-"
                )

                print(
                    f"  Topic "
                    f"{int(row['topic_id']):2d}: "
                    f"{category}"
                )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    validate_mapping()

    mapping = build_mapping()

    mapping.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(
        mapping
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"Saved refined mapping to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()