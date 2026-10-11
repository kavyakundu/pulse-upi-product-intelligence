import re
from pathlib import Path

import pandas as pd
import yaml
from lingua import Language, LanguageDetectorBuilder


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIGURATION
# =========================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    CONFIG = yaml.safe_load(file)


START_DATE = pd.Timestamp(
    CONFIG["analysis"]["start_date"]
)

# Make the configured end date fully inclusive.
END_DATE = (
    pd.Timestamp(
        CONFIG["analysis"]["end_date"]
    )
    + pd.Timedelta(days=1)
    - pd.Timedelta(microseconds=1)
)


# =========================================================
# DATASETS
# =========================================================

DATASETS = {
    "phonepe": {
        "name": "PhonePe",
        "package_id": "com.phonepe.app",
        "path": RAW_DATA_DIR
        / "phonepe_reviews.csv",
    },
    "paytm": {
        "name": "Paytm",
        "package_id": "net.one97.paytm",
        "path": RAW_DATA_DIR
        / "paytm_reviews.csv",
    },
    "googlepay": {
        "name": "Google Pay",
        "package_id": "com.google.android.apps.nbu.paisa.user",
        "path": RAW_DATA_DIR
        / "googlepay_reviews.csv",
    },
}


# =========================================================
# LANGUAGE DETECTION
# =========================================================

LANGUAGES = [
    Language.ENGLISH,
    Language.HINDI,
    Language.BENGALI,
    Language.GUJARATI,
    Language.MARATHI,
    Language.PUNJABI,
    Language.TAMIL,
    Language.TELUGU,
    Language.URDU,
]

LANGUAGE_DETECTOR = (
    LanguageDetectorBuilder
    .from_languages(*LANGUAGES)
    .build()
)


# =========================================================
# EMOJI PATTERN
# =========================================================

EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001FAFF"
    r"\U00002600-\U000027BF"
    r"\U0001F1E6-\U0001F1FF]"
)


# =========================================================
# SCRIPT RANGES
# =========================================================

SCRIPT_RANGES = {
    "Latin": (0x0041, 0x024F),
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}


# =========================================================
# CONSERVATIVE NORMALIZATION MAP
# =========================================================
#
# Only high-confidence spelling / abbreviation variants
# are normalized.
#
# IMPORTANT:
# The map is applied only to reviews that are likely to
# contain Romanized Indian-language or multilingual text.
# Pure English reviews are protected from these mappings.
# =========================================================

NORMALIZATION_MAP = {

    # -----------------------------------------------------
    # Common chat / abbreviation variants
    # -----------------------------------------------------

    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tq": "thanks",
    "bcoz": "because",
    "coz": "because",
    "bt": "but",
    "nd": "and",
    "abt": "about",
    "msg": "message",
    "msgs": "messages",

    # -----------------------------------------------------
    # Romanized Indian-language variants
    # -----------------------------------------------------

    "nhi": "nahi",
    "nahin": "nahi",
    "nai": "nahi",
    "ni": "nahi",

    "acha": "achha",
    "accha": "achha",
    "achhaa": "achha",

    "bhot": "bahut",
    "bhut": "bahut",
    "bht": "bahut",

    "gud": "good",

    "kr": "kar",
    "krna": "karna",
    "krne": "karne",
    "krta": "karta",
    "krti": "karti",

    "rha": "raha",
    "rhi": "rahi",

    "gya": "gaya",
    "gyi": "gayi",
    "gaye": "gaye",

    "muje": "mujhe",
    "mja": "maza",

    "bariya": "badiya",
    "bekaar": "bekar",
    "bakwaas": "bakwas",

    "prblm": "problem",
    "problm": "problem",

    # -----------------------------------------------------
    # Common review spelling errors
    # -----------------------------------------------------

    "tranfer": "transfer",
    "transection": "transaction",
    "transction": "transaction",

    "refferal": "referral",
    "refral": "referral",

    "cutmer": "customer",
    "costmer": "customer",
    "costumer": "customer",
    "cutomer": "customer",

    "supor": "support",
    "suppot": "support",

    "worest": "worst",
    "wrost": "worst",

    "samsya": "samasya",

    "usefull": "useful",

    "helo": "hello",
    "hlo": "hello",
}


# =========================================================
# HELPERS
# =========================================================

def count_script_characters(text: str) -> dict:
    """
    Count characters belonging to each supported script.
    """

    counts = {
        script: 0
        for script in SCRIPT_RANGES
    }

    for char in text:

        code_point = ord(char)

        for script, (start, end) in SCRIPT_RANGES.items():

            if start <= code_point <= end:

                counts[script] += 1
                break

    return counts


def get_script_label(text: str) -> str:
    """
    Return the dominant script label.

    Possible results include:
    Latin, Devanagari, Bengali, Tamil,
    Multiscript, Other/Unknown, etc.
    """

    counts = count_script_characters(text)

    present = [
        script
        for script, count in counts.items()
        if count > 0
    ]

    if not present:
        return "Other/Unknown"

    if len(present) == 1:
        return present[0]

    return "Multiscript"


def contains_code_mix(text: str) -> bool:
    """
    Detect Latin combined with any supported
    non-Latin Indian script.

    This is a script-level signal,
    not a full language classifier.
    """

    counts = count_script_characters(text)

    latin_present = (
        counts["Latin"] > 0
    )

    indian_script_present = any(
        counts[script] > 0
        for script in SCRIPT_RANGES
        if script != "Latin"
    )

    return (
        latin_present
        and indian_script_present
    )


def get_emoji_count(text: str) -> int:
    return len(
        EMOJI_PATTERN.findall(text)
    )


def remove_emojis(text: str) -> str:
    """
    Remove emoji characters and Unicode remnants
    commonly left behind by emoji sequences.
    """

    text = EMOJI_PATTERN.sub(
        " ",
        text,
    )

    # Remove emoji variation selectors and
    # zero-width joiners.
    text = re.sub(
        r"[\uFE0E\uFE0F\u200D]",
        "",
        text,
    )

    return text


def is_emoji_only(text: str) -> bool:
    return (
        remove_emojis(text).strip()
        == ""
    )


def normalize_repeated_characters(
    text: str,
) -> str:
    """
    Reduce exaggerated repeated LETTERS only.

    Examples:
        goooooood -> good
        bhtttttt -> bhtt
        tooooooo -> too

    Numbers are intentionally protected.

    Examples:
        6000 -> 6000
        999.65 -> 999.65
        20000 -> 20000
    """

    return re.sub(
        r"([a-zA-Z])\1{2,}",
        r"\1\1",
        text,
    )


def looks_romanized_indian(
    text: str,
) -> bool:
    """
    Heuristic flag for likely Romanized
    Indian-language text.

    This is NOT treated as language ground truth.
    """

    text = text.lower()

    romanized_terms = {
        "paisa",
        "paise",
        "nahi",
        "nahin",
        "nhi",
        "acha",
        "achha",
        "accha",
        "bahut",
        "bhot",
        "bhut",
        "bht",
        "kya",
        "kyu",
        "kyon",
        "hai",
        "hota",
        "raha",
        "rahi",
        "gaya",
        "gayi",
        "mujhe",
        "kaise",
        "karna",
        "krna",
        "chal",
        "chalta",
        "nahi mila",
        "kat gaya",
        "kat gya",
        "bheja",
        "bhej",
        "milta",
        "mil gaya",
        "chahiye",
        "bekar",
        "bakwas",
        "badhiya",
        "badiya",
        "sahi",
        "samasya",
    }

    words = text.split()

    if len(words) < 2:
        return False

    for term in romanized_terms:

        if " " in term:

            if term in text:
                return True

        else:

            if re.search(
                rf"\b{re.escape(term)}\b",
                text,
            ):
                return True

    return False


def should_apply_indian_normalization(
    script: str,
    code_mix: bool,
    romanized_indian: bool,
) -> bool:
    """
    Apply Indian/multilingual normalization if the review:
    - contains a non-Latin script, OR
    - is script-level code-mixed, OR
    - is likely Romanized Indian text.
    """

    return (
        script != "Latin"
        or code_mix
        or romanized_indian
    )


def normalize_word_variants(
    text: str,
    apply_indian_normalization: bool,
) -> str:
    """
    Apply high-confidence token-level normalization.

    Pure English reviews are protected from the
    Indian-language normalization map.
    """

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        match = re.match(
            r"^([^a-zA-Z0-9]*)"
            r"([a-zA-Z0-9]+)"
            r"([^a-zA-Z0-9]*)$",
            token,
        )

        if not match:

            normalized_tokens.append(
                token
            )

            continue

        prefix = match.group(1)

        word = (
            match.group(2)
            .lower()
        )

        suffix = match.group(3)

        replacement = word

        if (
            apply_indian_normalization
            and word in NORMALIZATION_MAP
        ):

            replacement = (
                NORMALIZATION_MAP[word]
            )

        normalized_tokens.append(
            prefix
            + replacement
            + suffix
        )

    return " ".join(
        normalized_tokens
    )


def normalize_text(
    raw_text: str,
    script: str,
    code_mix: bool,
    romanized_indian: bool,
) -> str:
    """
    Build normalized text while preserving
    the original raw review separately.
    """

    text = raw_text.lower()

    text = remove_emojis(text)

    text = normalize_repeated_characters(
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    apply_indian_normalization = (
        should_apply_indian_normalization(
            script=script,
            code_mix=code_mix,
            romanized_indian=romanized_indian,
        )
    )

    text = normalize_word_variants(
        text=text,
        apply_indian_normalization=(
            apply_indian_normalization
        ),
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def detect_language(
    text: str,
) -> str:

    stripped = text.strip()

    if not stripped:
        return "Unknown"

    language = (
        LANGUAGE_DETECTOR
        .detect_language_of(
            stripped
        )
    )

    if language is None:
        return "Unknown"

    return language.name


# =========================================================
# PROCESS ONE DATASET
# =========================================================

def process_dataset(
    app_key: str,
    app_info: dict,
) -> None:

    file_path = app_info["path"]

    app_name = app_info["name"]

    package_id = (
        app_info["package_id"]
    )

    print("\n" + "=" * 70)

    print(
        f"PROCESSING: "
        f"{app_name.upper()}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"Raw dataset not found: "
            f"{file_path}"
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = pd.read_csv(
        file_path
    )

    print(
        f"Loaded rows: "
        f"{len(df):,}"
    )

    # -----------------------------------------------------
    # Add stable app metadata
    # -----------------------------------------------------

    df["app_key"] = app_key

    df["app_name"] = app_name

    df["package_id"] = package_id

    # -----------------------------------------------------
    # Parse timestamps
    # -----------------------------------------------------

    df["at"] = pd.to_datetime(
        df["at"],
        errors="coerce",
    )

    invalid_dates = (
        df["at"].isna().sum()
    )

    if invalid_dates > 0:

        raise ValueError(
            f"{app_name}: "
            f"{invalid_dates:,} invalid "
            f"timestamps detected."
        )

    # -----------------------------------------------------
    # Enforce locked analysis window
    # -----------------------------------------------------

    before_date_filter = len(df)

    df = df[
        df["at"].between(
            START_DATE,
            END_DATE,
            inclusive="both",
        )
    ].copy()

    outside_window_removed = (
        before_date_filter
        - len(df)
    )

    print(
        f"Out-of-window rows removed: "
        f"{outside_window_removed:,}"
    )

    if df.empty:

        raise RuntimeError(
            f"{app_name}: no rows remain "
            f"after date filtering."
        )

    # -----------------------------------------------------
    # Defensive deduplication
    # -----------------------------------------------------

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["reviewId"],
        keep="first",
    ).copy()

    duplicates_removed = (
        before_dedup
        - len(df)
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed:,}"
    )

    # -----------------------------------------------------
    # Preserve original review text
    # -----------------------------------------------------

    df["raw_text"] = (
        df["content"]
        .fillna("")
        .astype(str)
    )

    # -----------------------------------------------------
    # Remove records with neither usable text nor rating
    # -----------------------------------------------------

    before_content_filter = len(df)

    df = df[
        (
            df["raw_text"]
            .str.strip()
            .ne("")
        )
        |
        (
            df["score"].notna()
        )
    ].copy()

    empty_records_removed = (
        before_content_filter
        - len(df)
    )

    print(
        f"Records without text/rating removed: "
        f"{empty_records_removed:,}"
    )

    # -----------------------------------------------------
    # Basic text metrics
    # -----------------------------------------------------

    df["char_count"] = (
        df["raw_text"].str.len()
    )

    df["word_count"] = (
        df["raw_text"]
        .str.split()
        .str.len()
    )

    df["emoji_count"] = (
        df["raw_text"]
        .apply(get_emoji_count)
    )

    df["emoji_only"] = (
        df["raw_text"]
        .apply(is_emoji_only)
    )

    # -----------------------------------------------------
    # Script detection
    # -----------------------------------------------------

    print(
        "Detecting scripts..."
    )

    df["script"] = (
        df["raw_text"]
        .apply(get_script_label)
    )

    # -----------------------------------------------------
    # Code-mix detection
    # -----------------------------------------------------

    df["contains_code_mix"] = (
        df["raw_text"]
        .apply(contains_code_mix)
    )

    # -----------------------------------------------------
    # Romanized Indian heuristic
    # -----------------------------------------------------

    print(
        "Detecting likely Romanized "
        "Indian-language reviews..."
    )

    df["likely_romanized_indian"] = (
        df["raw_text"]
        .apply(looks_romanized_indian)
    )

    # -----------------------------------------------------
    # Language detection
    # -----------------------------------------------------

    print(
        "Detecting languages..."
    )

    df["language_detected"] = (
        df["raw_text"]
        .apply(detect_language)
    )

    # -----------------------------------------------------
    # Basic clean text
    # -----------------------------------------------------

    df["clean_text"] = (
        df["raw_text"]
        .str.lower()
        .apply(remove_emojis)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    # -----------------------------------------------------
    # Multilingual normalized text
    # -----------------------------------------------------

    print(
        "Applying multilingual noise "
        "and abbreviation normalization..."
    )

    df["normalized_text"] = [
        normalize_text(
            raw_text=raw_text,
            script=script,
            code_mix=code_mix,
            romanized_indian=romanized,
        )
        for raw_text,
        script,
        code_mix,
        romanized
        in zip(
            df["raw_text"],
            df["script"],
            df["contains_code_mix"],
            df["likely_romanized_indian"],
        )
    ]

    # -----------------------------------------------------
    # Text eligibility
    # -----------------------------------------------------

    df["text_eligible"] = (
        df["word_count"] >= 2
    ) & (
        ~df["emoji_only"]
    ) & (
        df["normalized_text"]
        .str.len() >= 3
    )

    # -----------------------------------------------------
    # Weekly fields
    # -----------------------------------------------------

    iso_calendar = (
        df["at"].dt.isocalendar()
    )

    df["year_week"] = (
        iso_calendar["year"]
        .astype(str)
        + "-W"
        + iso_calendar["week"]
        .astype(str)
        .str.zfill(2)
    )

    df["week_start_date"] = (
        df["at"]
        - pd.to_timedelta(
            df["at"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    # -----------------------------------------------------
    # Rating validation
    # -----------------------------------------------------

    valid_ratings = df["score"].isin(
        [1, 2, 3, 4, 5]
    )

    if not valid_ratings.all():

        invalid_count = (
            (~valid_ratings).sum()
        )

        raise ValueError(
            f"{app_name}: "
            f"{invalid_count:,} invalid "
            f"ratings detected."
        )

    # -----------------------------------------------------
    # Rating flags
    # -----------------------------------------------------

    df["is_1_star"] = (
        df["score"] == 1
    )

    df["is_2_star"] = (
        df["score"] == 2
    )

    df["is_negative_rating"] = (
        df["score"].isin([1, 2])
    )

    # -----------------------------------------------------
    # Final columns
    # -----------------------------------------------------

    columns = [
        "reviewId",
        "app_key",
        "app_name",
        "package_id",
        "raw_text",
        "clean_text",
        "normalized_text",
        "score",
        "is_1_star",
        "is_2_star",
        "is_negative_rating",
        "thumbsUpCount",
        "at",
        "year_week",
        "week_start_date",
        "reviewCreatedVersion",
        "appVersion",
        "char_count",
        "word_count",
        "emoji_count",
        "emoji_only",
        "script",
        "language_detected",
        "contains_code_mix",
        "likely_romanized_indian",
        "text_eligible",
    ]

    df = df[columns].copy()

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "at",
            "reviewId",
        ]
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        PROCESSED_DATA_DIR
        / f"{app_key}_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        f"\nRows after processing: "
        f"{len(df):,}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Text-eligible reviews: "
        f"{df['text_eligible'].sum():,} "
        f"("
        f"{df['text_eligible'].mean() * 100:.2f}%"
        f")"
    )

    print(
        f"Likely Romanized Indian reviews: "
        f"{df['likely_romanized_indian'].sum():,} "
        f"("
        f"{df['likely_romanized_indian'].mean() * 100:.2f}%"
        f")"
    )

    print(
        f"Code-mixed script flag: "
        f"{df['contains_code_mix'].sum():,}"
    )

    print(
        f"Emoji-only reviews: "
        f"{df['emoji_only'].sum():,}"
    )

    # -----------------------------------------------------
    # Show meaningful normalization examples
    # -----------------------------------------------------

    print(
        "\nNormalization examples:"
    )

    comparison = df[
        df["normalized_text"]
        != df["raw_text"].str.lower()
    ][
        [
            "raw_text",
            "normalized_text",
        ]
    ].drop_duplicates()

    if comparison.empty:

        print(
            "  No normalization changes found."
        )

    else:

        examples = []

        for _, row in comparison.iterrows():

            raw = row["raw_text"]

            normalized = (
                row["normalized_text"]
            )

            # Ignore examples where the only change
            # is emoji removal, whitespace, or case.
            raw_without_emojis = (
                remove_emojis(
                    raw.lower()
                )
            )

            raw_without_emojis = re.sub(
                r"\s+",
                " ",
                raw_without_emojis,
            ).strip()

            if normalized != (
                raw_without_emojis
            ):

                examples.append(
                    row
                )

            if len(examples) >= 15:
                break

        if examples:

            for row in examples:

                print(
                    f"  RAW:        "
                    f"{row['raw_text']}"
                )

                print(
                    f"  NORMALIZED: "
                    f"{row['normalized_text']}"
                )

                print()

        else:

            print(
                "  Only case/emoji/whitespace "
                "changes found in sample."
            )

    print(
        f"Saved to: "
        f"{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    for app_key, app_info in DATASETS.items():

        process_dataset(
            app_key=app_key,
            app_info=app_info,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()