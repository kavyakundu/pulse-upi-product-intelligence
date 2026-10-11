import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from google_play_scraper import Sort, reviews


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


START_DATE = datetime.fromisoformat(
    config["analysis"]["start_date"]
).replace(tzinfo=None)

END_DATE = datetime.fromisoformat(
    config["analysis"]["end_date"]
).replace(tzinfo=None)

LANGUAGE = config["scraping"]["language"]
COUNTRY = config["scraping"]["country"]

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Scraper settings
# ---------------------------------------------------------

PAGE_SIZE = 500

# Conservative delay between requests.
REQUEST_DELAY_SECONDS = 1.5

# Retry configuration for temporary empty responses.
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 5


# ---------------------------------------------------------
# Scrape one app
# ---------------------------------------------------------

def scrape_app(app_key: str, app_info: dict) -> None:
    app_name = app_info["name"]
    package_id = app_info["package_id"]

    print(f"\n{'=' * 60}")
    print(f"Starting: {app_name}")
    print(f"Package: {package_id}")
    print(f"Window: {START_DATE.date()} → {END_DATE.date()}")
    print(f"{'=' * 60}")

    all_reviews = []
    continuation_token = None
    page_number = 0

    while True:

        page_number += 1

        print(f"Fetching page {page_number}...", end=" ")

        kwargs = {
            "lang": LANGUAGE,
            "country": COUNTRY,
            "sort": Sort.NEWEST,
            "count": PAGE_SIZE,
        }

        if continuation_token is not None:
            kwargs["continuation_token"] = continuation_token

        result, next_token = reviews(
            package_id,
            **kwargs,
        )

        print(f"{len(result)} reviews")

        # -------------------------------------------------
        # Handle temporary empty responses
        # -------------------------------------------------

        if not result:

            retry_successful = False

            print("Empty response received. Retrying...")

            for retry_number in range(1, MAX_RETRIES + 1):

                wait_time = RETRY_BASE_SECONDS * (
                    2 ** (retry_number - 1)
                )

                print(
                    f"Retry {retry_number}/{MAX_RETRIES} "
                    f"after {wait_time} seconds..."
                )

                time.sleep(wait_time)

                result, next_token = reviews(
                    package_id,
                    lang=LANGUAGE,
                    country=COUNTRY,
                    sort=Sort.NEWEST,
                    count=PAGE_SIZE,
                    continuation_token=continuation_token,
                )

                if result:
                    print(
                        f"Retry successful: "
                        f"{len(result)} reviews returned."
                    )
                    retry_successful = True
                    break

            if not retry_successful:
                raise RuntimeError(
                    f"Three consecutive retries returned no reviews "
                    f"for {app_name}. Stopping rather than assuming "
                    f"historical coverage has ended."
                )

        # -------------------------------------------------
        # Process page
        # -------------------------------------------------

        page_dates = [review["at"] for review in result]

        oldest_date = min(page_dates)

        for review in result:
            review_date = review["at"]

            if START_DATE <= review_date <= END_DATE:
                review["app_key"] = app_key
                review["app_name"] = app_name
                review["package_id"] = package_id
                all_reviews.append(review)

        # Update continuation token after successful retrieval.
        continuation_token = next_token

        # -------------------------------------------------
        # Stop after crossing the start-date boundary
        # -------------------------------------------------

        if oldest_date < START_DATE:
            print(
                "Reached the beginning of the analysis window."
            )
            break

        if continuation_token is None:
            raise RuntimeError(
                f"Pagination ended before crossing the configured "
                f"start date for {app_name}. "
                f"Oldest review reached: {oldest_date}. "
                f"Expected to reach before: {START_DATE}."
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    # -----------------------------------------------------
    # Validate that we actually captured data
    # -----------------------------------------------------

    if not all_reviews:
        raise RuntimeError(
            f"No reviews found in the analysis window for {app_name}."
        )

    # -----------------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(all_reviews)

    # -----------------------------------------------------
    # Remove duplicate review IDs defensively
    # -----------------------------------------------------

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["reviewId"],
        keep="first",
    ).copy()

    duplicates_removed = before_dedup - len(df)

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df = df.sort_values("at").reset_index(drop=True)

    # -----------------------------------------------------
    # Select stable/raw fields
    # -----------------------------------------------------

    columns = [
        "reviewId",
        "content",
        "score",
        "thumbsUpCount",
        "at",
        "reviewCreatedVersion",
        "appVersion",
        "app_key",
        "app_name",
        "package_id",
    ]

    df = df[columns]

    # -----------------------------------------------------
    # Save raw review data
    # -----------------------------------------------------

    output_path = RAW_DATA_DIR / f"{app_key}_reviews.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Collection metadata
    # -----------------------------------------------------

    scrape_timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "app_key": app_key,
        "app_name": app_name,
        "package_id": package_id,
        "analysis_start_date": config["analysis"]["start_date"],
        "analysis_end_date": config["analysis"]["end_date"],
        "language": LANGUAGE,
        "country": COUNTRY,
        "sort": "newest",
        "page_size": PAGE_SIZE,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "max_retries": MAX_RETRIES,
        "retry_base_seconds": RETRY_BASE_SECONDS,
        "scrape_timestamp_utc": scrape_timestamp,
        "pages_fetched": page_number,
        "reviews_before_deduplication": before_dedup,
        "duplicates_removed": duplicates_removed,
        "reviews_saved": len(df),
        "earliest_saved_review": df["at"].min().isoformat(),
        "latest_saved_review": df["at"].max().isoformat(),
    }

    metadata_path = (
        RAW_DATA_DIR
        / f"{app_key}_collection_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"\nCompleted: {app_name}")
    print(f"Reviews saved: {len(df):,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Earliest review: {df['at'].min()}")
    print(f"Latest review: {df['at'].max()}")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    # Test PhonePe first.
    # Once PhonePe completes successfully, switch this
    # to all three apps.
    app_key = "googlepay"
    app_info = config["apps"][app_key]

    scrape_app(app_key, app_info)


if __name__ == "__main__":
    main()