from pathlib import Path
import pandas as pd
import os
import tempfile


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

FILES = [
    VALIDATION_DIR / "complaint_filter_validation.csv",
    VALIDATION_DIR / "complaint_gold_set_batch2.csv",
]

AUDIT_FILE = (
    VALIDATION_DIR
    / "complaint_gold_label_corrections.csv"
)


# ============================================================
# FINAL GOLD-LABEL CORRECTION
# ============================================================
#
# This review is based ONLY on the agreed labeling rule:
#
# complaint:
# identifiable product/service problem, failure, friction,
# regression, missing functionality, etc.
#
# non_complaint:
# generic negativity, praise, preference, or opinion without
# an identifiable product problem.
#
# The remaining clearly inconsistent label is:
#
# "old phone pay is better than new version"
#
# This explicitly identifies a product/version regression.
# ============================================================

CORRECTIONS = {

    "d0909540-2521-4a31-8d0c-2edf5f31e80b": {
        "new_label": "complaint",
        "reason": (
            "Explicitly identifies dissatisfaction with the new "
            "version compared with the old version, indicating "
            "a product/version regression."
        ),
    },
}


# ============================================================
# LOAD VALIDATION FILES
# ============================================================

frames = []

for file_path in FILES:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required = {
        "reviewId",
        "review_text",
        "human_label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {sorted(missing)}"
        )

    df["source_file"] = file_path.name

    frames.append(df)


all_reviews = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# FIND ACTUAL CHANGES
# ============================================================

audit_rows = []

for review_id, correction in CORRECTIONS.items():

    matches = all_reviews[
        all_reviews["reviewId"].astype(str) == review_id
    ]

    if matches.empty:
        raise ValueError(
            f"Review ID not found: {review_id}"
        )

    for _, row in matches.iterrows():

        old_label = (
            str(row["human_label"])
            .strip()
            .lower()
        )

        new_label = correction["new_label"]

        if old_label == new_label:
            continue

        audit_rows.append(
            {
                "reviewId": review_id,
                "source_file": row["source_file"],
                "app_key": row.get("app_key", ""),
                "score": row.get("score", ""),
                "review_text": row["review_text"],
                "old_label": old_label,
                "new_label": new_label,
                "reason": correction["reason"],
            }
        )


audit_df = pd.DataFrame(audit_rows)

if audit_df.empty:
    print("No new gold-label corrections are required.")
    raise SystemExit(0)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("FINAL GOLD LABEL AUDIT")
print("=" * 70)

print(f"\nActual label changes: {len(audit_df)}")

for _, row in audit_df.iterrows():

    print("\n" + "-" * 70)

    print(f"ID:       {row['reviewId']}")
    print(f"App:      {row['app_key']}")
    print(f"Rating:   {row['score']}")
    print(f"Old:      {row['old_label']}")
    print(f"New:      {row['new_label']}")
    print(f"Text:     {row['review_text']}")
    print(f"Reason:   {row['reason']}")


# ============================================================
# SAVE AUDIT
# ============================================================

AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nAudit saved:")
print(AUDIT_FILE)


# ============================================================
# APPLY SAFELY
# ============================================================

correction_map = dict(
    zip(
        audit_df["reviewId"].astype(str),
        audit_df["new_label"],
    )
)


for file_path in FILES:

    df = pd.read_csv(file_path)

    review_ids = df["reviewId"].astype(str)

    mask = review_ids.isin(
        correction_map.keys()
    )

    if not mask.any():
        continue

    changed = int(mask.sum())

    df.loc[mask, "human_label"] = (
        review_ids[mask]
        .map(correction_map)
        .values
    )

    fd, temp_name = tempfile.mkstemp(
        suffix=".csv",
        dir=file_path.parent,
    )

    os.close(fd)

    temp_path = Path(temp_name)

    try:

        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temp_path,
            file_path,
        )

        print(
            f"Updated {changed} label(s): "
            f"{file_path.name}"
        )

    except PermissionError:

        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise PermissionError(
            f"\nWindows is locking this file:\n{file_path}\n\n"
            "Close the CSV/Excel file and run the script again."
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL GOLD LABEL CLEANUP COMPLETE")
print("=" * 70)

print(
    audit_df
    .groupby(["old_label", "new_label"])
    .size()
    .reset_index(name="count")
    .to_string(index=False)
)

print(
    "\nCorrection was made from the labeling rule, "
    "not from the classifier prediction."
)