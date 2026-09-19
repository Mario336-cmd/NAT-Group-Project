from pathlib import Path
from itertools import combinations
import json
import math

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "Dataset" / "Original" / "train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "Processed"

EXPECTED_LABELS = {
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO",
}

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

SAMPLES_PER_CLASS = 3


def load_dataset(file_path: Path) -> tuple[pd.DataFrame, list[str]]:
    records = []
    available_fields = set()

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}."
                )

            available_fields.update(record.keys())

            records.append(
                {
                    "id": record.get("id"),
                    "claim": record.get("claim"),
                    "label": record.get("label"),
                }
            )

    if not records:
        raise ValueError("The input dataset is empty.")

    return pd.DataFrame(records), sorted(available_fields)


def validate_ids(data: pd.DataFrame) -> None:
    if data["id"].isna().any():
        raise ValueError(
            "Missing record IDs: review the input before proceeding."
        )

    if data["id"].duplicated().any():
        raise ValueError(
            "Duplicate record IDs: review the input before proceeding."
        )


def inspect_dataset(
    data: pd.DataFrame,
    available_fields: list[str],
) -> None:
    print("\nFEVER DATASET INSPECTION")
    print("=" * 50)

    print(f"\nTotal records: {len(data):,}")

    print("\nAvailable fields:")
    for field in available_fields:
        print(f"- {field}")

    print("\nClass distribution:")
    class_counts = data["label"].value_counts(dropna=False)

    for label, count in class_counts.items():
        print(f"{label}: {count:,}")

    print("\nClass percentages:")
    class_percentages = (
        data["label"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    for label, percentage in class_percentages.items():
        print(f"{label}: {percentage:.2f}%")

    missing_claims = data["claim"].isna().sum()
    missing_labels = data["label"].isna().sum()

    empty_claims = (
        data["claim"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    string_claims = data.loc[
        data["claim"].map(
            lambda value: isinstance(value, str)
        ),
        "claim",
    ].str.strip()

    duplicate_claims = string_claims.duplicated().sum()

    labels_found = set(
        data["label"].dropna().map(str).unique()
    )

    unexpected_labels = labels_found - EXPECTED_LABELS

    print("\nData quality checks:")
    print(f"Missing claims: {missing_claims:,}")
    print(f"Missing labels: {missing_labels:,}")
    print(f"Empty claims: {empty_claims:,}")
    print(f"Duplicate claims: {duplicate_claims:,}")

    if unexpected_labels:
        print("\nUnexpected labels:")

        for label in sorted(unexpected_labels):
            print(f"- {label}")
    else:
        print("Unexpected labels: None")


def show_samples(data: pd.DataFrame) -> None:
    print("\nSAMPLE CLAIMS")
    print("=" * 50)

    for label in sorted(EXPECTED_LABELS):
        print(f"\n{label}:")

        samples = data[
            data["label"] == label
        ]["claim"].head(SAMPLES_PER_CLASS)

        for number, claim in enumerate(samples, start=1):
            print(f"{number}. {claim}")


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()

    cleaned_data["claim"] = cleaned_data["claim"].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

    cleaned_data["label"] = cleaned_data["label"].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

    valid_claims = cleaned_data["claim"].map(
        lambda value: isinstance(value, str) and bool(value)
    )

    valid_labels = cleaned_data["label"].map(
        lambda value: isinstance(value, str) and bool(value)
    )

    invalid_rows = ~valid_claims | ~valid_labels
    removed_rows = invalid_rows.sum()

    cleaned_data = cleaned_data[~invalid_rows].copy()

    if cleaned_data.empty:
        raise ValueError(
            "No usable records remain after cleaning."
        )

    print("\nCLEANING")
    print("=" * 50)
    print(f"Records with missing or invalid claims/labels removed: {removed_rows:,}")

    return cleaned_data.reset_index(drop=True)


def handle_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    label_counts = (
        data.groupby("claim")["label"]
        .transform("nunique")
    )

    conflicts = data[label_counts > 1].copy()
    consistent = data[label_counts == 1].copy()

    unique_data = consistent.drop_duplicates(
        subset="claim",
        keep="first",
    ).copy()

    print(
        f"Records with conflicting labels removed: "
        f"{len(conflicts):,}"
    )

    print(
        f"Duplicate records removed: "
        f"{len(consistent) - len(unique_data):,}"
    )

    print(
        f"Claims remaining: "
        f"{len(unique_data):,}"
    )

    if unique_data.empty:
        raise ValueError(
            "No consistently labelled claims remain."
        )

    return unique_data.reset_index(drop=True)


def validate_labels(data: pd.DataFrame) -> None:
    labels_found = set(data["label"].unique())
    unexpected_labels = labels_found - EXPECTED_LABELS

    if unexpected_labels:
        raise ValueError(
            "Unexpected labels found. "
            "Dataset splitting stopped so they can be reviewed first. "
            f"Unexpected labels: {sorted(unexpected_labels)}"
        )


def split_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sizes = (
        TRAIN_SIZE,
        VALIDATION_SIZE,
        TEST_SIZE,
    )

    if not all(
        0 < size < 1 for size in sizes
    ) or not math.isclose(
        sum(sizes),
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Split proportions must be between 0 and 1 and sum to 1."
        )

    remaining_size = VALIDATION_SIZE + TEST_SIZE

    train_data, remaining_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    test_ratio = TEST_SIZE / remaining_size

    validation_data, test_data = train_test_split(
        remaining_data,
        test_size=test_ratio,
        random_state=RANDOM_STATE,
        stratify=remaining_data["label"],
    )

    return train_data, validation_data, test_data


def validate_splits(
    data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    for name, split in splits.items():
        if set(split["label"]) != EXPECTED_LABELS:
            raise ValueError(
                f"The {name} split must contain all three classes."
            )

        if (
            split["id"].duplicated().any()
            or split["claim"].duplicated().any()
        ):
            raise ValueError(
                f"Duplicate IDs or claims within the {name} split."
            )

    for (
        (name_a, split_a),
        (name_b, split_b),
    ) in combinations(splits.items(), 2):

        for column in ("id", "claim"):
            if set(split_a[column]) & set(split_b[column]):
                raise ValueError(
                    f"Overlapping {column} between "
                    f"{name_a} and {name_b}."
                )

    combined = pd.concat(
        splits.values(),
        ignore_index=True,
    )

    if (
        len(combined) != len(data)
        or set(combined["id"]) != set(data["id"])
    ):
        raise ValueError(
            "The splits do not cover the prepared dataset exactly once."
        )


def print_split_summary(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    total = (
        len(train_data)
        + len(validation_data)
        + len(test_data)
    )

    print("\nDATASET SPLIT")
    print("=" * 50)

    print(
        f"Training: {len(train_data):,} "
        f"({len(train_data) / total * 100:.2f}%)"
    )

    print(
        f"Validation: {len(validation_data):,} "
        f"({len(validation_data) / total * 100:.2f}%)"
    )

    print(
        f"Test: {len(test_data):,} "
        f"({len(test_data) / total * 100:.2f}%)"
    )


def save_datasets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_data.to_csv(
        train_path,
        index=False,
    )

    validation_data.to_csv(
        validation_path,
        index=False,
    )

    test_data.to_csv(
        test_path,
        index=False,
    )

    print("\nFILES CREATED")
    print("=" * 50)

    print(
        train_path.relative_to(PROJECT_ROOT)
    )

    print(
        validation_path.relative_to(PROJECT_ROOT)
    )

    print(
        test_path.relative_to(PROJECT_ROOT)
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find: {INPUT_FILE}"
        )

    data, available_fields = load_dataset(
        INPUT_FILE
    )

    validate_ids(data)
    inspect_dataset(
        data,
        available_fields,
    )

    show_samples(data)

    cleaned_data = clean_dataset(data)

    validate_labels(cleaned_data)

    prepared_data = handle_duplicates(
        cleaned_data
    )

    train_data, validation_data, test_data = split_dataset(
        prepared_data
    )

    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }

    validate_splits(
        prepared_data,
        splits,
    )

    print_split_summary(
        train_data,
        validation_data,
        test_data,
    )

    save_datasets(
        train_data,
        validation_data,
        test_data,
    )

    print(
        "\nDataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()
