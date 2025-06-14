from pathlib import Path

from loguru import logger
import typer

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from src.utils.io_utils import  load_csv, save_csv

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcardfraud.csv",
    interim_path: Path = INTERIM_DATA_DIR / "creditcardfraud_cleaned.csv"
):
    logger.info("Cleaning dataset...")

    logger.info(f"Loading raw dataset from: {input_path}")
    df = load_csv(input_path, delimiter=",")
    logger.info(f"Initial shape (rows, columns): {df.shape}")

    # Check for null columns
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)

    if null_cols.empty:
        logger.info("No missing values detected in any column.")
    else:
        logger.warning("Missing values per column:")
        for col, n_missing in null_cols.items():
            logger.warning(f" - {col}: {n_missing} ({n_missing / df.shape[0] * 100:.3f}%)")

    # Check for rows with more than 50% of null values
    n_before = df.shape[0]
    row_null_percent = df.isnull().mean(axis=1) * 100
    rows_to_drop = df[row_null_percent > 50].index
    df.drop(index=rows_to_drop, inplace=True)
    n_dropped = n_before - df.shape[0]
    logger.info(f"Rows with >50% NaNs removed: {n_dropped}/{n_before} "
                f"({n_dropped / n_before * 100:.3f}%)")

    # Check class distribution before removing duplicates
    logger.info("Class distribution before removing duplicates:")
    class_counts = df['Class'].value_counts()
    total = class_counts.sum()
    for label, count in class_counts.items():
        logger.info(f" - Class {label}: {count} ({count / total * 100:.3f}%)")

    # Remove duplicated rows
    n_samples = df.shape[0]
    df.drop_duplicates(ignore_index=True, inplace=True)
    duplicates = n_samples - df.shape[0]
    logger.info(f"Number of duplicates removed: {duplicates}/{n_samples} "
                f"({duplicates / n_samples * 100:.3f}%)")

    # Check class distribution after removing duplicates
    logger.info("Class distribution after removing duplicates:")
    class_counts = df['Class'].value_counts()
    total = class_counts.sum()
    for label, count in class_counts.items():
        logger.info(f" - Class {label}: {count} ({count / total * 100:.3f}%)")

    # Save cleaned dataset
    logger.info(f"Saving cleaned dataset to: {interim_path}")
    save_csv(df, interim_path, index=False)
    logger.success("Data cleaning complete.")


if __name__ == "__main__":
    app()
