from pathlib import Path

from loguru import logger
import typer

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from src.utils.io_utils import  load_csv, save_dataframe_to_csv

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcardfraud.csv",
    output_path: Path = INTERIM_DATA_DIR / "creditcardfraud_cleaned.csv"
) -> None:
    """
    Clean the raw credit card fraud dataset and save a deduplicated, NA-filtered CSV.

    The function loads the raw dataset, reports per-column missing values, drops rows
    containing more than 50% missing entries, removes duplicate rows, logs the class
    distribution (before and after deduplication) using the ``'Class'`` column, and
    finally writes the cleaned data to ``output_path``.

    Parameters
    ----------
    input_path : Path, optional
        Path to the raw dataset CSV to be cleaned. Defaults to
        ``RAW_DATA_DIR / "creditcardfraud.csv"``.
    output_path : Path, optional
        Destination path for the cleaned dataset CSV. Defaults to
        ``INTERIM_DATA_DIR / "creditcardfraud_cleaned.csv"``.

    Returns
    -------
    None
        All effects are side effects (logging and file I/O).

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist (surfaced by ``load_csv``).
    pandas.errors.EmptyDataError
        If the input file is empty or has no columns to parse.
    pandas.errors.ParserError
        If the CSV is malformed and cannot be parsed.
    KeyError
        If the ``'Class'`` column is missing when computing class distributions.
    PermissionError
        If the cleaned file cannot be written due to insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.
    ValueError
        If writing the cleaned CSV fails for other reasons (surfaced by
        ``save_dataframe_to_csv``).

    Examples
    --------
    Clean using default locations:

    >>> main()

    Clean from a specific raw path to a custom output:

    >>> from pathlib import Path
    >>> main(
    ...     input_path=Path("data/raw/creditcardfraud.csv"),
    ...     output_path=Path("data/interim/creditcardfraud_cleaned.csv")
    ... )
    """
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
    logger.info(f"Saving cleaned dataset to: {output_path}")
    save_dataframe_to_csv(df, output_path, index=False)

    logger.success("Data cleaning complete.")


if __name__ == "__main__":
    app()
