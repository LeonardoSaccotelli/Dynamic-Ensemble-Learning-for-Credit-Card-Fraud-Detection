from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from fraud_dynamic_ensemble.config import (
    INTERIM_DATA_DIR,
    INTERIM_FILENAME,
    RAW_DATA_DIR,
    RAW_FILENAME,
)
from fraud_dynamic_ensemble.data_preparation.data_clean import remove_duplicates
from fraud_dynamic_ensemble.data_preparation.sampling import get_class_stats

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / RAW_FILENAME,
    output_path: Path = INTERIM_DATA_DIR / INTERIM_FILENAME,
    target: str = "Class",
) -> None:
    """
    Clean the raw credit card fraud dataset and save a **deduplicated** CSV.

    This script performs only the cleaning steps that are currently needed,
    given that the Data Understanding phase already confirmed the absence of
    NA rows. Specifically, it:
      1) Loads the RAW dataset.
      2) Reports per-column missing values (for transparency; no NA-row drop).
      3) Removes duplicate rows.
      4) Logs class distribution **before** and **after** deduplication using ``target``.
      5) Ensures the INTERIM output directory exists and writes the cleaned CSV.

    Parameters
    ----------
    input_path : Path, optional
        Path to the RAW dataset CSV to be cleaned. Default:
        ``RAW_DATA_DIR / RAW_FILENAME``.
    output_path : Path, optional
        Destination path for the cleaned dataset CSV. Default:
        ``INTERIM_DATA_DIR / INTERIM_FILENAME``.
    target : str, optional
        Target column used for class distribution logs. Default: ``"Class"``.

    Returns
    -------
    None
        Side effects only (logging and file I/O).

    Raises
    ------
    typer.Exit
        If the input file is missing or the target column is not found.
    pandas.errors.EmptyDataError
        If the input file is empty or has no columns to parse.
    pandas.errors.ParserError
        If the CSV is malformed and cannot be parsed.
    PermissionError
        If the cleaned file cannot be written due to insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.
    """

    logger.info("Running fraud_dynamic_ensemble/cleaning.py ...")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the sampling step first, e.g.: `python fraud_dynamic_ensemble/dataset_sampling.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading RAW dataset at path:\n\t{input_path}")
    df = pd.read_csv(input_path, header=0, sep=",")

    # Report BEFORE cleaning phase
    counts, perc, rows, cols = get_class_stats(df, target)
    logger.info(f"Raw shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (before cleaning):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    ################################# DUPLICATED #################################
    # Remove duplicated rows
    df = remove_duplicates(df, subset=None, keep="first", inplace=False, ignore_index=True)

    # Report AFTER removing duplicates
    counts_after, perc_after, rows_after, cols = get_class_stats(df, target)
    duplicates = rows - rows_after

    logger.info(
        f"Number of duplicates removed: {duplicates}/{rows} ({duplicates / rows * 100:.2f}%)"
    )

    logger.info(f"Raw shape: rows={rows_after}, cols={cols}")
    logger.info("Class distribution (after cleaning):")
    for cls in counts_after.index:
        logger.info(f"  class={cls}: count={counts_after[cls]}, perc={perc_after[cls]:.6f}")

    # Ensure INTERIM_DATA_DIR exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Store the cleaned dataset
    df.to_csv(output_path, index=False, sep=",")
    logger.success(f"Wrote cleaned dataset to path:\n\t{output_path}")

    logger.success("Running fraud_dynamic_ensemble/cleaning.py COMPLETED!")


if __name__ == "__main__":
    app()
