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
    Clean the raw credit card fraud dataset by removing duplicate rows and writing an INTERIM CSV.

    This entry point performs the minimum cleaning currently required by the project. It:

    1) loads the RAW CSV at ``input_path``,
    2) logs dataset shape and class distribution (using ``target``) before cleaning,
    3) removes duplicate rows (all columns; first occurrence kept),
    4) logs the number of duplicates removed, updated shape, and class distribution after cleaning,
    5) ensures ``INTERIM_DATA_DIR`` exists, then writes the cleaned CSV to ``output_path``.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Path to the RAW dataset CSV to clean. Defaults to ``RAW_DATA_DIR / RAW_FILENAME``.
    output_path : pathlib.Path, optional
        Destination path for the cleaned CSV. Defaults to ``INTERIM_DATA_DIR / INTERIM_FILENAME``.
    target : str, optional
        Name of the target column used to compute and log class distribution.
        Defaults to ``"Class"``.

    Returns
    -------
    None
        This function returns nothing. It performs logging and writes a CSV file to disk.

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist.
    KeyError
        If ``target`` is not present in the loaded dataframe and downstream helpers
        (e.g., class statistics computation) require it.
    pandas.errors.EmptyDataError
        If ``input_path`` is empty or contains no parsable columns.
    pandas.errors.ParserError
        If the CSV is malformed and cannot be parsed by pandas.
    PermissionError
        If the output directory cannot be created or the cleaned CSV cannot be written due to
        insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.

    Notes
    -----
    - Deduplication is performed across all columns (``subset=None``) and keeps the first
      occurrence (``keep="first"``).
    - The function ensures ``INTERIM_DATA_DIR`` exists (not necessarily ``output_path.parent``
      if a different path is provided).
    - All outputs are produced via side effects (logging + file I/O).

    Examples
    --------
    Run using the module script (paths shown as examples)::

        python fraud_dynamic_ensemble/cleaning.py --input-path data/raw/credit_card_fraud_sampling.csv --output-path data/interim/credit_card_fraud_cleaned.csv --target Class
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
