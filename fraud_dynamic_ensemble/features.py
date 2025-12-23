from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from fraud_dynamic_ensemble.config import (
    DAY_SECONDS,
    INTERIM_DATA_DIR,
    INTERIM_FILENAME,
    PROCESSED_DATA_DIR,
    PROCESSED_FILENAME,
)
from fraud_dynamic_ensemble.data_preparation.data_construct import (
    transform_log1p,
    transform_sin_cos,
)
from fraud_dynamic_ensemble.data_preparation.sampling import (
    get_class_stats,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / INTERIM_FILENAME,
    output_path: Path = PROCESSED_DATA_DIR / PROCESSED_FILENAME,
    target: str = "Class",
    period: int = DAY_SECONDS,
) -> None:
    """
    Perform deterministic, leakage-safe feature engineering and write the processed fraud dataset.

    This CLI entry point loads the cleaned (interim) dataset and applies strictly row-wise
    transformations that do not depend on global dataset statistics:

    1) Creates ``Amount_log1p`` using ``np.log1p`` applied to the ``Amount`` column and drops
       the original ``Amount`` column.
    2) Encodes the ``Time`` column into periodic features ``Time_sin`` and ``Time_cos`` using
       the provided ``period`` and drops the original ``Time`` column.
    3) Reorders columns so that the target column appears last.
    4) Ensures the processed data directory exists and writes the resulting dataset to
       ``output_path`` as a CSV.

    Because all transformations are element-wise, the procedure is leakage-safe. Any scaling,
    standardization, resampling, or model-dependent preprocessing is expected to be performed
    later within the modeling pipeline.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Path to the cleaned interim dataset CSV. Defaults to
        ``INTERIM_DATA_DIR / INTERIM_FILENAME``.
    output_path : pathlib.Path, optional
        Destination path for the processed dataset CSV. Defaults to
        ``PROCESSED_DATA_DIR / PROCESSED_FILENAME``.
    target : str, optional
        Name of the target column. Used for class-distribution logging and moved to the last
        column position in the output. Defaults to ``"Class"``.
    period : int, optional
        Period used to compute the trigonometric encoding of ``Time`` into ``Time_sin`` and
        ``Time_cos``. Must be expressed in the same unit as the ``Time`` column
        (e.g., seconds per day = 86,400). Defaults to ``DAY_SECONDS``.

    Returns
    -------
    None
        This function returns nothing. It performs side effects only (logging, directory
        creation, and CSV writing).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if ``input_path`` does not exist.
    pandas.errors.EmptyDataError
        If the input CSV is empty or has no columns to parse.
    pandas.errors.ParserError
        If the input CSV is malformed and cannot be parsed.
    KeyError
        If required columns are missing (for example: ``"Amount"``, ``"Time"``, or ``target``),
        potentially raised by downstream transformation utilities.
    ValueError
        Propagated from transformation utilities for invalid values or configuration (for
        example: invalid ``period`` or values outside the domain of ``log1p``).
    PermissionError
        If the processed directory cannot be created or the output CSV cannot be written due to
        insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.

    Notes
    -----
    - All transformations are row-wise and deterministic, so they do not introduce data leakage.
    - ``Amount_log1p`` is created via ``np.log1p`` (defined for inputs ``>= -1``).
    - ``Time_sin`` and ``Time_cos`` are computed using the standard periodic encoding:
      ``sin(2π·t/period)`` and ``cos(2π·t/period)``.
    - The target column is moved to the last position to simplify downstream inspection and
      pipeline construction.

    Examples
    --------
    Run the script with the default daily period (seconds-in-day)::

        python fraud_dynamic_ensemble/features.py

    Run the script with a custom period (only if ``Time`` is expressed in that unit)::

        python fraud_dynamic_ensemble/features.py --period 86400
    """

    logger.info("Running fraud_dynamic_ensemble/features.py ...")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the cleaning step first, e.g.: `python fraud_dynamic_ensemble/cleaning.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading CLEANED dataset at path:\n\t{input_path}")
    df = pd.read_csv(input_path, header=0, sep=",")

    # Report BEFORE feature engineering phase
    counts, perc, rows, cols = get_class_stats(df, target)
    logger.info(f"Raw shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (before feature engineering):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    logger.info("Starting feature engineering...")

    ################################# TRANSFORM AMOUNT #################################
    # Transform Amount using log_1p function
    logger.info("Computing log1p of 'Amount'...")
    df = transform_log1p(df, cols="Amount", drop_original=True)

    ################################# TRANSFORM TIME #################################
    # Transform Time using trigonometric features (sin-cos)
    logger.info("Computing sin,cos of 'Time'...")
    df = transform_sin_cos(df, "Time", period=period, drop_original=True)

    ################################# SORT COLUMNS #################################
    # Move the Class column as last column
    df = df[[col for col in df.columns if col not in [target]] + [target]]

    # Report AFTER feature engineering phase
    counts, perc, rows, cols = get_class_stats(df, target)
    logger.info(f"Raw shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (after feature engineering):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    # Ensure PROCESSED_DATA_DIR exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Store the features dataset
    df.to_csv(output_path, index=False, sep=",")
    logger.success(f"Wrote features dataset to path:\n\t{output_path}")

    logger.success("Running fraud_dynamic_ensemble/features.py COMPLETED!")


if __name__ == "__main__":
    app()
