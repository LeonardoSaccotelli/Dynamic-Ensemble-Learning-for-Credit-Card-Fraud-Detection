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
    Deterministic, leakage-safe feature engineering for the fraud dataset.

    This CLI entrypoint loads the **cleaned** dataset, applies purely row-wise
    transformations, and writes a processed CSV:

    1) Add ``Amount_log1p`` via ``np.log1p(Amount)`` and drop ``Amount``.
    2) Add ``Time_sin`` and ``Time_cos`` using the given ``period`` and drop ``Time``.
    3) Move the target column to the last position for convenience.
    4) Persist the result to ``output_path``.

    All transforms are element-wise (no data-wide statistics), so they do **not**
    introduce data leakage. Any scaling/standardization should be handled later
    inside the modeling pipeline.

    Parameters
    ----------
    input_path : pathlib.Path, default: ``INTERIM_DATA_DIR / INTERIM_FILENAME``
        Path to the cleaned (interim) dataset CSV.
    output_path : pathlib.Path, default: ``PROCESSED_DATA_DIR / PROCESSED_FILENAME``
        Destination path for the processed dataset CSV with engineered features.
    target : str, default: ``"Class"``
        Name of the target column; logged before/after transformations and moved last.
    period : int, default: ``DAY_SECONDS``
        Period used to encode ``Time`` into ``Time_sin``/``Time_cos``. Must match the
        unit of ``Time`` (e.g., seconds-in-day = 86,400).

    Returns
    -------
    None
        Side-effecting function: logging and file I/O.

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist.
    KeyError
        If required columns (e.g., ``"Amount"``, ``"Time"``, or ``target``) are missing.
    ValueError
        Propagated from row-wise transformers (e.g., invalid ``period`` or
        ``Amount < -1`` for ``log1p``).
    pandas.errors.EmptyDataError
        If the input CSV is empty.
    pandas.errors.ParserError
        If the CSV cannot be parsed.
    PermissionError
        If the output CSV cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during directory creation or file writing.

    Notes
    -----
    - Row-wise transforms only → **no leakage**.
    - ``Amount_log1p`` uses ``np.log1p`` (defined for ``x >= -1``).
    - ``Time_sin``/``Time_cos`` are computed as ``sin(2π⋅x/period)`` and ``cos(2π⋅x/period)``.
    - The function ensures ``PROCESSED_DATA_DIR`` exists before saving.
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

    # --- Transform Amount using log_1p function ---
    logger.info("Computing log1p of 'Amount'...")
    df = transform_log1p(df, cols="Amount", drop_original=True)

    # --- Transform Time using trigonometric features (sin-cos) ---
    logger.info("Computing sin,cos of 'Time'...")
    df = transform_sin_cos(df, "Time", period=period, drop_original=True)

    # --- Move the Class column as last column
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
