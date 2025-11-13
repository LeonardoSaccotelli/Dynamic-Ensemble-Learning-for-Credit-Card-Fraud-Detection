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
):
    """
    Perform deterministic, leakage-safe feature engineering.

    This function computes a log-transformed version of the 'Amount' column and
    removes 'Amount' and 'Time' to prevent data leakage.

    Parameters
    ----------
    input_path : Path
        Path to the cleaned dataset.
    output_path : Path
        Path to save the processed dataset with engineered features.
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

    logger.success("Completed fraud_dynamic_ensemble/features.py")


if __name__ == "__main__":
    app()
