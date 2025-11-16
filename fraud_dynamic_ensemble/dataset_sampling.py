from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from fraud_dynamic_ensemble.config import (
    EXTERNAL_DATA_DIR,
    EXTERNAL_FILENAME,
    FRAC,
    N_ROWS,
    POLICY,
    RANDOM_STATE,
    RATIO,
    RAW_DATA_DIR,
    RAW_FILENAME,
)
from fraud_dynamic_ensemble.data_preparation.sampling import (
    apply_sampling,
    get_class_stats,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_FILENAME,
    output_path: Path = RAW_DATA_DIR / RAW_FILENAME,
    target: str = "Class",
    policy: str = POLICY,
    n_rows: int | None = N_ROWS,
    frac: float | None = FRAC,
    ratio: int | None = RATIO,
    seed: int = RANDOM_STATE,
) -> None:
    """
    CLI entry-point (dataset_sampling.py) to create a RAW subsample from the EXTERNAL full dataset.

    Workflow
    --------
    1) Load the external dataset (CSV).
    2) Log **pre-sampling** class statistics.
    3) Apply the chosen sampling strategy via ``apply_sampling``.
    4) Log **post-sampling** class statistics.
    5) Write the subsample to ``output_path`` as CSV.

    Parameters
    ----------
    input_path : pathlib.Path, default: EXTERNAL_DATA_DIR / EXTERNAL_FILENAME
        Path to the **external** full dataset (CSV).
    output_path : pathlib.Path, default: RAW_DATA_DIR / RAW_FILENAME
        Destination path for the **raw subsample** (CSV).
    target : str, default: "Class"
        Name of the target column.
    policy : {'random', 'stratified', 'keep_all_minority'}, default: POLICY
        Sampling policy (case-insensitive).
    n_rows : int or None, optional, default: N_ROWS
        Absolute number of rows requested (mutually exclusive with ``frac``).
    frac : float or None, optional, default: FRAC
        Fraction of the dataset in ``(0, 1]`` (mutually exclusive with ``n_rows``).
    ratio : int or None, optional, default: RATIO
        Only for ``'keep_all_minority'``: majority-per-minority ratio (e.g., 50 → 1:50).
    seed : int, default: RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    None
        Side effects only (logging, sampling, CSV writing).

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist or ``target`` is missing from the dataset.
    ValueError
        May be raised by the underlying sampling functions for invalid combinations
        (e.g., both ``n_rows`` and ``frac``, non-binary target for keep-all-minority, etc.).

    Notes
    -----
    - I/O is **CSV-only** by design.
    """

    logger.info("Running fraud_dynamic_ensemble/dataset_sampling.py ...")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error("Run the downloader first, e.g.: `python fraud_dynamic_ensemble/dataset.py`.")
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading EXTERNAL dataset at path:\n\t{input_path}")
    df = pd.read_csv(input_path, header=0, sep=",")

    if target not in df.columns:
        logger.error(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
        raise typer.Exit(code=1)

    # Report BEFORE
    counts, perc, rows, cols = get_class_stats(df, target)
    logger.info(f"External shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (before sampling):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    # Normalize policy for logging consistency
    policy_key = (policy or "").strip().lower()
    logger.info(
        f"Sampling plan → policy='{policy_key}', seed={seed}, "
        f"n_rows={n_rows}, frac={frac}, ratio={ratio}"
    )

    # Apply sampling
    sample = apply_sampling(
        df,
        policy=policy_key,
        target=target,  # used by 'stratified' & 'keep_all_minority'
        n_rows=n_rows,
        frac=frac,
        ratio=ratio,  # only for 'keep_all_minority'
        seed=seed,
    )

    logger.info(f"Sample prepared: rows={len(sample)}, cols={sample.shape[1]}")

    # Report AFTER
    counts_after, perc_after, rows_after, cols_after = get_class_stats(sample, target)
    logger.info(f"Shape (after): rows={rows_after}, cols={cols_after}")
    logger.info("Class distribution (after sampling):")
    for cls in counts_after.index:
        logger.info(f"  class={cls}: count={counts_after[cls]}, perc={perc_after[cls]:.6f}")

    # Ensure RAW_DATA_DIR exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Store the sampling dataset
    sample.to_csv(output_path, index=False, sep=",")
    logger.success(f"Wrote RAW subsampled dataset to path:\n\t{output_path}")

    logger.success("Running fraud_dynamic_ensemble/dataset_sampling.py COMPLETED!")


if __name__ == "__main__":
    app()
