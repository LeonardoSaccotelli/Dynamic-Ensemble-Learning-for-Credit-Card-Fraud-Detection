from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger
import pandas as pd
import typer

from fraud_dynamic_ensemble.config import (
    EXTERNAL_DATA_DIR,
    FRAC,
    N_ROWS,
    POLICY,
    RANDOM_STATE,
    RAW_DATA_DIR,
)
from fraud_dynamic_ensemble.utils.sampling import (
    apply_sampling,
    derive_target_size,
    get_class_stats,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "creditcardfraud.csv",
    output_path: Path = RAW_DATA_DIR / "credit_card_fraud_subsample.csv",
    target: str = "Class",
    policy: Literal["random", "stratified", "keep_all_minority"] = POLICY,
    n_rows: int | None = N_ROWS,
    frac: float | None = FRAC,
    seed: int = RANDOM_STATE,
) -> None:
    """
    CLI entry-point to create a RAW subsample from the EXTERNAL full dataset.

    The function:
    1) Loads the external dataset,
    2) Logs **pre-sampling** class stats,
    3) Validates the requested sampling policy and derives the exact target size
       from either ``n_rows`` or ``frac``,
    4) Applies the chosen sampling strategy,
    5) Logs **post-sampling** class stats,
    6) Writes the subsample to ``output_path`` as CSV.

    Parameters
    ----------
    input_path : pathlib.Path, default: EXTERNAL_DATA_DIR / "creditcardfraud.csv"
        Path to the **external** full dataset (CSV).
    output_path : pathlib.Path, default: RAW_DATA_DIR / "credit_card_fraud_subsample.csv"
        Destination path for the **raw subsample** (CSV).
    target : str, default: "Class"
        Name of the target column (binary in typical usage).
    policy : {'random', 'stratified', 'keep_all_minority'}, default: POLICY
        Sampling policy:
        - ``'random'``: uniform random (no class ratio preservation),
        - ``'stratified'``: preserve class proportions (scikit-learn),
        - ``'keep_all_minority'``: include **all** minority rows and fill with majority.
    n_rows : int or None, optional, default: N_ROWS
        Absolute number of rows requested. Mutually exclusive with ``frac``.
    frac : float or None, optional, default: FRAC
        Fraction of the dataset requested in ``(0, 1]``. Mutually exclusive with ``n_rows``.
    seed : int, default: RANDOM_STATE
        Random seed for reproducibility across policies.

    Returns
    -------
    None
        Side-effecting function: logs, sampling, and CSV writing.

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist or ``target`` is missing from the dataset.
    ValueError
        May be raised by ``derive_target_size`` (invalid size request/policy) or
        by ``apply_sampling`` (e.g., non-binary target for ``keep_all_minority``).

    Notes
    -----
    - I/O is **CSV-only** by design.
    - For ``'stratified'``, at least one sample per class is required in the split.
    - For ``'keep_all_minority'``, the requested size must be >= minority count.
    """

    logger.info("Running fraud_dynamic_ensemble/dataset_subsample.py ...")

    # Check if the original EXTERNAL dataset exists
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error("Run the downloader first, e.g.: `python fraud_dynamic_ensemble/dataset.py`.")
        raise typer.Exit(code=1)

    # Ensure RAW_DATA_DIR exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset from EXTERNAL_DATA_DIR and check if the target exists in the df
    logger.info(f"Loading EXTERNAL dataset at path:\n\t{input_path}")
    df = pd.read_csv(input_path, header=0, sep=",")

    if target not in df.columns:
        logger.error(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
        raise typer.Exit(code=1)

    # Print statistics for the original EXTERNAL dataset
    rows, cols = df.shape
    counts, perc = get_class_stats(df, target)

    logger.info(f"Shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (before sampling):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    # Validate policy and derive target size from n_rows or from frac
    valid_policies = {"random", "stratified", "keep_all_minority"}
    if policy not in valid_policies:
        logger.error(f"Invalid --policy '{policy}'. Choose from {sorted(valid_policies)}.")
        raise typer.Exit(code=1)

    minority_count = int(counts.min())
    target_size = derive_target_size(
        total_rows=rows,
        policy=policy,
        n_rows=n_rows,
        frac=frac,
        minority_count=minority_count,
    )

    logger.info(
        f"Sampling plan → policy='{policy}', seed={seed}, "
        f"requested_size={target_size} (minority_count={minority_count})"
    )

    # Apply requested sampling strategy
    sample = apply_sampling(
        df,
        policy=policy,
        target=target,
        target_size=target_size,
        seed=seed,
    )

    logger.info(f"Sample prepared: rows={len(sample)}, cols={sample.shape[1]}")

    # Post-sampling statistics
    rows_after, cols_after = sample.shape
    counts_after, perc_after = get_class_stats(sample, target)
    logger.info(f"Shape (after): rows={rows_after}, cols={cols_after}")
    logger.info("Class distribution (after sampling):")
    for cls in counts_after.index:
        logger.info(f"  class={cls}: count={counts_after[cls]}, perc={perc_after[cls]:.6f}")

    # Store subsampling dataset to csv
    sample.to_csv(output_path, index=False, header=True, sep=",")
    logger.success(f"Wrote RAW subsampled dataset to path:\n\t{output_path}")

    logger.success("Running fraud_dynamic_ensemble/dataset_subsample.py COMPLETED!")


if __name__ == "__main__":
    app()
