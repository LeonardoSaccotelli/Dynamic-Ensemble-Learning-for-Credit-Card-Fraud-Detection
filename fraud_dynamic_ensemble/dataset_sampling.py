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
    Create a RAW subsample from the EXTERNAL credit card fraud dataset.

    This CLI entry point loads the full external CSV dataset, logs pre-sampling class
    distribution statistics, applies a sampling strategy via ``apply_sampling``, logs the
    post-sampling class distribution, ensures the RAW data directory exists, and writes the
    resulting subsample to ``output_path`` as a CSV.

    The sampling behavior is controlled by ``policy`` and the mutually exclusive size controls
    ``n_rows`` and ``frac``. When using the ``keep_all_minority`` policy, ``ratio`` determines
    the majority-per-minority sampling ratio.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Path to the external (full) dataset CSV. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_FILENAME``.
    output_path : pathlib.Path, optional
        Destination path for the raw subsampled dataset CSV. Defaults to
        ``RAW_DATA_DIR / RAW_FILENAME``.
    target : str, optional
        Name of the target column used to compute class statistics and to drive sampling policies
        that require labels. Defaults to ``"Class"``.
    policy : str, optional
        Sampling policy identifier (case-insensitive) forwarded to ``apply_sampling``.
        Typical values include ``"random"``, ``"stratified"``, and ``"keep_all_minority"``.
        Defaults to ``POLICY``.
    n_rows : int or None, optional
        Absolute number of rows requested for the subsample. Mutually exclusive with ``frac``.
        Defaults to ``N_ROWS``.
    frac : float or None, optional
        Fraction of the dataset to sample in ``(0, 1]``. Mutually exclusive with ``n_rows``.
        Defaults to ``FRAC``.
    ratio : int or None, optional
        Majority-per-minority ratio used only when ``policy`` is ``"keep_all_minority"``
        (e.g., ``50`` corresponds to a 1:50 minority:majority ratio). Defaults to ``RATIO``.
    seed : int, optional
        Random seed forwarded to the sampling routine for reproducibility. Defaults to
        ``RANDOM_STATE``.

    Returns
    -------
    None
        This function returns nothing. It performs side effects only (logging, sampling, and CSV
        writing).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if ``input_path`` does not exist or if ``target`` is not a
        column in the loaded dataset.
    pandas.errors.EmptyDataError
        If the input CSV is empty or has no columns to parse.
    pandas.errors.ParserError
        If the input CSV is malformed and cannot be parsed.
    ValueError
        May be raised by ``apply_sampling`` for invalid configurations (for example: both
        ``n_rows`` and ``frac`` provided, invalid ``policy``, invalid ``frac`` range, invalid
        ``ratio`` for keep-all-minority, or non-binary targets when required).
    PermissionError
        If the RAW directory cannot be created or the output CSV cannot be written due to
        insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.

    Notes
    -----
    - I/O is intentionally CSV-only in this CLI workflow.
    - The function normalizes ``policy`` to lowercase before calling ``apply_sampling`` for
      consistent behavior and logging.
    - ``ratio`` is meaningful only for the ``keep_all_minority`` policy; other policies may
      ignore it.
    - The function logs class distribution before and after sampling using ``get_class_stats``.

    Examples
    --------
    Run the script to produce a stratified subsample by fraction::

        python fraud_dynamic_ensemble/dataset_sampling.py --policy stratified --frac 0.10

    Run the script to keep all minority samples and cap majority at 50× minority::

        python fraud_dynamic_ensemble/dataset_sampling.py --policy keep_all_minority --ratio 50
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
