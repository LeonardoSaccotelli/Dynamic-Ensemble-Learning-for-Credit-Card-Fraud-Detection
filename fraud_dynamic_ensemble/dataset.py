from __future__ import annotations

from pathlib import Path

import kagglehub
from loguru import logger
import typer

from fraud_dynamic_ensemble.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    output_path: Path = RAW_DATA_DIR / "creditcardfraud.csv",
) -> None:
    """
    Ensure that the credit card fraud dataset is available locally, downloading it from Kaggle if missing.

    The function checks whether ``output_path`` exists. If not, it attempts to download the dataset
    ``mlg-ulb/creditcardfraud`` via ``kagglehub.dataset_download``, verifies the expected CSV file
    (``creditcard.csv``) is present in the downloaded directory, creates ``RAW_DATA_DIR`` if needed,
    and copies the file to ``output_path``. Progress and outcomes are logged.

    Parameters
    ----------
    output_path : Path, optional
        Expected file output path of the raw dataset (default is ``RAW_DATA_DIR / "creditcardfraud.csv"``).

    Returns
    -------
    None
        All effects are side effects (download, directory creation, file copy, and logging).

    Raises
    ------
    typer.Exit
        If the dataset cannot be downloaded or the expected file is not found after download.
        The exit code is set to ``1``.
    """

    logger.info("Running fraud_dynamic_ensemble/dataset.py ...")

    # Check if the raw data exists.
    # If data does not exist, try to download from kagglehub.
    if not output_path.exists():
        logger.warning(f"Data not found in directory: {output_path}")
        logger.info("Attempting to download dataset from Kaggle...")

        try:
            kaggle_dest_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            logger.success(f"Dataset downloaded to: {kaggle_dest_path}")

            # Assume file is called 'creditcard.csv' inside the downloaded folder
            downloaded_file = Path(kaggle_dest_path) / "creditcard.csv"

            # Check if data have been downloaded correctly.
            if not downloaded_file.exists():
                logger.error("Downloaded dataset file not found in the expected location.")
                raise typer.Exit(code=1)

            # Ensure RAW_DATA_DIR exists
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Copy file to expected input_path
            output_path.write_bytes(downloaded_file.read_bytes())
            logger.success(f"Copied dataset to folder: {output_path}")

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Original dataset ALREADY available in: {output_path}")

    logger.success("Running fraud_dynamic_ensemble/dataset.py COMPLETED!")


if __name__ == "__main__":
    app()
