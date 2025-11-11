from __future__ import annotations

from pathlib import Path

import kagglehub
from loguru import logger
import typer

from fraud_dynamic_ensemble.config import EXTERNAL_DATA_DIR, EXTERNAL_FILENAME

app = typer.Typer()


@app.command()
def main(
    output_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_FILENAME,
) -> None:
    """
    Ensure the **external** credit card fraud dataset is available locally,
    downloading it from Kaggle if missing.

    The function checks whether ``output_path`` exists. If not, it downloads the
    Kaggle dataset ``mlg-ulb/creditcardfraud`` via ``kagglehub.dataset_download()``,
    verifies that the expected CSV file (``creditcard.csv``) is present in the
    downloaded directory, creates ``EXTERNAL_DATA_DIR`` if needed, and copies the
    file to ``output_path``. Progress and outcomes are logged via Loguru.

    Parameters
    ----------
    output_path : pathlib.Path, optional
        Destination path for the **external** dataset CSV. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_FILENAME``.

    Returns
    -------
    None
        Side effects only (download, directory creation, file copy, and logging).

    Raises
    ------
    typer.Exit
        Raised with code ``1`` if the dataset cannot be downloaded or the expected
        CSV is not found after download.

    Notes
    -----
    - ``kagglehub`` may require prior Kaggle credentials/configuration.
    - The expected filename inside the Kaggle package is ``creditcard.csv``.
    """

    logger.info("Running fraud_dynamic_ensemble/dataset.py ...")

    # Check if the external data exists. If data does not exist, try to download from kagglehub.
    if not output_path.exists():
        logger.warning(f"Dataset not found at path:\n\t{output_path}")
        logger.info("Attempting to download dataset from Kaggle...")

        try:
            kaggle_dest_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            logger.success(f"Dataset downloaded to:\n\t{kaggle_dest_path}")

            # Assume file is called 'creditcard.csv' inside the downloaded folder
            downloaded_file = Path(kaggle_dest_path) / "creditcard.csv"

            # Check if data have been downloaded correctly.
            if not downloaded_file.exists():
                logger.error(
                    f"Downloaded file not found at the expected path:\n\t{downloaded_file}"
                )
                raise typer.Exit(code=1)

            # Ensure EXTERNAL_DATA_DIR exists
            EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Copy file to expected input_path
            output_path.write_bytes(downloaded_file.read_bytes())
            logger.success(f"Copied dataset to path:\n\t{output_path}")

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Dataset ALREADY available at path:\n\t{output_path}")

    logger.success("Running fraud_dynamic_ensemble/dataset.py COMPLETED!")


if __name__ == "__main__":
    app()
