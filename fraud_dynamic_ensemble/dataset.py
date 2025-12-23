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
    Ensure the external credit card fraud dataset is available locally, downloading it if missing.

    This entry point verifies whether the dataset CSV exists at ``output_path``. If it is not
    present, it attempts to download the Kaggle dataset ``mlg-ulb/creditcardfraud`` using
    ``kagglehub.dataset_download()``, checks that the expected file ``creditcard.csv`` exists
    in the downloaded directory, ensures ``EXTERNAL_DATA_DIR`` exists, and copies the CSV to
    ``output_path``. If the file already exists, it logs that no action is needed.

    Parameters
    ----------
    output_path : pathlib.Path, optional
        Destination path for the external dataset CSV. Defaults to
        ``EXTERNAL_DATA_DIR / EXTERNAL_FILENAME``.

    Returns
    -------
    None
        This function returns nothing. It performs side effects only (download, directory
        creation, file copy, and logging).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if the dataset download fails or if the expected
        ``creditcard.csv`` file is not found after download.
    PermissionError
        If the output directory cannot be created or the destination file cannot be written due
        to insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file copying.
    Exception
        Any unexpected exception raised by the Kaggle download utility or file I/O may be caught
        and converted into ``typer.Exit(code=1)`` by this function's error handling.

    Notes
    -----
    - ``kagglehub`` may require Kaggle credentials/configuration to be available in the runtime
      environment.
    - The function assumes the downloaded Kaggle package contains a file named
      ``creditcard.csv`` at the top level of the download directory.
    - The function ensures ``EXTERNAL_DATA_DIR`` exists (not necessarily ``output_path.parent``
      if a different path is provided).
    - The implementation uses logging for progress and outcomes and does not return data.

    Examples
    --------
    Run using the module script (paths shown as examples)::

        python fraud_dynamic_ensemble/dataset.py --output-path data/external/creditcardfraud.csv
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
