from pathlib import Path

from loguru import logger
import typer

import kagglehub

from src.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcardfraud.csv",
):
    logger.info("Creating dataset...")

    # Check if the raw data exists.
    # If raw data does not exist, try to download from kagglehub
    if not input_path.exists():
        logger.warning(f"Input file not found in raw directory: {input_path}")
        logger.info("Attempting to download dataset from Kaggle...")

        try:
            path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            logger.info(f"Dataset downloaded to: {path}")

            # Assume file is called 'creditcard.csv' inside the downloaded folder
            downloaded_file = Path(path) / "creditcard.csv"

            # Check if raw data have been downloaded correctly.
            if not downloaded_file.exists():
                logger.error("Downloaded dataset file not found in the expected location.")
                raise typer.Exit(code=1)

            # Ensure RAW_DATA_DIR exists
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Copy file to expected input_path
            input_path.write_bytes(downloaded_file.read_bytes())
            logger.success(f"Copied dataset to raw folder: {input_path}")

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise typer.Exit(code=1)
    else:
        logger.info(f"Raw dataset available in: {input_path}")

    logger.success("Dataset creation complete.")


if __name__ == "__main__":
    app()
