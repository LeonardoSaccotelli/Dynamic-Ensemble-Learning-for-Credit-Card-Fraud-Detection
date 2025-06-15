from pathlib import Path

from loguru import logger
import typer
import numpy as np

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.io_utils import load_csv, save_csv

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "creditcardfraud_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "creditcardfraud_features.csv",
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

    logger.info("Generating features from dataset...")
    logger.info(f"Loading interim dataset from: {input_path}")
    df = load_csv(input_path, delimiter=",")

    logger.info("Starting feature engineering...")

    logger.info("Computing log1p of 'Amount'...")
    df["Log_Amount"] = np.log1p(df["Amount"])

    logger.info("Dropping 'Amount' and 'Time' columns...")
    df.drop(columns=["Amount", "Time"], inplace=True)

    # Save cleaned dataset
    logger.info(f"Saving final features dataset to: {output_path}")
    save_csv(df, output_path, index=False)

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
