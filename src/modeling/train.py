from pathlib import Path

import pandas as pd
from loguru import logger
import typer
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.config import  RANDOM_STATE, CV_N_SPLITS, CV_N_REPEATS, DSEL_SIZE
from src.config import BASE_MODELS, DES_MODELS
from src.utils.io_utils import  load_csv, save_csv
from src.modeling.train_utils import train_and_evaluate
from src.modeling.models import get_model_and_search_space

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "creditcardfraud_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("Training models...")

    # Load features dataset
    logger.info(f"Loading features dataset from: {features_path}")
    df = load_csv(features_path, delimiter=",")
    logger.info(f"Initial shape (rows, columns): {df.shape}")

    # Initial shuffle of the data (frac=1.0 means that all rows will be kept but shuffled)
    logger.info(f"Shuffling dataset with random_state={RANDOM_STATE}")
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split data into features and labels
    logger.info(f"Splitting dataset into features and labels")
    X, y = df.drop(["Class"], axis=1), df["Class"]
    logger.info(f"Shape of X: {X.shape} - Shape of y: {y.shape}")

    # Fix the evaluation strategy: RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    cv_outer = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)
    splits = list(cv_outer.split(X, y))

    # Starting training models
    logger.info("Starting training models...")

    # List to store all the metrics for all the iterations
    resubstitution_metrics_summary = []
    test_metrics_summary = []

    for run_id, (train_idx, test_idx) in enumerate(splits):

        # TODO CAMBIARE SE DECIDIAMO DI NON USARE 10 X 10
        iteration_idx, fold_idx = divmod(run_id, CV_N_SPLITS)
        logger.info(f"Starting training models for [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}]")

        # Split the data into training and test data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Split the data into training and DSEL for DS techniques
        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=DSEL_SIZE,
                                                            random_state=RANDOM_STATE, stratify=y_train)

        logger.info(f"X_train_i, X_dsel_i, X_test_i shape: {X_train.shape}, {X_dsel.shape}, {X_test.shape}")
        for name, target in zip(["y_train", "y_dsel", "y_test"], [y_train, y_dsel, y_test]):
            stats = pd.DataFrame({"count": target.value_counts(), "percentage": target.value_counts(normalize=True) * 100}).round(3)
            logger.info(f"Stratification class balance for {name}:\n{stats}")

        # Dict to collect base models already fitted
        base_models_fitted = {}

        ################################# TRAINING BASE MODELS #################################
        for model_name in BASE_MODELS:
            logger.info(f"Training base model: {model_name}")

            # Get the base model to tune with its hyperparameter search space
            base_model, search_space = get_model_and_search_space(model_name, RANDOM_STATE)

            #pipeline_base_model = build_base_model_pipeline()


            # Tune the base model, fit on the training folds and evaluate on the test fold
            fitted_model, resubstitution_metrics, test_metrics  = train_and_evaluate()
            base_models_fitted[model_name] = fitted_model

            # Log resubstitution metrics with iteration and fold
            row_resubstitution_metrics = {
                "iteration": iteration_idx + 1,
                "fold": fold_idx + 1,
                "model": model_name,
                **resubstitution_metrics
            }
            resubstitution_metrics_summary.append(row_resubstitution_metrics)

            # Log test metrics with iteration and fold
            row_test_metrics = {
                "iteration": iteration_idx + 1,
                "fold": fold_idx + 1,
                "model": model_name,
                **test_metrics
            }
            test_metrics_summary.append(row_test_metrics)





        ################################# TRAINING DES MODELS  #################################

        logger.success(f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}]")


    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
