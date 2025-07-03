import numpy as np
import pandas as pd
import typer
from pathlib import Path
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.config import RUN_ID, RANDOM_STATE
from src.config import NUMERICAL_FEATURES_TO_NORMALIZE, K_BEST_TO_KEEP
from src.config import CV_N_SPLITS, CV_N_REPEATS, DSEL_SIZE
from src.config import N_ITER_TUNING, CV_TUNING, SCORING_TUNING, N_JOBS_TUNING
from src.config import BASE_MODELS, STATIC_ENS_MODELS, DES_MODELS, POOL_MODELS
from src.config import RESAMPLING_METHOD
from src.utils.io_utils import load_csv, save_dataframe_to_excel
from src.modeling.train_utils import train_and_evaluate_base_model, train_and_evaluate_ensemble_model
from src.modeling.metrics_utils import compute_voting_weights_from_dsel, append_metrics
from src.modeling.models import (get_base_model_and_search_space, get_static_ensemble_model,
                                 get_des_model, get_resampling_pipeline)
from src.modeling.pipeline import build_base_model_pipeline, get_final_selected_features

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")

app = typer.Typer()


@app.command()
def main(
        features_path: Path = PROCESSED_DATA_DIR / "creditcardfraud_features.csv",
        model_path: Path = MODELS_DIR
):
    # Set the results folder
    results_path = model_path / RUN_ID
    results_path.mkdir(parents=True, exist_ok=True)

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

    # Map column names to their corresponding indices
    feature_names = X.columns.to_list()
    column_to_index = {name: idx for idx, name in enumerate(X.columns)}

    missing = set(NUMERICAL_FEATURES_TO_NORMALIZE) - column_to_index.keys()
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    # Convert the list of column names to indices for use in ColumnTransformer
    numerical_features_indices = [column_to_index[col] for col in NUMERICAL_FEATURES_TO_NORMALIZE]

    # Convert from pandas dataframe to a numpy array to be compatible with deslib
    X, y = X.to_numpy(), y.to_numpy()

    # Fix the evaluation strategy: RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    cv_outer = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)

    # Starting training models
    logger.info("Starting training models...")

    # List to store all the metrics for all the iterations
    resubstitution_metrics_summary = []
    test_metrics_summary = []

    for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        iteration_idx, fold_idx = divmod(run_id, CV_N_SPLITS)
        logger.info(f"[ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1} - RUN_ID {run_id}]")

        # Split the data into training (9 training folds) and test data (1 test fold)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Split the data into training (for base models) and DSEL (for DS techniques) data
        X_train, X_dsel, y_train, y_dsel = train_test_split(
            X_train, y_train,
            test_size=DSEL_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

        logger.info(f"X_train_i, X_dsel_i, X_test_i shape: {X_train.shape}, {X_dsel.shape}, {X_test.shape}")
        for name, target in zip(["y_train", "y_dsel", "y_test"], [y_train, y_dsel, y_test]):
            unique, frequency = np.unique(target, return_counts=True)
            logger.info(f"Stratification class balance for {name} [class, frequency]: {unique, frequency}")

        # Fit and resample only the training data
        if RESAMPLING_METHOD is not None:
            resampler_pipeline = get_resampling_pipeline(RESAMPLING_METHOD, RANDOM_STATE)
            X_train, y_train = resampler_pipeline.fit_resample(X_train, y_train)
            logger.info(f"Stratification class balance for post-resampling y_train: {X_train.shape}, {np.unique(y_train, return_counts=True)}")

        # Collect fitted base models
        fitted_base_models = {}

        ################################# TRAINING BASE MODELS #################################
        for base_model_name in BASE_MODELS:
            logger.info(f"Training base model: {base_model_name}")

            # Get the base model to tune with its hyperparameter search space
            base_model, search_space = get_base_model_and_search_space(base_model_name, RANDOM_STATE)

            # Build the pipeline: Preprocessing + Feature Selection + Classifier
            pipeline_base_model = build_base_model_pipeline(estimator=base_model,
                                                            numerical_columns_to_scale=numerical_features_indices,
                                                            k_best=K_BEST_TO_KEEP,
                                                            random_state=RANDOM_STATE)

            # Tune the base model, fit on the training folds and evaluate on the test fold
            fitted_model, tuning_results, resubstitution_metrics, test_metrics =\
                train_and_evaluate_base_model(
                    base_model=pipeline_base_model,
                    search_space=search_space,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_iter=N_ITER_TUNING,
                    cv=CV_TUNING,
                    scoring=SCORING_TUNING,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS_TUNING,
                )

            # Store the fitted model
            fitted_base_models[base_model_name] = fitted_model

            # Extract selected feature indices and names
            selected_indices, selected_names = get_final_selected_features(fitted_model, feature_names)

            # Log resubstitution metrics with iteration and fold
            append_metrics(
                resubstitution_metrics_summary,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=base_model_name,
                metrics=resubstitution_metrics,
                data_split="resubstitution",
                fold_size=len(X_train),
                **tuning_results,
                selected_features_indices=selected_indices,
                selected_features_names=selected_names,
            )

            # Log test metrics with iteration and fold
            append_metrics(
                test_metrics_summary,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=base_model_name,
                metrics=test_metrics,
                data_split="test",
                fold_size=len(X_test),
            )

        # Validate the pool of classifiers for static and dynamic ensemble models
        invalid_pool = set(POOL_MODELS) - set(BASE_MODELS)
        if invalid_pool:
            raise ValueError(f"The following POOL_MODELS are not defined in BASE_MODELS: {invalid_pool}")
        logger.info(f"Pool of classifiers for ensemble models (Static / DES): {POOL_MODELS}")

        ################################# TRAINING STATIC ENS MODELS  #################################
        # Prepare the pool_classifiers_static_ens (name, estimator) tuples for static ensemble
        pool_classifiers_static_ens = [(name, fitted_base_models[name]) for name in POOL_MODELS]

        for static_ensemble_model_name in STATIC_ENS_MODELS:
            logger.info(f"Training static ensemble model: {static_ensemble_model_name}")

            # Compute weights on dsel dataset for VotingClassifier_weighted
            weights = None
            if static_ensemble_model_name == "VotingClassifier_weighted":
                logger.info("Computing weights for VotingClassifier_weighted...")
                weights = compute_voting_weights_from_dsel(
                    pool=pool_classifiers_static_ens,
                    X_dsel=X_dsel,
                    y_dsel=y_dsel,
                    metric="f1"
                )

            # Get the static ensemble model to train on the pool
            static_ensemble_model = get_static_ensemble_model(model_name=static_ensemble_model_name,
                                                              estimators=pool_classifiers_static_ens,
                                                              weights=weights)

            # Train the static ensemble models using the pool of classifiers
            fitted_static_ensemble_model, test_metrics = train_and_evaluate_ensemble_model(
                ensemble_model=static_ensemble_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )

            # Log test metrics with iteration and fold
            append_metrics(
                test_metrics_summary,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=static_ensemble_model_name,
                metrics=test_metrics,
                data_split="test",
                fold_size=len(X_test),
            )

        ################################# TRAINING DES MODELS  #################################
        # Prepare pool_classifiers for DESlib (list of fitted estimators)
        pool_classifiers_des = [fitted_base_models[name] for name in POOL_MODELS]

        for des_model_name in DES_MODELS:
            logger.info(f"Training DES model: {des_model_name}")

            # Get the DES model to train on the pool
            des_model = get_des_model(model_name=des_model_name, pool_classifiers=pool_classifiers_des)

            # Train the des models using the pool of classifiers
            fitted_des_model, test_metrics = train_and_evaluate_ensemble_model(
                ensemble_model=des_model,
                X_train=X_dsel,
                y_train=y_dsel,
                X_test=X_test,
                y_test=y_test
            )

            # Log test metrics with iteration and fold
            append_metrics(
                test_metrics_summary,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=des_model_name,
                metrics=test_metrics,
                data_split="test",
                fold_size=len(X_test),
            )

        logger.success(f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]")

    # Store experimental results
    save_dataframe_to_excel(pd.DataFrame(resubstitution_metrics_summary),
                            results_path / "resubstitution_metrics_summary.xlsx")
    save_dataframe_to_excel(pd.DataFrame(test_metrics_summary),
                            results_path / "test_metrics_summary.xlsx")

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
