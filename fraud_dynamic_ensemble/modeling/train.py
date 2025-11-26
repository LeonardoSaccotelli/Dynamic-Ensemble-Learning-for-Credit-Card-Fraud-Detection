from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import typer

from fraud_dynamic_ensemble.config import (
    CV_INNER_N_SPLITS_TUNING,
    CV_OUTER_N_REPEATS,
    CV_OUTER_N_SPLITS,
    DES_MODELS,
    DSEL_SIZE,
    EXPERIMENT_DESCRIPTION,
    EXPERIMENT_NAME,
    EXPERIMENT_RUN_ID,
    FS_K_BEST_TO_KEEP,
    MODELS_DIR,
    N_ITER_TUNING,
    N_JOBS_TUNING,
    NUMERICAL_FEATURES_TO_STANDARDIZE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILENAME,
    RANDOM_STATE,
    RESAMPLING_METHOD,
    RESAMPLING_PARAMS,
    SCORING_TUNING,
    STATIC_MODELS,
)
from fraud_dynamic_ensemble.data_preparation.sampling import get_class_stats
from fraud_dynamic_ensemble.modeling.utils.models import get_base_model_and_search_space
from fraud_dynamic_ensemble.modeling.utils.pipeline import build_model_pipeline
from fraud_dynamic_ensemble.modeling.utils.training import train_and_evaluate_one_fold_static_model
from fraud_dynamic_ensemble.utils.io_utils import save_dict_json

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / PROCESSED_FILENAME,
    experiment_name: str = EXPERIMENT_NAME,
    experiment_description: str = EXPERIMENT_DESCRIPTION,
    model_path: Path = MODELS_DIR,
    target: str = "Class",
):
    logger.info("Running fraud_dynamic_ensemble/train.py ...")

    # --- Set experiment folder and experiment tracking
    experiment_id_name = f"{EXPERIMENT_RUN_ID}_{experiment_name}"
    experiment_path = model_path / experiment_id_name

    experiment_tracking = {
        "experiment_id_name": experiment_id_name,
        "experiment_description": experiment_description,
        "feature_transformation_standard_scaler": NUMERICAL_FEATURES_TO_STANDARDIZE,
        "feature_selection_KBest": FS_K_BEST_TO_KEEP,
        "resampling_method": RESAMPLING_METHOD,
        "resampling_params": RESAMPLING_PARAMS,
        "outer_evaluation_loop": f"RepeatedStratifiedKFold_{CV_OUTER_N_REPEATS}_times_{CV_OUTER_N_SPLITS}_folds",
        "DSEL_size": DSEL_SIZE,
        "inner_evaluation_hyperparameters_tuning_n_iter": N_ITER_TUNING,
        "inner_evaluation_hyperparameters_tuning_cv_splits": CV_INNER_N_SPLITS_TUNING,
        "inner_evaluation_hyperparameters_tuning_scoring": SCORING_TUNING,
        "inner_evaluation_hyperparameters_tuning_n_jobs": N_JOBS_TUNING,
        "models_to_train": STATIC_MODELS,
        "des_models_to_train": DES_MODELS,
    }

    experiment_path.mkdir(parents=True, exist_ok=True)
    save_dict_json(
        data=experiment_tracking, path=experiment_path / "experiment_config.json", mode="w"
    )

    logger.info(f"Initialized experiment: {experiment_id_name}")

    ################################# INITIAL CHECKS #################################
    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the feature engineering step first, e.g.: `python fraud_dynamic_ensemble/features.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading FEATURES dataset at path:\n\t{input_path}")
    df = pd.read_csv(input_path, header=0, sep=",")

    # Report BEFORE training.py phase
    counts, perc, rows, cols = get_class_stats(df, target)
    logger.info(f"Raw shape: rows={rows}, cols={cols}")
    logger.info("Class distribution (full dataset statistics):")
    for cls in counts.index:
        logger.info(f"  class={cls}: count={counts[cls]}, perc={perc[cls]:.6f}")

    ################################# PREPARE DATASET FOR TRAINING #################################
    # Initial shuffle of the data (frac=1.0 means that all rows will be kept but shuffled)
    logger.info(f"Shuffling dataset with random_state={RANDOM_STATE}")
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split data into features and labels
    X, y = df.drop(["Class"], axis=1), df["Class"]
    logger.info(
        f"Splitting dataset into features and labels. "
        f"Shape of X: {X.shape} - Shape of y: {y.shape}"
    )

    # Map column names to their corresponding indices
    features_name = X.columns.tolist()
    features_index = {name: idx for idx, name in enumerate(features_name)}

    # Check if the requested features to standardize are available in the dataset
    missing_feature_to_standardize = set(NUMERICAL_FEATURES_TO_STANDARDIZE) - features_index.keys()
    if missing_feature_to_standardize:
        raise ValueError(
            f"The following features requested for standardization were not "
            f"found in the features_index: {missing_feature_to_standardize}.\n"
            f"Available features: {features_name}"
        )

    # Convert the list of features name to indices for use in Feature Selection pipeline
    idx_num_features_to_standardize = [
        features_index[idx] for idx in NUMERICAL_FEATURES_TO_STANDARDIZE
    ]
    logger.info(
        f"Features to standardize: name={NUMERICAL_FEATURES_TO_STANDARDIZE} - idx_col={idx_num_features_to_standardize}"
    )

    # Convert dataset from pandas dataframe to a numpy array for compatibility  with deslib
    X, y = X.to_numpy(), y.to_numpy()
    logger.info(f"Dataset type after conversion: type(X)={type(X)}, type(y)={type(y)}")

    ################################# PREPARE SETTINGS FOR TRAINING #################################
    # Fix the evaluation strategy: RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    cv_outer = RepeatedStratifiedKFold(
        n_splits=CV_OUTER_N_SPLITS, n_repeats=CV_OUTER_N_REPEATS, random_state=RANDOM_STATE
    )

    # List to store all the resubstitution (train) and generalization (test) metrics
    # for each iteration of the RepeatedStratifiedKFold
    resubstitution_metrics_summary = []
    generalization_metrics_summary = []

    ################################# START TRAINING PHASE #################################
    logger.info("Starting training.py models...")

    for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        iteration_idx, fold_idx = divmod(run_id, CV_OUTER_N_SPLITS)
        logger.info(
            "-----------------------------------------------------------------------------"
        )
        logger.info(
            f"[ITERATION {iteration_idx + 1:2} - FOLD {fold_idx + 1:2} - RUN_ID {run_id:3}]"
        )

        # Split the data into training.py set (9 training.py folds) and test set (1 test fold)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Report class balance statistics for iteration
        for name, target in zip(["train dataset", "test dataset"], [y_train, y_test]):
            unique, frequency = np.unique(target, return_counts=True)
            logger.info(
                f"Class distribution ({name} statistics) [class, frequency]: {unique, frequency}"
            )

        # ----- Start training STATIC MODELS -----
        for static_model_name in STATIC_MODELS:
            logger.info(f"Training STATIC model: {static_model_name}")

            # Get the static model estimator and with its hyperparameter search space
            static_model_estimator, static_model_search_space = get_base_model_and_search_space(
                static_model_name, random_state=RANDOM_STATE
            )

            # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
            static_model_pipeline = build_model_pipeline(
                estimator=static_model_estimator,
                numerical_features_to_standardize=idx_num_features_to_standardize,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                resampling_method=RESAMPLING_METHOD,
                resampling_params=RESAMPLING_PARAMS,
            )

            # Tune the static model, fit on the training folds and evaluate on the test fold
            tuning_results, resubstitution_metrics, test_metrics = (
                train_and_evaluate_one_fold_static_model(
                    base_model=static_model_pipeline,
                    search_space=static_model_search_space,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_iter=N_ITER_TUNING,
                    val_cv_split=CV_INNER_N_SPLITS_TUNING,
                    scoring=SCORING_TUNING,
                    random_state=RANDOM_STATE,
                    n_jobs=-N_JOBS_TUNING,
                )
            )



            print(resubstitution_metrics_summary, generalization_metrics_summary)
            print(tuning_results, resubstitution_metrics, test_metrics)

            break
        break

    logger.success("Running fraud_dynamic_ensemble/train.py COMPLETED!")


if __name__ == "__main__":
    app()
