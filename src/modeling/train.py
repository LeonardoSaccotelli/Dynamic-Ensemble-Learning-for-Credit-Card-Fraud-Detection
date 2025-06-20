from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.config import RUN_ID, RANDOM_STATE
from src.config import NUMERICAL_FEATURES_TO_NORMALIZE, K_BEST_TO_KEEP
from src.config import CV_N_SPLITS, CV_N_REPEATS, DSEL_SIZE
from src.config import N_ITER_TUNING, CV_TUNING, SCORING_TUNING, N_JOBS_TUNING
from src.config import BASE_MODELS,STATIC_ENS_MODELS, DES_MODELS, POOL_MODELS
from src.utils.io_utils import load_csv, save_csv
from src.modeling.train_utils import train_and_evaluate_base_model, train_and_evaluate_ensemble_model
from src.modeling.models import get_base_model_and_search_space, get_static_ensemble_model
from src.modeling.pipeline import build_base_model_pipeline

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

    #TODO PRENDIAMO UN SUBSET PICCOLO PER TESTARE
    pos = df[df['Class'] == 0].sample(n=1000).reset_index(drop=True)
    neg = df[df['Class'] == 1]

    df = pd.concat([pos, neg])

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

        # Split the data into training (9 training folds) and test data (1 test fold)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Split the data into training (for base models) and DSEL (for DS techniques) data
        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=DSEL_SIZE,
                                                            random_state=RANDOM_STATE, stratify=y_train)

        logger.info(f"X_train_i, X_dsel_i, X_test_i shape: {X_train.shape}, {X_dsel.shape}, {X_test.shape}")
        for name, target in zip(["y_train", "y_dsel", "y_test"], [y_train, y_dsel, y_test]):
            stats = pd.DataFrame(
                {"count": target.value_counts(), "percentage": target.value_counts(normalize=True) * 100}).round(3)
            logger.info(f"Stratification class balance for {name}:\n{stats}")

        # Collect fitted base models
        fitted_base_models = {}

        ################################# TRAINING BASE MODELS #################################
        for base_model_name in BASE_MODELS:
            logger.info(f"Training base model: {base_model_name}")

            # Get the base model to tune with its hyperparameter search space
            base_model, search_space = get_base_model_and_search_space(base_model_name, RANDOM_STATE)

            # Build the pipeline: Preprocessing + Feature Selection + Classifier
            pipeline_base_model = build_base_model_pipeline(estimator=base_model,
                                                            numerical_columns_to_scale=NUMERICAL_FEATURES_TO_NORMALIZE,
                                                            k_best=K_BEST_TO_KEEP,
                                                            random_state=RANDOM_STATE)

            # Tune the base model, fit on the training folds and evaluate on the test fold
            (fitted_model,
             tuning_results,
             resubstitution_metrics,
             test_metrics) = train_and_evaluate_base_model(
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

            # Log resubstitution metrics with iteration and fold
            row_resubstitution_metrics = {
                "iteration": iteration_idx + 1,
                "fold": fold_idx + 1,
                "model": base_model_name,
                **resubstitution_metrics,
                **tuning_results,
            }
            resubstitution_metrics_summary.append(row_resubstitution_metrics)

            # Log test metrics with iteration and fold
            row_test_metrics = {
                "iteration": iteration_idx + 1,
                "fold": fold_idx + 1,
                "model": base_model_name,
                **test_metrics
            }
            test_metrics_summary.append(row_test_metrics)

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

            # Get the static ensemble model to train on the pool
            static_ensemble_model = get_static_ensemble_model(model_name=static_ensemble_model_name,
                                                              estimators=pool_classifiers_static_ens)
            fitted_static_ensemble_model, test_metrics = train_and_evaluate_ensemble_model(
                ensemble_model=static_ensemble_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )

            # Log test metrics with iteration and fold
            row_test_metrics = {
                "iteration": iteration_idx + 1,
                "fold": fold_idx + 1,
                "model": static_ensemble_model_name,
                **test_metrics
            }
            test_metrics_summary.append(row_test_metrics)


        ################################# TRAINING DES MODELS  #################################
        # Prepare pool_classifiers for DESlib (list of fitted estimators)
        pool_classifiers_des = [fitted_base_models[name] for name in POOL_MODELS]

        for des_model_name in DES_MODELS:
            logger.info(f"Training DES model: {des_model_name}")



        logger.success(f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}]")


    print(pd.DataFrame(test_metrics_summary))
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
