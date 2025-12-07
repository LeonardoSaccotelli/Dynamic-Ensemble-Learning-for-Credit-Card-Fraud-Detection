from pathlib import Path
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import typer

from fraud_dynamic_ensemble.config import (
    CV_OUTER_N_REPEATS,
    CV_OUTER_N_SPLITS,
    DES_MODELS,
    DSEL_SIZE,
    EXPERIMENT_DESCRIPTION,
    EXPERIMENT_NAME,
    EXPERIMENT_RUN_ID,
    FS_K_BEST_TO_KEEP,
    MODELS_DIR,
    NUMERICAL_FEATURES_TO_STANDARDIZE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILENAME,
    RANDOM_STATE,
    RESAMPLING_METHOD,
    RESAMPLING_PARAMS,
    STATIC_MODELS,
    TUNING_AGGRESSIVE_ELIMINATION,
    TUNING_CV_INNER_N_SPLITS,
    TUNING_FACTOR,
    TUNING_MAX_RESOURCES,
    TUNING_MIN_RESOURCES,
    TUNING_N_CANDIDATES,
    TUNING_N_JOBS,
    TUNING_SCORING,
)
from fraud_dynamic_ensemble.data_preparation.sampling import get_class_stats
from fraud_dynamic_ensemble.evaluation.metrics_evaluation import collect_report_one_fold
from fraud_dynamic_ensemble.modeling.utils.models import (
    get_des_model,
    get_static_model_and_search_space,
)
from fraud_dynamic_ensemble.modeling.utils.pipeline import (
    build_model_pipeline,
    build_standardization_and_feature_order,
    get_final_selected_features,
)
from fraud_dynamic_ensemble.modeling.utils.training import (
    train_and_evaluate_one_fold_des_model,
    train_and_evaluate_one_fold_static_model,
)
from fraud_dynamic_ensemble.utils.io_utils import save_dict_json

warnings.filterwarnings("ignore", category=FutureWarning)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / PROCESSED_FILENAME,
    experiment_name: str = EXPERIMENT_NAME,
    experiment_description: str = EXPERIMENT_DESCRIPTION,
    model_path: Path = MODELS_DIR,
    target: str = "Class",
):
    """
    Run the full training workflow (static models + DES) for the credit-card fraud project.

    This Typer command:
      1) loads the processed features CSV,
      2) shuffles and splits features/labels,
      3) iterates a RepeatedStratifiedKFold outer loop,
      4) for each outer fold:
         - tunes and evaluates each STATIC model (pipeline: scaling → feature selection → resampling → classifier),
         - tunes a bagging pool, fits a DES model on DSEL, and evaluates the DES pipeline on the test fold,
      5) logs and persists configuration and per-fold results as CSVs.

    Parameters
    ----------
    input_path : pathlib.Path, default=PROCESSED_DATA_DIR / PROCESSED_FILENAME
        Path to the **processed** dataset (CSV) containing engineered features and the target column.
    experiment_name : str, default=EXPERIMENT_NAME
        Human-readable experiment label used to build the run folder name.
    experiment_description : str, default=EXPERIMENT_DESCRIPTION
        Free-text description stored in the run’s `experiment_config.json`.
    model_path : pathlib.Path, default=MODELS_DIR
        Root directory where the experiment subfolder will be created and results saved.
    target : str, default="Class"
        Name of the target column in `input_path`.

    Side Effects
    ------------
    - Creates/updates files under `<MODELS_DIR>/<RUN_ID>_<experiment_name>/`.
    - Writes experiment configuration and two CSV summaries.
    - Produces extensive logging via the configured `logger`.

    Raises
    ------
    typer.Exit
        If `input_path` does not exist (precondition failure).
    ValueError
        If requested features to standardize are not present in the dataset.

    Notes
    -----
    Workflow
    - **Tracking & config dump**: creates `<MODELS_DIR>/<RUN_ID>_<experiment_name>/` and writes
      `experiment_config.json` with all key settings (CV, scaling, FS, resampling, model lists).
    - **Data loading**: reads `input_path` CSV; logs shape and global class distribution.
    - **Shuffle & split**: shuffles the dataset; separates `X` and `y` and converts to NumPy arrays.
    - **Preprocessing indexes**: validates requested features to standardize and maps them to indices.
    - **Outer CV**: `RepeatedStratifiedKFold(CV_OUTER_N_SPLITS × CV_OUTER_N_REPEATS)`.
      For each fold:
        * **STATIC models**:
          - build the full pipeline with `build_model_pipeline(...)`,
          - tune via `RandomizedSearchCV` using `get_static_model_and_search_space(...)`,
          - evaluate with `train_and_evaluate_one_fold_static_model(...)`,
          - extract selected features via `get_final_selected_features(...)`,
          - collect resubstitution and test metrics with `collect_report_one_fold(...)`.
        * **DES models**:
          - get pool + DES via `get_des_model(...)`,
          - build and tune the pool pipeline,
          - split TRAIN into pool-train/DSEL, fit DES with `train_and_evaluate_one_fold_des_model(...)`,
          - extract selected features and collect test metrics.
    - **Persistence**: writes:
        * `resubstitution_metrics_summary.csv`
        * `generalization_metrics_summary.csv`
    - Ensure the processed dataset exists (run `features.py` beforehand as indicated in logs).
    - `feature_names` passed to `get_final_selected_features` must match the **preprocessor output
      order**. When using `ColumnTransformer(remainder='passthrough')`, transformed columns
      typically precede passthrough columns.
    - Tuning can be computationally intensive; configure `N_JOBS_TUNING`, `N_ITER_TUNING`,
      and CV splits according to available resources.
    """

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
        "tuning_hyperparameters_n_candidates": TUNING_N_CANDIDATES,
        "tuning_hyperparameters_cv_splits": TUNING_CV_INNER_N_SPLITS,
        "tuning_hyperparameters_scoring": TUNING_SCORING,
        "tuning_hyperparameters_n_jobs": TUNING_N_JOBS,
        "tuning_hyperparameters_factor": TUNING_FACTOR,
        "tuning_hyperparameters_min_resources": TUNING_MIN_RESOURCES,
        "tuning_hyperparameters_max_resources": TUNING_MAX_RESOURCES,
        "tuning_hyperparameters_aggressive_elimination": TUNING_AGGRESSIVE_ELIMINATION,
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

    # Prepare standardization indices and post-preprocessing feature order
    (
        idx_num_features_to_standardize,
        features_name,
        transformed_feature_names,
    ) = build_standardization_and_feature_order(
        X=X,
        numerical_features_to_standardize=NUMERICAL_FEATURES_TO_STANDARDIZE,
    )

    logger.info(
        "Features to standardize: name=%s - idx_col=%s",
        NUMERICAL_FEATURES_TO_STANDARDIZE,
        idx_num_features_to_standardize,
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
        print("=" * 165)
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
            static_model_estimator, static_model_search_space = get_static_model_and_search_space(
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
            best_static_model, tuning_results, resubstitution_metrics, test_metrics = (
                train_and_evaluate_one_fold_static_model(
                    base_model=static_model_pipeline,
                    search_space=static_model_search_space,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_candidates=TUNING_N_CANDIDATES,
                    factor=TUNING_FACTOR,
                    min_resources=TUNING_MIN_RESOURCES,
                    max_resources=TUNING_MAX_RESOURCES,
                    aggressive_elimination=TUNING_AGGRESSIVE_ELIMINATION,
                    val_cv_split=TUNING_CV_INNER_N_SPLITS,
                    scoring=TUNING_SCORING,
                    random_state=RANDOM_STATE,
                    n_jobs=TUNING_N_JOBS,
                )
            )

            # Extract selected feature indices and names
            selected_indices, selected_names = get_final_selected_features(
                pipeline=best_static_model, feature_names=transformed_feature_names
            )

            # Collect resubstitution metrics and log
            collect_report_one_fold(
                resubstitution_metrics_summary,
                experiment_name=experiment_name,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=static_model_name,
                metrics=resubstitution_metrics,
                data_split="resubstitution",
                fold_size=len(X_train),
                **tuning_results,
                selected_features_indices=selected_indices,
                selected_features_names=selected_names,
            )

            # Collect generalization metrics and log
            collect_report_one_fold(
                generalization_metrics_summary,
                experiment_name=experiment_name,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=static_model_name,
                metrics=test_metrics,
                data_split="test",
                fold_size=len(X_test),
                selected_features_indices=selected_indices,
                selected_features_names=selected_names,
            )

        print("-" * 165)

        # ----- Start training DES MODELS -----
        for des_model_name in DES_MODELS:
            logger.info(f"Training DES model: {des_model_name}")

            # Get the des model estimator and its configuration, with the
            # pool of classifiers and its hyperparameter search space
            pool_classifiers, pool_search_space, des_model_estimator, des_model_conf = (
                get_des_model(des_model_name, random_state=RANDOM_STATE)
            )

            # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
            pool_classifiers_pipeline = build_model_pipeline(
                estimator=pool_classifiers,
                numerical_features_to_standardize=idx_num_features_to_standardize,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                resampling_method=RESAMPLING_METHOD,
                resampling_params=RESAMPLING_PARAMS,
            )

            best_des_model, test_metrics = train_and_evaluate_one_fold_des_model(
                des_model=des_model_estimator,
                des_conf=des_model_conf,
                pool_classifiers=pool_classifiers_pipeline,
                search_space=pool_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_candidates=TUNING_N_CANDIDATES,
                factor=TUNING_FACTOR,
                min_resources=TUNING_MIN_RESOURCES,
                max_resources=TUNING_MAX_RESOURCES,
                aggressive_elimination=TUNING_AGGRESSIVE_ELIMINATION,
                dsel_size=DSEL_SIZE,
                val_cv_split=TUNING_CV_INNER_N_SPLITS,
                scoring=TUNING_SCORING,
                random_state=RANDOM_STATE,
                n_jobs=TUNING_N_JOBS,
            )

            # Extract selected feature indices and names
            selected_indices, selected_names = get_final_selected_features(
                pipeline=best_des_model, feature_names=transformed_feature_names
            )

            # Collect generalization metrics and log
            collect_report_one_fold(
                generalization_metrics_summary,
                experiment_name=experiment_name,
                iteration=iteration_idx + 1,
                fold=fold_idx + 1,
                model=des_model_name,
                metrics=test_metrics,
                data_split="test",
                fold_size=len(X_test),
                selected_features_indices=selected_indices,
                selected_features_names=selected_names,
            )

        logger.success(
            f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]"
        )

    # Store experimental results
    resubstitution_metrics_summary = pd.DataFrame(resubstitution_metrics_summary)
    resubstitution_metrics_summary.to_csv(
        experiment_path / "resubstitution_metrics_summary.csv", index=False, sep=","
    )

    generalization_metrics_summary = pd.DataFrame(generalization_metrics_summary)
    generalization_metrics_summary.to_csv(
        experiment_path / "generalization_metrics_summary.csv", index=False, sep=","
    )

    logger.success("Running fraud_dynamic_ensemble/train.py COMPLETED!")


if __name__ == "__main__":
    app()
