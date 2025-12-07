from pathlib import Path
import warnings

from loguru import logger
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
from fraud_dynamic_ensemble.modeling.utils.pipeline import (
    build_standardization_and_feature_order,
)
from fraud_dynamic_ensemble.modeling.utils.training import (
    train_and_evaluate_one_fold_all_models,
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

    ################################# START TRAINING PHASE #########################################
    logger.info("Starting training.py models...")

    for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        iteration_idx, fold_idx = divmod(run_id, CV_OUTER_N_SPLITS)
        print("=" * 165)
        logger.info(
            f"[ITERATION {iteration_idx + 1:2} - FOLD {fold_idx + 1:2} - RUN_ID {run_id:3}]"
        )

        resubstitution_rows, generalization_rows = train_and_evaluate_one_fold_all_models(
            run_id=run_id,
            iteration_idx=iteration_idx,
            fold_idx=fold_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            X=X,
            y=y,
            experiment_name=experiment_name,
            idx_num_features_to_standardize=idx_num_features_to_standardize,
            transformed_feature_names=transformed_feature_names,
            static_models=STATIC_MODELS,
            des_models=DES_MODELS,
            fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
            resampling_method=RESAMPLING_METHOD,
            resampling_params=RESAMPLING_PARAMS,
            tuning_n_candidates=TUNING_N_CANDIDATES,
            tuning_factor=TUNING_FACTOR,
            tuning_min_resources=TUNING_MIN_RESOURCES,
            tuning_max_resources=TUNING_MAX_RESOURCES,
            tuning_aggressive_elimination=TUNING_AGGRESSIVE_ELIMINATION,
            tuning_cv_inner_n_splits=TUNING_CV_INNER_N_SPLITS,
            tuning_scoring=TUNING_SCORING,
            tuning_n_jobs=TUNING_N_JOBS,
            dsel_size=DSEL_SIZE,
            random_state=RANDOM_STATE,
        )

        resubstitution_metrics_summary.extend(resubstitution_rows)
        generalization_metrics_summary.extend(generalization_rows)

    ############################### STORE EXPERIMENTAL RESULTS #####################################
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
