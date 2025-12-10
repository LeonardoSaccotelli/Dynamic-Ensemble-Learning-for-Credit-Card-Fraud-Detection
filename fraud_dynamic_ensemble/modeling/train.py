from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import typer

from fraud_dynamic_ensemble.config import (
    CV_OUTER_N_REPEATS,
    CV_OUTER_N_SPLITS,
    CV_OUTER_PARALLEL_N_JOBS,
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
    TUNING_CV_INNER_N_SPLITS,
    TUNING_N_ITER,
    TUNING_N_JOBS,
    TUNING_SCORING,
    USE_COST_SENSITIVE_LEARNING,
)
from fraud_dynamic_ensemble.data_preparation.sampling import get_class_stats
from fraud_dynamic_ensemble.modeling.utils.pipeline import (
    build_standardization_and_feature_order,
)
from fraud_dynamic_ensemble.modeling.utils.training import (
    train_and_evaluate_one_fold_all_models,
)
from fraud_dynamic_ensemble.utils.io_utils import save_dict_json

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / PROCESSED_FILENAME,
    experiment_name: str = EXPERIMENT_NAME,
    experiment_description: str = EXPERIMENT_DESCRIPTION,
    model_path: Path = MODELS_DIR,
    target: str = "Class",
    outer_n_jobs: int = CV_OUTER_PARALLEL_N_JOBS,
):
    """
    Run the full training workflow (static models + DES) for the credit-card fraud project.

    This Typer command:
      1) loads the processed features CSV,
      2) shuffles and splits features/labels,
      3) builds a RepeatedStratifiedKFold outer loop,
      4) executes all outer folds (sequentially or in parallel, depending on
         ``outer_n_jobs``); for each outer fold it:
         - tunes and evaluates each STATIC model
           (pipeline: scaling → feature selection → resampling → classifier),
         - tunes a bagging pool, fits a DES model on DSEL, and evaluates the
           DES pipeline on the outer test fold,
      5) logs and persists configuration and per-fold results as CSVs.

    Parameters
    ----------
    input_path : pathlib.Path, default=PROCESSED_DATA_DIR / PROCESSED_FILENAME
        Path to the **processed** dataset (CSV) containing engineered features
        and the target column.
    experiment_name : str, default=EXPERIMENT_NAME
        Human-readable experiment label used to build the run folder name.
    experiment_description : str, default=EXPERIMENT_DESCRIPTION
        Free-text description stored in the run’s ``experiment_config.json``.
    model_path : pathlib.Path, default=MODELS_DIR
        Root directory where the experiment subfolder will be created and
        results saved.
    target : str, default="Class"
        Name of the target column in ``input_path``.
    outer_n_jobs : int, default=CV_OUTER_PARALLEL_N_JOBS
        Number of outer CV folds to execute in parallel.
        - ``1``  → sequential execution of outer folds (current behaviour).
        - ``>1`` → parallelize outer folds with :mod:`joblib.Parallel`.
          In this case it is usually recommended to set the inner tuning
          parameter ``TUNING_N_JOBS`` to ``1`` to avoid nested parallelism.

    Side Effects
    ------------
    - Creates/updates files under ``<MODELS_DIR>/<RUN_ID>_<experiment_name>/``.
    - Writes experiment configuration and two CSV summaries.
    - Produces extensive logging via the configured :mod:`loguru` ``logger``.

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist (precondition failure).
    ValueError
        If requested features to standardize are not present in the dataset.

    Notes
    -----
    Workflow
    - **Tracking & config dump**:
      creates ``<MODELS_DIR>/<RUN_ID>_<experiment_name>/`` and writes
      ``experiment_config.json`` with all key settings (outer CV, scaling,
      feature selection, resampling, tuning hyperparameters, model lists).
    - **Data loading**:
      reads ``input_path`` CSV; logs shape and global class distribution.
    - **Shuffle & split**:
      shuffles the dataset with ``RANDOM_STATE``; separates ``X`` and ``y``
      and converts them to NumPy arrays.
    - **Preprocessing indices & feature order**:
      uses :func:`build_standardization_and_feature_order` to
      - validate the requested features to standardize,
      - map them to column indices used by the scaler, and
      - obtain the post-preprocessing feature names, which are later passed
        to :func:`get_final_selected_features`.
    - **Outer CV**:
      ``RepeatedStratifiedKFold(n_splits=CV_OUTER_N_SPLITS,
      n_repeats=CV_OUTER_N_REPEATS, random_state=RANDOM_STATE)`` defines the
      outer evaluation loop. Outer folds can be run sequentially or in
      parallel depending on ``outer_n_jobs``.
    - **STATIC models per outer fold**:
      for each model in ``STATIC_MODELS`` it:
        * builds the pipeline with :func:`build_model_pipeline`,
        * obtains the estimator and search space via
          :func:`get_static_model_and_search_space`,
        * performs hyperparameter tuning with
          :class:`sklearn.model_selection.HalvingRandomSearchCV` inside
          :func:`train_and_evaluate_one_fold_static_model`,
        * computes resubstitution and test metrics via
          :func:`compute_classification_metrics`,
        * extracts selected features via
          :func:`get_final_selected_features`,
        * appends rows to the resubstitution and generalization summaries
          via :func:`collect_report_one_fold`.
    - **DES models per outer fold**:
      for each model in ``DES_MODELS`` it:
        * builds a bagging pool and DES configuration via
          :func:`get_des_model`,
        * builds the pool pipeline with :func:`build_model_pipeline`,
        * tunes the pool with :class:`HalvingRandomSearchCV` and
          splits TRAIN into pool-training and DSEL inside
          :func:`train_and_evaluate_one_fold_des_model`,
        * fits the DES model on DSEL (transformed space),
        * evaluates the final DES pipeline on the outer test fold and logs
          the metrics via :func:`collect_report_one_fold`.
    - **Persistence**:
      writes two CSV files under the experiment folder:
        * ``resubstitution_metrics_summary.csv``
        * ``generalization_metrics_summary.csv``.
      These contain one row per model, per outer fold, per data split
      (resubstitution/test), including confusion-matrix counts, metrics,
      tuning summaries and selected feature information.
    - Tuning can be computationally intensive; configure
      ``TUNING_N_CANDIDATES``, ``TUNING_FACTOR``, ``TUNING_MIN_RESOURCES``,
      ``TUNING_MAX_RESOURCES``, ``TUNING_AGGRESSIVE_ELIMINATION``,
      ``TUNING_N_JOBS`` and ``TUNING_CV_INNER_N_SPLITS`` according to the
      available computational resources and desired search budget.
    """

    logger.info("Running fraud_dynamic_ensemble/train.py ...")

    # --- Set experiment folder and experiment tracking
    experiment_id_name = f"{EXPERIMENT_RUN_ID}_{experiment_name}"
    experiment_path = model_path / experiment_id_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    experiment_tracking = {
        "experiment_id_name": experiment_id_name,
        "experiment_description": experiment_description,
        "experiment_start_time": datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
        "use_cost_sensitive_learning": USE_COST_SENSITIVE_LEARNING,
        "feature_transformation_standard_scaler": NUMERICAL_FEATURES_TO_STANDARDIZE,
        "feature_selection_KBest": FS_K_BEST_TO_KEEP,
        "resampling_method": RESAMPLING_METHOD,
        "resampling_params": RESAMPLING_PARAMS,
        "outer_evaluation_loop": f"RepeatedStratifiedKFold_{CV_OUTER_N_REPEATS}_times_{CV_OUTER_N_SPLITS}_folds",
        "DSEL_size": DSEL_SIZE,
        "tuning_hyperparameters_n_iter": TUNING_N_ITER,
        "tuning_hyperparameters_cv_splits": TUNING_CV_INNER_N_SPLITS,
        "tuning_hyperparameters_scoring": TUNING_SCORING,
        "tuning_hyperparameters_n_jobs": TUNING_N_JOBS,
        "models_to_train": STATIC_MODELS,
        "des_models_to_train": DES_MODELS,
    }

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

    ################################# PREPARE SETTINGS FOR TRAINING ##################################
    # Fix the evaluation strategy: RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    cv_outer = RepeatedStratifiedKFold(
        n_splits=CV_OUTER_N_SPLITS, n_repeats=CV_OUTER_N_REPEATS, random_state=RANDOM_STATE
    )

    # List to store all the resubstitution (train) and generalization (test) metrics
    # for each iteration of the RepeatedStratifiedKFold
    resubstitution_metrics_summary = []
    generalization_metrics_summary = []

    ################################# START TRAINING PHASE ###########################################
    logger.info("Starting model training over all outer folds...")

    if outer_n_jobs == 1:
        logger.info("Training all outer folds sequentially")

        # ---- Sequential execution (current behaviour) ----
        for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
            iteration_idx, fold_idx = divmod(run_id, CV_OUTER_N_SPLITS)

            print("=" * 165)

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
                use_cost_sensitive_learning=USE_COST_SENSITIVE_LEARNING,
                resampling_method=RESAMPLING_METHOD,
                resampling_params=RESAMPLING_PARAMS,
                tuning_n_iter=TUNING_N_ITER,
                tuning_cv_inner_n_splits=TUNING_CV_INNER_N_SPLITS,
                tuning_scoring=TUNING_SCORING,
                tuning_n_jobs=TUNING_N_JOBS,
                dsel_size=DSEL_SIZE,
                random_state=RANDOM_STATE,
                logger=logger,
            )

            resubstitution_metrics_summary.extend(resubstitution_rows)
            generalization_metrics_summary.extend(generalization_rows)

    else:
        # ---- Parallel execution of outer folds ----
        logger.info(f"Parallelizing outer folds with outer_n_jobs={outer_n_jobs}")

        # Run the 10x10 CV in parallel
        parallel_results = Parallel(n_jobs=outer_n_jobs, verbose=10)(
            delayed(train_and_evaluate_one_fold_all_models)(
                run_id=run_id,
                iteration_idx=divmod(run_id, CV_OUTER_N_SPLITS)[0],
                fold_idx=divmod(run_id, CV_OUTER_N_SPLITS)[1],
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
                use_cost_sensitive_learning=USE_COST_SENSITIVE_LEARNING,
                resampling_method=RESAMPLING_METHOD,
                resampling_params=RESAMPLING_PARAMS,
                tuning_n_candidates=TUNING_N_ITER,
                tuning_cv_inner_n_splits=TUNING_CV_INNER_N_SPLITS,
                tuning_scoring=TUNING_SCORING,
                tuning_n_jobs=TUNING_N_JOBS,
                dsel_size=DSEL_SIZE,
                random_state=RANDOM_STATE,
                logger=logger,
            )
            for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y))
        )

        # ---- Aggregation (Post-Processing) ----
        # Parallel returns a list of tuples: [(res_rows, gen_rows), (res_rows, gen_rows), ...]
        # We must flatten this back into your summary lists.
        for resubstitution_rows, generalization_rows in parallel_results:
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

    experiment_tracking["experiment_end_time"] = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
    save_dict_json(
        data=experiment_tracking, path=experiment_path / "experiment_config.json", mode="w"
    )

    logger.success("Running fraud_dynamic_ensemble/train.py COMPLETED!")


if __name__ == "__main__":
    app()
