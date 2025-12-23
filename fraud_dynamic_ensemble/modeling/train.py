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
    FS_K_BEST_CANDIDATES,
    FS_K_BEST_TO_KEEP,
    MODELS_DIR,
    NUMERICAL_FEATURES_TO_STANDARDIZE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILENAME,
    RANDOM_STATE,
    RESAMPLING_METHOD,
    RESAMPLING_PARAMS,
    STATIC_ENSEMBLE_MODELS,
    STATIC_ENSEMBLE_POOLS,
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
    Run the full training workflow for the credit-card fraud project (STATIC, STATIC-ENSEMBLE, DES).

    This Typer command executes a repeated stratified outer CV loop and, for each outer split,
    trains and evaluates:

    - **STATIC** models (single estimators wrapped in an imblearn pipeline),
    - **STATIC ENSEMBLES** (e.g., Voting/Stacking ensembles wrapped in the same pipeline), and
    - **DES** models (DESlib competence models using a tuned pool and a DSEL split).

    High-level workflow
    -------------------
    1) Load the processed dataset (CSV) from ``input_path``.
    2) Shuffle rows once (global shuffle) using ``RANDOM_STATE``.
    3) Split into features/labels::

           X = df.drop([target], axis=1)
           y = df[target]

    4) Build standardization indices and post-preprocessing feature ordering using
       :func:`build_standardization_and_feature_order`.
    5) Convert ``X`` and ``y`` to NumPy arrays for DESlib compatibility.
    6) Build the outer evaluation loop via :class:`sklearn.model_selection.RepeatedStratifiedKFold`.
    7) Execute all outer folds (sequentially if ``outer_n_jobs == 1``; otherwise in parallel via joblib).
       For each outer fold, the fold orchestrator (:func:`train_and_evaluate_one_fold_all_models`) will:
       - tune and evaluate each STATIC model using the pipeline
         (scaling → SelectKBest → optional resampling → classifier),
       - tune and evaluate each STATIC ENSEMBLE model using the same pipeline structure,
       - tune the DES pool pipeline on the pool-training split, fit the DES competence model on DSEL,
         and evaluate the DES inference pipeline on the outer test fold,
       - compute **resubstitution metrics** for:
           * STATIC / STATIC-ENSEMBLE models on the full outer training set, and
           * DES models on the fitted **pool** (on the pool-training subset used internally by the
             DES helper),
       - compute **generalization metrics** on the outer test fold for all model families.
    8) Aggregate fold-level rows in memory, then persist results **per model** under the
       experiment folder.

    Parameters
    ----------
    input_path : pathlib.Path, default=PROCESSED_DATA_DIR / PROCESSED_FILENAME
        Path to the **processed** dataset (CSV) containing engineered features and the
        target column.
    experiment_name : str, default=EXPERIMENT_NAME
        Experiment folder name created under ``model_path``.
    experiment_description : str, default=EXPERIMENT_DESCRIPTION
        Free-text description stored in each model folder’s ``experiment_config.json``.
    model_path : pathlib.Path, default=MODELS_DIR
        Root directory where the experiment folder will be created and results saved.
    target : str, default="Class"
        Name of the target column in ``input_path``.
    outer_n_jobs : int, default=CV_OUTER_PARALLEL_N_JOBS
        Number of outer CV folds to execute in parallel.
        - ``1``  → sequential execution (default in many HPC-safe settings).
        - ``>1`` → parallelize outer folds with :class:`joblib.Parallel`.

        When using ``outer_n_jobs > 1``, it is usually recommended to set the inner-tuning
        parallelism (``TUNING_N_JOBS``) to ``1`` to avoid nested parallelism and CPU
        oversubscription.

    Feature Selection
    -----------------
    The training pipelines include :class:`sklearn.feature_selection.SelectKBest`:

    - ``FS_K_BEST_TO_KEEP`` sets the default ``k`` used to build the pipeline.
    - ``FS_K_BEST_CANDIDATES`` is forwarded to :func:`train_and_evaluate_one_fold_all_models`
      to optionally extend the tuning search space with ``feature_selection_filter__k``.

    Side Effects
    ------------
    - Creates/updates directories under::

          <model_path>/<experiment_name>/<MODEL_NAME>/

    - Writes (per model) the following files:
      - ``generalization_metrics_summary.csv``:
        metrics on the outer test fold for the model (STATIC / STATIC-ENSEMBLE / DES).
      - ``resubstitution_metrics_summary.csv``:
        - STATIC / STATIC-ENSEMBLE: metrics on the outer training set,
        - DES: metrics on the tuned/fitted **pool** used by DES (computed within the DES helper).
      - ``experiment_config.json``:
        the experiment tracking dictionary replicated in each model folder.

    - Produces logging via the configured :mod:`loguru` ``logger``.

    Raises
    ------
    typer.Exit
        If ``input_path`` does not exist.
    KeyError
        If ``target`` is not a column in the loaded dataset.
    ValueError
        If requested features to standardize are not present in the dataset, or if output
        DataFrames are missing mandatory columns (e.g., ``"model"``).
    RuntimeError
        If no models are found in the produced results (nothing to persist), or if a model
        has missing metrics rows (generalization or resubstitution), since both are expected
        for every trained model key.

    Notes
    -----
    Persistence layout
        Results are saved only inside per-model subfolders. The previously-used “global”
        CSVs at the experiment root are not produced.

    Parallel execution note
        When ``outer_n_jobs > 1``, joblib may use process-based parallelism (backend-dependent).
        Ensure objects passed to workers are picklable. If you encounter serialization issues
        due to the logger, consider adapting the fold worker signature to support ``logger=None``
        and configure per-process logging (or emit fold logs to stdout only).
    """

    logger.info("Running fraud_dynamic_ensemble/train.py ...")

    # --- Set experiment folder and experiment tracking
    experiment_path = model_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    experiment_tracking = {
        "experiment_description": experiment_description,
        "experiment_start_time": datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
        "use_cost_sensitive_learning": USE_COST_SENSITIVE_LEARNING,
        "feature_transformation_standard_scaler": NUMERICAL_FEATURES_TO_STANDARDIZE,
        "feature_selection_KBest_candidates": FS_K_BEST_CANDIDATES,
        "resampling_method": RESAMPLING_METHOD,
        "outer_evaluation_loop": f"RepeatedStratifiedKFold_{CV_OUTER_N_REPEATS}_times_{CV_OUTER_N_SPLITS}_folds",
        "DSEL_size": DSEL_SIZE,
        "tuning_hyperparameters_n_iter": TUNING_N_ITER,
        "tuning_hyperparameters_cv_splits": TUNING_CV_INNER_N_SPLITS,
        "tuning_hyperparameters_scoring": TUNING_SCORING,
        "tuning_hyperparameters_n_jobs": TUNING_N_JOBS,
        "static_models_to_train": STATIC_MODELS,
        "static_ensemble_models_to_train": STATIC_ENSEMBLE_MODELS,
        "static_ensemble_pools": STATIC_ENSEMBLE_POOLS,
        "des_models_to_train": DES_MODELS,
    }

    logger.info(f"Initialized experiment: {experiment_name}")
    logger.info(experiment_tracking)

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
    X, y = df.drop([target], axis=1), df[target]
    logger.info(
        f"Splitting dataset into features and labels. "
        f"Shape of X: {X.shape} - Shape of y: {y.shape}"
    )

    # Prepare standardization indices and post-preprocessing feature order
    idx_num_features_to_standardize, original_feature_names, transformed_feature_names = (
        build_standardization_and_feature_order(
            X=X,
            numerical_features_to_standardize=NUMERICAL_FEATURES_TO_STANDARDIZE,
        )
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
                static_ensemble_models=STATIC_ENSEMBLE_MODELS,
                static_ensemble_pools=STATIC_ENSEMBLE_POOLS,
                des_models=DES_MODELS,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                fs_k_best_candidates=FS_K_BEST_CANDIDATES,
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
                static_ensemble_models=STATIC_ENSEMBLE_MODELS,
                static_ensemble_pools=STATIC_ENSEMBLE_POOLS,
                des_models=DES_MODELS,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                fs_k_best_candidates=FS_K_BEST_CANDIDATES,
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
            for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y))
        )

        # ---- Aggregation (Post-Processing) ----
        # Parallel returns a list of tuples: [(res_rows, gen_rows), (res_rows, gen_rows), ...]
        # We must flatten this back into your summary lists.
        for resubstitution_rows, generalization_rows in parallel_results:
            resubstitution_metrics_summary.extend(resubstitution_rows)
            generalization_metrics_summary.extend(generalization_rows)

    experiment_tracking["experiment_end_time"] = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

    ############################### STORE EXPERIMENTAL RESULTS #####################################
    resubstitution_metrics_summary_df = pd.DataFrame(resubstitution_metrics_summary)
    generalization_metrics_summary_df = pd.DataFrame(generalization_metrics_summary)

    # Basic validation
    if (not resubstitution_metrics_summary_df.empty) and (
        "model" not in resubstitution_metrics_summary_df.columns
    ):
        raise ValueError("Missing 'model' column in resubstitution_metrics_summary.")
    if (not generalization_metrics_summary_df.empty) and (
        "model" not in generalization_metrics_summary_df.columns
    ):
        raise ValueError("Missing 'model' column in generalization_metrics_summary.")

    # Union of models across both summaries (covers DES-only runs too)
    models_in_results: set[str] = set()
    if not resubstitution_metrics_summary_df.empty:
        models_in_results |= set(resubstitution_metrics_summary_df["model"].astype(str).unique())
    if not generalization_metrics_summary_df.empty:
        models_in_results |= set(generalization_metrics_summary_df["model"].astype(str).unique())

    if not models_in_results:
        raise RuntimeError("No models found in results; nothing to persist.")

    for model_name in sorted(models_in_results):
        model_dir = experiment_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Filter rows for the model
        resubstitution_metrics_model = (
            resubstitution_metrics_summary_df[
                resubstitution_metrics_summary_df["model"].astype(str) == model_name
            ].copy()
            if not resubstitution_metrics_summary_df.empty
            else None
        )
        generalization_metrics_model = (
            generalization_metrics_summary_df[
                generalization_metrics_summary_df["model"].astype(str) == model_name
            ].copy()
            if not generalization_metrics_summary_df.empty
            else None
        )

        # Generalization expected for both STATIC and DES
        if generalization_metrics_model is None or generalization_metrics_model.empty:
            raise RuntimeError(f"No generalization rows found for model='{model_name}'.")
        generalization_metrics_model.to_csv(
            model_dir / "generalization_metrics_summary.csv", index=False, sep=","
        )

        # Resubstitution now expected for both STATIC and DES
        if resubstitution_metrics_model is None or resubstitution_metrics_model.empty:
            raise RuntimeError(f"No resubstitution rows found for model='{model_name}'.")
        resubstitution_metrics_model.to_csv(
            model_dir / "resubstitution_metrics_summary.csv", index=False, sep=","
        )

        # Store experiment settings
        save_dict_json(
            data=experiment_tracking, path=model_dir / "experiment_config.json", mode="w"
        )

    logger.success("Running fraud_dynamic_ensemble/train.py COMPLETED!")


if __name__ == "__main__":
    app()
