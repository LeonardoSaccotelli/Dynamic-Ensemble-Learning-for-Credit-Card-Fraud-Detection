from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from fraud_dynamic_ensemble.evaluation.metrics_evaluation import (
    collect_report_one_fold,
    compute_classification_metrics,
)
from fraud_dynamic_ensemble.modeling.utils.models import (
    get_des_model,
    get_static_model_and_search_space,
)
from fraud_dynamic_ensemble.modeling.utils.pipeline import (
    build_model_pipeline,
    get_final_selected_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_evaluate_one_fold_static_model(
    base_model: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_iter: int,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Union[ImbPipeline, Pipeline, BaseEstimator],
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float],
]:
    """
    Tune a model (or pipeline) with ``RandomizedSearchCV`` on the training set,
    refit the best configuration, and report metrics on both train and test.

    The procedure uses a stratified CV splitter during the hyperparameter search,
    then evaluates the selected estimator on:
      1) the **training** set (resubstitution error), and
      2) the **held-out test** set (generalization).

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        Estimator/pipeline to optimize. Must implement ``fit`` and ``predict``.
        This implementation also calls ``predict_proba`` to compute probability-based
        metrics and will fail if it is not available.
    search_space : dict
        Parameter distributions for ``RandomizedSearchCV``. Keys must match the
        estimator (or pipeline step) parameter names, using scikit-learn’s
        double-underscore convention, e.g.:
        - ``"classifier__C"``, ``"classifier__max_depth"``
        - ``"feature_selection_filter__k"`` (to tune SelectKBest ``k``)
        - ``"preprocessor__..."`` (if your preprocessor exposes tunable params)
    X_train, X_test : array-like of shape (n_samples, n_features)
        Training and test features.
    y_train, y_test : array-like of shape (n_samples,)
        Training and test labels.
    n_iter : int
        Number of parameter settings sampled by ``RandomizedSearchCV``.
    val_cv_split : int, default=5
        Number of stratified folds used during the hyperparameter search (inner CV).
    scoring : str, default="f1"
        Optimization metric passed to ``RandomizedSearchCV`` (e.g., ``"f1"``,
        ``"average_precision"``, ``"roc_auc"``).
    random_state : int, default=42
        Random seed for both the ``StratifiedKFold`` splitter (with shuffling) and
        the ``RandomizedSearchCV`` search process.
    n_jobs : int, default=-1
        Number of parallel jobs for the randomized search (``-1`` uses all available cores).
        The underlying estimators should typically use ``n_jobs=1`` to avoid nested parallelism.

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        The refit estimator corresponding to the best hyperparameter configuration.
    tuning_results : dict
        Summary of the tuning at the best index, including:
        - ``cv_tuning_mean_train_score`` : float
        - ``cv_tuning_std_train_score`` : float
        - ``cv_tuning_mean_val_score`` : float
        - ``cv_tuning_std_val_score`` : float
        - ``best_params`` : dict
        - ``tuning_time`` : float (seconds)
    resubstitution_metrics : dict[str, float]
        Metrics on the training set computed via ``compute_classification_metrics``.
    test_metrics : dict[str, float]
        Metrics on the test set computed via ``compute_classification_metrics``,
        plus ``"score_time"`` (seconds to generate test predictions).

    Raises
    ------
    AttributeError
        If ``best_model`` does not implement ``predict_proba`` (this function uses
        ``predict_proba(X)[:, 1]``).
    ValueError
        If ``RandomizedSearchCV`` fails due to invalid ``search_space`` keys or an
        incompatible ``scoring`` metric.

    Notes
    -----
    - **Binary classification assumption:** probabilities are extracted as
      ``predict_proba(... )[:, 1]`` (positive class). If you generalize beyond
      binary classification, this must be adapted.
    - The inner CV splitter is
      ``StratifiedKFold(n_splits=val_cv_split, shuffle=True, random_state=random_state)``.
    - ``return_train_score=True`` is enabled so train CV scores are available in
      ``search.cv_results_`` for reporting.

    Examples
    --------
    Optimize a pipeline (including tuning SelectKBest ``k``) and evaluate:

    >>> search_space = {
    ...     "feature_selection_filter__k": [10, 20, 30, "all"],
    ...     "classifier__C": loguniform(1e-3, 1e3),
    ... }
    >>> best_model, tuning, resub_metrics, test_metrics = (
    ...     train_and_evaluate_one_fold_static_model(
    ...         base_model=base_model,
    ...         search_space=search_space,
    ...         X_train=X_train, y_train=y_train,
    ...         X_test=X_test,   y_test=y_test,
    ...         n_iter=30,
    ...         val_cv_split=5,
    ...         scoring="average_precision",
    ...         random_state=42,
    ...         n_jobs=-1,
    ...     )
    ... )
    """
    splitter = StratifiedKFold(
        n_splits=val_cv_split,
        random_state=random_state,
        shuffle=True,
    )

    print(
        f"[RANDOMIZED SEARCH SETTINGS]: scoring: {scoring}, random_state: {random_state}, n_jobs: {n_jobs}"
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        cv=splitter,
        verbose=3,
        random_state=random_state,
        return_train_score=True,
    )

    # Fit the search and measure the tuning time
    start_tuning_time = time.time()
    search.fit(X_train, y_train)
    end_tuning_time = time.time()

    # Get the best estimator
    best_model = search.best_estimator_

    # Retrieve best search info
    tuning_results = {
        "cv_tuning_mean_train_score": search.cv_results_["mean_train_score"][search.best_index_],
        "cv_tuning_std_train_score": search.cv_results_["std_train_score"][search.best_index_],
        "cv_tuning_mean_val_score": search.cv_results_["mean_test_score"][search.best_index_],
        "cv_tuning_std_val_score": search.cv_results_["std_test_score"][search.best_index_],
        "best_params": search.best_params_,
        "tuning_time": end_tuning_time - start_tuning_time,
    }
    print(f"[RANDOMIZED SEARCH BEST PARAMS]: {tuning_results['best_params']}")

    # Evaluate on the training set (resubstitution error)
    print("[COMPUTING RESUBSTITUTION METRICS]...")
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    resubstitution_metrics = compute_classification_metrics(
        y_train, y_train_pred, y_train_pred_prob
    )
    print(f"[RESUBSTITUTION METRICS]: {resubstitution_metrics}")

    # Evaluate on the test set (generalization error)
    print("[COMPUTING GENERALIZATION METRICS]...")
    start_score_time = time.time()
    y_test_pred = best_model.predict(X_test)
    end_score_time = time.time()
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_prob)
    test_metrics["score_time"] = end_score_time - start_score_time
    print(f"[GENERALIZATION METRICS]: {test_metrics}")

    return best_model, tuning_results, resubstitution_metrics, test_metrics


def train_and_evaluate_one_fold_des_model(
    des_model: BaseEstimator,
    des_conf: Dict[str, Any],
    pool_classifiers: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_iter: int,
    dsel_size: float = 0.2,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Pipeline,
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float | int],
]:
    """
    Train a **Dynamic Ensemble Selection (DES)** model on one outer fold and
    evaluate it on the held-out test set, while also reporting tuning and
    resubstitution metrics for the **pool**.

    Workflow
    --------
    1) Split the provided outer training set into:
       - **pool-training** subset: used to tune/fit the base pool pipeline
         (``pool_classifiers``) with ``RandomizedSearchCV``.
       - **DSEL** subset: used to fit the DES competence model.
    2) Tune and refit ``pool_classifiers`` on the pool-training subset.
    3) Compute **pool resubstitution metrics** on the pool-training subset
       using the tuned pool pipeline (analogous to the STATIC reference function).
    4) From the best pool pipeline:
       - Extract the **fitted preprocessing** steps (all steps before ``"resampling"``).
       - Extract the tuned **classifier** step to be used as the fitted **pool**.
    5) Transform DSEL with the fitted preprocessing and fit the DES model on
       ``(X_dsel_trans, y_dsel)`` after injecting the tuned pool via
       ``des_conf["pool_classifiers"] = fitted_pool``.
    6) Build the **final inference pipeline** = ``preprocessing -> DES`` (no resampling).
    7) Evaluate the final DES pipeline on ``(X_test, y_test)``.

    Parameters
    ----------
    des_model : sklearn.base.BaseEstimator
        Unfitted DESlib estimator instance (e.g., ``KNORAE``, ``OLA``, ``METADES``).
        Must accept ``pool_classifiers=...`` in ``set_params`` and implement
        ``fit`` / ``predict`` (``predict_proba`` is optional).
    des_conf : dict[str, Any]
        Hyperparameters for ``des_model`` (e.g., ``k``, ``DFP``, ``IH_rate``, ``voting``,
        ``n_jobs``). Do not include ``pool_classifiers`` initially; it is injected internally.
        This dict is **mutated in-place** by adding ``"pool_classifiers"`` before fitting.
    pool_classifiers : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline
        Pool-training pipeline to be tuned. Expected to contain the steps:
        - ``"preprocessor"``
        - ``"feature_selection_filter"`` (e.g., ``SelectKBest``)
        - ``"resampling"`` (train-only; excluded from final inference)
        - ``"classifier"`` (the bagging/ensemble object used as the fitted pool)
    search_space : dict[str, Any]
        Hyperparameter distributions for tuning the pool (e.g.,
        ``"classifier__n_estimators"``, ``"feature_selection_filter__k"``).
    X_train, y_train : array-like
        Features/labels of the **outer training** fold (split internally into pool-training and DSEL).
    X_test, y_test : array-like
        Features/labels of the **outer test** fold.
    n_iter : int
        Number of parameter settings sampled by ``RandomizedSearchCV`` for tuning the pool.
    dsel_size : float, default=0.2
        Proportion of ``X_train`` reserved for DSEL (``0 < dsel_size < 1``).
    val_cv_split : int, default=5
        Number of stratified folds for the inner hyperparameter search.
    scoring : str, default="f1"
        Scoring metric passed to ``RandomizedSearchCV`` (e.g., ``"f1"``, ``"roc_auc"``,
        ``"average_precision"``).
    random_state : int, default=42
        Random seed used in the DSEL split, inner CV splitter, and randomized search.
    n_jobs : int, default=-1
        Parallel jobs for ``RandomizedSearchCV`` (``-1`` uses all cores). The underlying
        estimators should typically use ``n_jobs=1`` to avoid nested parallelism.

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline (resampling excluded):
        ``preprocessing_steps + [('classifier', DES)]``.
    tuning_results : dict
        Summary of the pool tuning at the best index, including:
        - ``cv_tuning_mean_train_score`` : float
        - ``cv_tuning_std_train_score`` : float
        - ``cv_tuning_mean_val_score`` : float
        - ``cv_tuning_std_val_score`` : float
        - ``best_params`` : dict
        - ``tuning_time`` : float (seconds)
    pool_resubstitution_metrics : dict[str, float]
        Metrics computed on the **pool-training subset** using the tuned pool pipeline
        (resubstitution error of the pool only).
    test_metrics : dict[str, float | int]
        Metrics computed on the test set by ``compute_classification_metrics``,
        plus ``"score_time"`` (seconds).

    Raises
    ------
    KeyError
        If required pipeline steps are missing from ``pool_classifiers.named_steps``
        (notably ``"resampling"`` or ``"classifier"``).
    AttributeError
        If the tuned pool pipeline does not implement ``predict_proba`` (pool resubstitution
        requires probabilities, consistent with the STATIC reference function).
    ValueError
        If ``RandomizedSearchCV`` fails due to invalid ``search_space`` keys or an
        incompatible ``scoring`` metric.

    Notes
    -----
    - **No leakage:** the outer test set is never used for pool tuning nor DES fitting.
    - **Train-time only resampling:** pool ``"resampling"`` is used only during pool tuning/fit
      and excluded from the final inference pipeline.
    - **Probabilities (test):** ``predict_proba`` is attempted for test probabilities; if unavailable,
      probabilities are set to ``None``.
    - **Binary classification convention:** when ``predict_proba`` is available, the positive class is
      extracted as ``predict_proba(X)[:, 1]``.

    Examples
    --------
    Tune a bagging pool, compute pool resubstitution metrics, fit DES on DSEL, and evaluate on test:

    >>> final_pipe, tuning, pool_resub, test_metrics = train_and_evaluate_one_fold_des_model(
    ...     des_model=des_model,
    ...     des_conf=des_conf,
    ...     pool_classifiers=pool_pipeline,
    ...     search_space=pool_space,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,   y_test=y_test,
    ...     n_iter=30,
    ...     dsel_size=0.2,
    ...     val_cv_split=5,
    ...     scoring="average_precision",
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )
    """
    # Split TRAIN into pool-training and DSEL
    X_train_pool, X_dsel, y_train_pool, y_dsel = train_test_split(
        X_train,
        y_train,
        test_size=dsel_size,
        stratify=y_train,
        random_state=random_state,
    )

    # Inner CV for pool tuning
    splitter = StratifiedKFold(
        n_splits=val_cv_split,
        shuffle=True,
        random_state=random_state,
    )

    print(
        f"[RANDOMIZED SEARCH SETTINGS]: scoring: {scoring}, random_state: {random_state}, n_jobs: {n_jobs}"
    )

    search = RandomizedSearchCV(
        estimator=pool_classifiers,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        cv=splitter,
        verbose=3,
        random_state=random_state,
        return_train_score=True,
    )

    # Fit the search and measure the tuning time
    start_tuning_time = time.time()
    search.fit(X_train_pool, y_train_pool)
    end_tuning_time = time.time()

    best_pipe_pool_classifiers = search.best_estimator_
    print(f"[RANDOMIZED SEARCH BEST PARAMS]: {search.best_params_}")

    # Retrieve best search info (same structure as STATIC reference function)
    tuning_results = {
        "cv_tuning_mean_train_score": search.cv_results_["mean_train_score"][search.best_index_],
        "cv_tuning_std_train_score": search.cv_results_["std_train_score"][search.best_index_],
        "cv_tuning_mean_val_score": search.cv_results_["mean_test_score"][search.best_index_],
        "cv_tuning_std_val_score": search.cv_results_["std_test_score"][search.best_index_],
        "best_params": search.best_params_,
        "tuning_time": end_tuning_time - start_tuning_time,
    }

    # --- Pool resubstitution metrics (only for the tuned pool pipeline) ---
    print("[COMPUTING POOL RESUBSTITUTION METRICS]...")
    y_train_pool_pred = best_pipe_pool_classifiers.predict(X_train_pool)

    # Consistent with the STATIC reference function: require predict_proba
    y_train_pool_pred_prob = best_pipe_pool_classifiers.predict_proba(X_train_pool)[:, 1]

    pool_resubstitution_metrics = compute_classification_metrics(
        y_train_pool, y_train_pool_pred, y_train_pool_pred_prob
    )
    print(f"[POOL RESUBSTITUTION METRICS]: {pool_resubstitution_metrics}")

    # Extract the preprocessing pipeline (fitted)
    # We skip resampling step since it is required just at training time
    resampling_idx = list(best_pipe_pool_classifiers.named_steps.keys()).index("resampling")
    fitted_preproc = best_pipe_pool_classifiers[:resampling_idx]

    # Apply the preprocessing steps on the DSEL dataset
    X_dsel_trans = fitted_preproc.transform(X_dsel)

    # Extract the fitted pool of classifiers
    fitted_pool = best_pipe_pool_classifiers.named_steps["classifier"]

    # Add the trained pool of classifiers to the DES model config
    des_conf["pool_classifiers"] = fitted_pool

    # Fit DES model on DSEL in transformed space
    print("[FITTING DSEL METHOD]...")
    des_model.set_params(**des_conf)
    des_model.fit(X_dsel_trans, y_dsel)

    # Final inference pipeline: preprocessing -> DES
    final_des_pipeline = Pipeline(fitted_preproc.steps + [("classifier", des_model)])

    # Evaluate on the test set (generalization error)
    print("[COMPUTING GENERALIZATION METRICS]...")
    start_score_time = time.time()
    y_test_pred = final_des_pipeline.predict(X_test)
    try:
        y_test_pred_proba = final_des_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # Some DES models / configurations may not implement predict_proba
        y_test_pred_proba = None
    end_score_time = time.time()

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_proba)
    test_metrics["score_time"] = end_score_time - start_score_time
    print(f"[GENERALIZATION METRICS]: {test_metrics}")

    return final_des_pipeline, tuning_results, pool_resubstitution_metrics, test_metrics


def train_and_evaluate_one_fold_all_models(
    run_id: int,
    iteration_idx: int,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str,
    idx_num_features_to_standardize: Sequence[int],
    transformed_feature_names: Sequence[str],
    static_models: Sequence[str],
    des_models: Sequence[str],
    fs_k_best_to_keep: int | str,
    fs_k_best_candidates: Sequence[int | str] | None,
    use_cost_sensitive_learning: bool,
    resampling_method: str | None,
    resampling_params: Dict[str, Any] | None,
    tuning_n_iter: int,
    tuning_cv_inner_n_splits: int,
    tuning_scoring: str,
    tuning_n_jobs: int,
    dsel_size: float,
    random_state: int,
    logger: Any | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train and evaluate all STATIC and DES models on a single outer CV fold.

    This helper encapsulates the full workflow for one outer split of a
    ``RepeatedStratifiedKFold`` experiment. Given the global feature/label arrays
    and the current train/test indices, it:

    1. Splits ``X`` and ``y`` into an outer training and test set.
    2. For each **STATIC** model in ``static_models``:
       - retrieves the estimator and its **estimator-level** hyperparameter search space
         via :func:`get_static_model_and_search_space`,
       - optionally **extends** the search space with the pipeline-level parameter
         ``"feature_selection_filter__k"`` using ``fs_k_best_candidates``,
       - builds the training pipeline (preprocessing → SelectKBest → resampling → classifier),
       - tunes and refits it via :func:`train_and_evaluate_one_fold_static_model`,
       - extracts final selected features via :func:`get_final_selected_features`,
       - collects both resubstitution (train) and generalization (test) metrics.
    3. For each **DES** model in ``des_models``:
       - retrieves a bagging pool estimator and its **estimator-level** search space plus an
         unfitted DESlib model via :func:`get_des_model`,
       - optionally **extends** the pool search space with ``"feature_selection_filter__k"``
         using ``fs_k_best_candidates``,
       - builds the pool pipeline (preprocessing → SelectKBest → resampling → bagging pool),
       - tunes the pool, fits the DES competence model on DSEL, and evaluates the final
         preprocessing→DES pipeline via :func:`train_and_evaluate_one_fold_des_model`,
       - extracts final selected features via :func:`get_final_selected_features`,
       - collects:
           * **resubstitution metrics** for the tuned/fitted **pool** (training-set error),
           * **generalization metrics** for the DES inference pipeline on the outer test set.
    4. Returns two lists of rows (dicts) ready to be aggregated and saved at experiment level.

    Parameters
    ----------
    run_id : int
        Global counter from ``enumerate(cv_outer.split(X, y))`` identifying the
        current outer split (across repetitions and folds).
    iteration_idx : int
        0-based index of the outer repetition within ``RepeatedStratifiedKFold``.
    fold_idx : int
        0-based index of the fold within the current repetition.
    train_idx : numpy.ndarray
        Indices of samples used as training data for this outer fold.
    test_idx : numpy.ndarray
        Indices of samples used as test data for this outer fold.
    X : numpy.ndarray of shape (n_samples, n_features)
        Full feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Full target vector.
    experiment_name : str
        Experiment identifier stored in every metrics row.
    idx_num_features_to_standardize : sequence of int
        Column indices to be standardized by the preprocessing step.
    transformed_feature_names : sequence of str
        Feature names aligned with the input of ``"feature_selection_filter"``
        (i.e., after ``"preprocessor"``). Used to map selected indices back to
        human-readable names.
    static_models : sequence of str
        List of static model names to be trained
        (e.g. ``["SVC", "RandomForestClassifier"]``).
    des_models : sequence of str
        List of DES model names to be trained
        (e.g. ``["KNORAU", "METADES"]``).
    fs_k_best_to_keep : int or {"all"}
        Default ``k`` used to build the ``SelectKBest(k=...)`` step inside
        :func:`build_model_pipeline`.

        Important: this is the *initial* value used when constructing the pipeline.
        If the tuning search space includes ``"feature_selection_filter__k"``,
        the tuned value may differ at fit time.
    fs_k_best_candidates : sequence of (int or {"all"}) or None
        Optional candidate values for ``SelectKBest.k`` to be explored during tuning.

        If provided, this function injects the candidates directly into the
        hyperparameter search spaces used for tuning by adding::

            "feature_selection_filter__k": list(fs_k_best_candidates)

        to both STATIC pipelines and DES pool pipelines. If ``None``, feature selection
        is treated as fixed at ``fs_k_best_to_keep`` (unless the search space is modified
        upstream).
    use_cost_sensitive_learning : bool
        Whether to enable cost-sensitive / imbalance-aware behaviour in both
        STATIC models and DES pools. Forwarded to the model factories used here.
    resampling_method : str or None
        Canonical name of the resampling strategy (e.g. ``"SMOTE"``,
        ``"RandomUnderSampler"``, ``"SMOTEENN"``) or ``None``/``"none"``
        to disable resampling.
    resampling_params : dict or None
        Extra keyword arguments for the sampler constructor.
    tuning_n_iter : int
        Value passed as ``n_iter`` to the CV-based hyperparameter search used
        for both STATIC models and DES bagging pools.
    tuning_cv_inner_n_splits : int
        Number of stratified folds for the inner model-selection CV.
    tuning_scoring : str
        Scoring metric passed to the hyperparameter search.
    tuning_n_jobs : int
        Number of parallel jobs for the hyperparameter search.
    dsel_size : float
        Fraction of the outer training set reserved for the DSEL subset used
        to fit DES models (``0 < dsel_size < 1``).
    random_state : int
        Base random seed forwarded to models, inner CV splitters, and the DSEL split.
        ``random_state + run_id`` is used for tuning calls to diversify the randomized
        search across outer folds.
    logger : Any or None, optional
        Optional logger exposing ``.info(str)`` (e.g., Loguru). If ``None``, messages
        are printed to stdout.

    Returns
    -------
    resubstitution_rows : list of dict
        Metrics rows on the **training** side of the current outer fold:
        - STATIC models: resubstitution metrics computed on ``(X_train, y_train)``.
        - DES models: resubstitution metrics computed for the tuned/fitted **pool**
          (the base ensemble used by the DES model).
    generalization_rows : list of dict
        Metrics rows on the **test** side of the current outer fold:
        - STATIC models: test metrics computed on ``(X_test, y_test)``.
        - DES models: test metrics computed by the final preprocessing→DES pipeline
          on ``(X_test, y_test)``.

    Notes
    -----
    - The ``iteration`` and ``fold`` fields written in the output rows are 1-based
      (``iteration_idx + 1``, ``fold_idx + 1``) for readability in downstream reports.
    - Feature-selection tuning is performed by extending the search spaces with
      ``"feature_selection_filter__k"`` (when ``fs_k_best_candidates`` is provided).
      This assumes your pipeline step is named exactly ``"feature_selection_filter"``.
    - For DES models, resubstitution metrics refer to the **pool** performance (not the
      final DES inference pipeline), since the DES pipeline is trained via pool tuning
      + competence learning on DSEL.

    Examples
    --------
    Typical usage inside the main outer CV loop::

        from loguru import logger
        from sklearn.model_selection import RepeatedStratifiedKFold

        cv_outer = RepeatedStratifiedKFold(
            n_splits=10,
            n_repeats=10,
            random_state=42,
        )

        resubstitution_metrics_summary = []
        generalization_metrics_summary = []

        for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
            iteration_idx, fold_idx = divmod(run_id, 10)

            res_rows, gen_rows = train_and_evaluate_one_fold_all_models(
                run_id=run_id,
                iteration_idx=iteration_idx,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                experiment_name="baseline-v1",
                idx_num_features_to_standardize=idx_num_features_to_standardize,
                transformed_feature_names=transformed_feature_names,
                static_models=["SVC", "RandomForestClassifier"],
                des_models=["KNORAU", "METADES"],
                fs_k_best_to_keep=20,
                fs_k_best_candidates=[10, 20, 30, "all"],  # injects feature_selection_filter__k
                use_cost_sensitive_learning=True,
                resampling_method="SMOTE",
                resampling_params={"sampling_strategy": 0.2, "random_state": 42},
                tuning_n_iter=35,
                tuning_cv_inner_n_splits=5,
                tuning_scoring="average_precision",
                tuning_n_jobs=-1,
                dsel_size=0.2,
                random_state=42,
                logger=logger,
            )

            resubstitution_metrics_summary.extend(res_rows)
            generalization_metrics_summary.extend(gen_rows)
    """

    resubstitution_rows: List[Dict[str, Any]] = []
    generalization_rows: List[Dict[str, Any]] = []

    # Setup logging function (Loguru if provided, stdout otherwise)
    if logger is None:

        def _log(message: str) -> None:
            print(message)
    else:

        def _log(message: str) -> None:
            logger.info(message)

    # Split the data into training set (9 training folds) and test set (1 test fold)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Report class balance statistics for iteration
    for name, target_arr in zip(["train dataset", "test dataset"], [y_train, y_test]):
        unique, frequency = np.unique(target_arr, return_counts=True)
        _log(f"Class distribution ({name} statistics) [class, frequency]: {(unique, frequency)}")

    _log(f"[ITERATION {iteration_idx + 1:2} - FOLD {fold_idx + 1:2} - RUN_ID {run_id:3}]")

    # ----- Start training STATIC MODELS -----
    for static_model_name in static_models:
        print("-" * 165)
        _log(f"Training STATIC model: {static_model_name}")

        # Get the static model estimator and with its hyperparameter search space
        static_model_estimator, static_model_search_space = get_static_model_and_search_space(
            static_model_name,
            random_state=random_state,
            use_cost_sensitive_learning=use_cost_sensitive_learning,
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            static_model_search_space["feature_selection_filter__k"] = list(fs_k_best_candidates)

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
        static_model_pipeline = build_model_pipeline(
            estimator=static_model_estimator,
            numerical_features_to_standardize=idx_num_features_to_standardize,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=resampling_method,
            resampling_params=resampling_params,
        )

        # Tune the static model, fit on the training folds and evaluate on the test fold
        (
            best_static_model,
            tuning_results,
            resubstitution_metrics,
            test_metrics,
        ) = train_and_evaluate_one_fold_static_model(
            base_model=static_model_pipeline,
            search_space=static_model_search_space,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_iter=tuning_n_iter,
            val_cv_split=tuning_cv_inner_n_splits,
            scoring=tuning_scoring,
            random_state=random_state + run_id,
            n_jobs=tuning_n_jobs,
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_static_model,
            feature_names=transformed_feature_names,
        )

        # Collect resubstitution metrics and log
        collect_report_one_fold(
            resubstitution_rows,
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
            generalization_rows,
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

    # ----- Start training DES MODELS -----
    for des_model_name in des_models:
        print("-" * 165)
        _log(f"Training DES model: {des_model_name}")

        # Get the des model estimator and its configuration, with the
        # pool of classifiers and its hyperparameter search space
        pool_classifiers, pool_search_space, des_model_estimator, des_model_conf = get_des_model(
            des_model_name,
            random_state=random_state,
            use_cost_sensitive_learning=use_cost_sensitive_learning,
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            pool_search_space["feature_selection_filter__k"] = list(fs_k_best_candidates)

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
        pool_classifiers_pipeline = build_model_pipeline(
            estimator=pool_classifiers,
            numerical_features_to_standardize=idx_num_features_to_standardize,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=resampling_method,
            resampling_params=resampling_params,
        )

        # Tune the des model, fit on the training folds and evaluate on the test fold
        (best_des_model, tuning_results, resubstitution_metrics, test_metrics) = (
            train_and_evaluate_one_fold_des_model(
                des_model=des_model_estimator,
                des_conf=des_model_conf,
                pool_classifiers=pool_classifiers_pipeline,
                search_space=pool_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_n_iter,
                dsel_size=dsel_size,
                val_cv_split=tuning_cv_inner_n_splits,
                scoring=tuning_scoring,
                random_state=random_state + run_id,
                n_jobs=tuning_n_jobs,
            )
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_des_model,
            feature_names=transformed_feature_names,
        )

        # Collect resubstitution metrics and log for the pool
        collect_report_one_fold(
            resubstitution_rows,
            experiment_name=experiment_name,
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model=des_model_name,
            metrics=resubstitution_metrics,
            data_split="resubstitution",
            fold_size=len(X_train),
            **tuning_results,
            selected_features_indices=selected_indices,
            selected_features_names=selected_names,
        )

        # Collect generalization metrics and log
        collect_report_one_fold(
            generalization_rows,
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

    _log(f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]")

    return resubstitution_rows, generalization_rows
