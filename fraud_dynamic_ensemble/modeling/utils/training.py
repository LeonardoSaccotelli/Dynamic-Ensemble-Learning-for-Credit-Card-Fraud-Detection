from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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
    collect_fold_reports,
    compute_classification_metrics,
)
from fraud_dynamic_ensemble.modeling.utils.models import (
    get_des_model,
    get_static_ensemble_model_and_search_space,
    get_static_model_and_search_space,
)
from fraud_dynamic_ensemble.modeling.utils.pipeline import (
    build_model_pipeline,
    get_final_selected_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def run_randomized_search_cv(
    estimator: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    *,
    n_iter: int,
    val_cv_split: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    verbose: int = 3,
) -> tuple[Union[ImbPipeline, Pipeline, BaseEstimator], Dict[str, Any]]:
    """
    Run a randomized hyperparameter search (inner CV) and return the refit best estimator.

    This helper runs :class:`sklearn.model_selection.RandomizedSearchCV` over
    ``search_space`` using an inner stratified K-fold splitter:

    ``StratifiedKFold(n_splits=val_cv_split, shuffle=True, random_state=random_state)``

    The search is fit on ``(X_train, y_train)`` and the best configuration is refit on the
    full training set (``refit=True``). A compact tuning summary is extracted from
    ``search.cv_results_`` at ``search.best_index_``.

    Parameters
    ----------
    estimator : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Estimator (or pipeline) to tune. Must implement ``fit`` and expose hyperparameters
        via ``get_params`` so RandomizedSearchCV can route parameters.
    search_space : dict[str, Any]
        Hyperparameter search space forwarded as ``param_distributions`` to
        :class:`sklearn.model_selection.RandomizedSearchCV`.

        Keys must match valid parameter names for ``estimator`` (e.g., ``"C"`` for a bare
        estimator, or ``"classifier__C"`` for a pipeline step named ``"classifier"``).
        Values may be candidate lists and/or SciPy distribution objects.
    X_train : pandas.DataFrame or numpy.ndarray
        Training features of shape ``(n_samples, n_features)``.
    y_train : pandas.Series or numpy.ndarray
        Training labels of shape ``(n_samples,)``.
    n_iter : int
        Number of hyperparameter configurations to sample. Must be >= 1.
    val_cv_split : int
        Number of folds for inner CV. Must be >= 2.
    scoring : str
        Scikit-learn scoring identifier (e.g., ``"f1"``, ``"roc_auc"``, ``"average_precision"``).
    random_state : int
        Random seed used for CV shuffling and randomized hyperparameter sampling.
    n_jobs : int
        Number of parallel jobs used by RandomizedSearchCV. Use ``-1`` for all available cores.
    verbose : int, default=3
        Verbosity level forwarded to RandomizedSearchCV.

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Best estimator found by RandomizedSearchCV, refit on the full training set.
    tuning_results : dict[str, Any]
        Standardized tuning summary for the best candidate with keys:

        - ``"cv_tuning_mean_train_score"`` : float
        - ``"cv_tuning_std_train_score"`` : float
        - ``"cv_tuning_mean_val_score"`` : float
        - ``"cv_tuning_std_val_score"`` : float
        - ``"best_params"`` : dict[str, Any]
        - ``"tuning_time"`` : float
            Wall-clock time in seconds spent inside ``search.fit``.

    Raises
    ------
    ValueError
        If ``n_iter < 1`` or ``val_cv_split < 2``.
    Exception
        Any exception raised during fitting is propagated. Because ``error_score="raise"``
        is set, failures during CV are not masked.

    Notes
    -----
    - The reported train/validation scores refer to the **inner** CV used by the randomized
      search (not the outer evaluation loop).
    - To avoid nested parallelism, keep estimator-level parallelism disabled (e.g., model
      ``n_jobs=1``) when RandomizedSearchCV runs with ``n_jobs > 1``.
    - This function prints basic tuning settings and the best params to stdout. If you need
      structured logging, wrap this function and redirect/replace prints with a logger.

    Examples
    --------
    >>> clf, space = get_static_model_and_search_space("SVC", random_state=42)
    >>> best_model, tuning = run_randomized_search_cv(
    ...     estimator=clf,
    ...     search_space=space,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     n_iter=30,
    ...     val_cv_split=5,
    ...     scoring="f1",
    ...     random_state=42,
    ...     n_jobs=-1,
    ...     verbose=2,
    ... )
    >>> tuning["best_params"]  # doctest: +ELLIPSIS
    {...}
    """

    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1. Got {n_iter}.")
    if val_cv_split < 2:
        raise ValueError(f"val_cv_split must be >= 2. Got {val_cv_split}.")

    print(
        f"[RANDOMIZED SEARCH SETTINGS]: scoring: {scoring}, random_state: {random_state}, n_jobs: {n_jobs}"
    )

    splitter = StratifiedKFold(
        n_splits=val_cv_split,
        random_state=random_state,
        shuffle=True,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        cv=splitter,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )

    start_tuning_time = time.time()
    search.fit(X_train, y_train)
    end_tuning_time = time.time()

    best_model = search.best_estimator_

    tuning_results = {
        "cv_tuning_mean_train_score": search.cv_results_["mean_train_score"][search.best_index_],
        "cv_tuning_std_train_score": search.cv_results_["std_train_score"][search.best_index_],
        "cv_tuning_mean_val_score": search.cv_results_["mean_test_score"][search.best_index_],
        "cv_tuning_std_val_score": search.cv_results_["std_test_score"][search.best_index_],
        "best_params": search.best_params_,
        "tuning_time": end_tuning_time - start_tuning_time,
    }

    print(f"[RANDOMIZED SEARCH BEST PARAMS]: {tuning_results['best_params']}")

    return best_model, tuning_results


def train_and_evaluate_one_fold_static_model(
    base_model: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    logger: Any,
    n_iter: int,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Union[ImbPipeline, Pipeline, BaseEstimator],
    Dict[str, Any],
    Dict[str, Optional[float] | int],
    Dict[str, Optional[float] | int],
]:
    """
    Tune and evaluate a static (non-DES) classifier on a single outer CV fold.

    The function performs inner-CV hyperparameter tuning on the provided outer training
    split using :func:`run_randomized_search_cv`, refits the best configuration on the
    full outer training data, and then computes classification metrics on both the
    training (resubstitution) and outer test (generalization) splits using
    :func:`compute_classification_metrics`.

    Test-time inference latency is measured on ``X_test`` around the prediction calls
    and returned as ``"score_time"`` within the test metrics.

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Estimator or pipeline to tune and evaluate. Must implement ``fit`` and
        ``predict``. This implementation requires ``predict_proba`` to compute
        probability-based metrics.
    search_space : dict[str, Any]
        Hyperparameter search space passed to the tuning routine. Keys must match valid
        parameter names of ``base_model`` (e.g., ``"classifier__C"`` for a pipeline
        step named ``"classifier"``).
    X_train : pandas.DataFrame or numpy.ndarray
        Training features for the current outer fold of shape ``(n_train, n_features)``.
    y_train : pandas.Series or numpy.ndarray
        Training labels for the current outer fold of shape ``(n_train,)``.
    X_test : pandas.DataFrame or numpy.ndarray
        Test features for the current outer fold of shape ``(n_test, n_features)``.
    y_test : pandas.Series or numpy.ndarray
        Test labels for the current outer fold of shape ``(n_test,)``.
    logger : Any
        Logger-like object exposing an ``info(str)`` method used for progress messages.
    n_iter : int
        Number of parameter configurations sampled during randomized search.
    val_cv_split : int, default=5
        Number of inner stratified CV folds used during tuning.
    scoring : str, default="f1"
        Scoring identifier used to select the best configuration during tuning.
    random_state : int, default=42
        Random seed forwarded to the tuning routine (inner splitter and parameter sampling).
    n_jobs : int, default=-1
        Number of parallel jobs used during tuning. To avoid nested parallelism, set
        estimator-level ``n_jobs`` appropriately (often ``1``).

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Best estimator found by tuning, refit on the full outer training split.
    tuning_results : dict[str, Any]
        Standardized tuning summary returned by :func:`run_randomized_search_cv`
        (e.g., inner-CV mean/std scores, best params, tuning time).
    resubstitution_metrics : dict[str, float | int | None]
        Metrics computed on the outer training split via
        :func:`compute_classification_metrics`.
    test_metrics : dict[str, float | int | None]
        Metrics computed on the outer test split via
        :func:`compute_classification_metrics`, with an additional key:

        - ``"score_time"`` : float
          Wall-clock time in seconds measured around ``predict`` and ``predict_proba``
          on ``X_test``.

    Raises
    ------
    AttributeError
        If the refit ``best_model`` does not expose ``predict_proba``.
    ValueError
        If tuning fails due to invalid ``search_space`` keys, incompatible ``scoring``,
        or an invalid CV configuration (raised by scikit-learn inside the tuning helper).
    Exception
        Any exception raised by the underlying estimator or pipeline during tuning,
        fitting, or prediction may propagate.

    Notes
    -----
    - This implementation assumes binary classification and extracts positive-class
      probabilities as ``predict_proba(X)[:, 1]``.
    - ``score_time`` includes both hard predictions and probability predictions on the
      test split.
    - Leakage safety depends on encapsulating preprocessing, feature selection, and
      resampling inside ``base_model`` when the function is used within an outer CV loop.

    Examples
    --------
    >>> best_model, tuning_results, resub_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
    ...     base_model=base_model,
    ...     search_space=search_space,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     logger=logger,
    ...     n_iter=30,
    ...     val_cv_split=5,
    ...     scoring="average_precision",
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )
    >>> tuning_results["best_params"]  # doctest: +ELLIPSIS
    {...}
    """

    # Run RandomizedSearchCV
    best_model, tuning_results = run_randomized_search_cv(
        estimator=base_model,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        n_iter=n_iter,
        val_cv_split=val_cv_split,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Evaluate on the training set (resubstitution error)
    logger.info("[COMPUTING RESUBSTITUTION METRICS]...")
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    resubstitution_metrics = compute_classification_metrics(
        y_train, y_train_pred, y_train_pred_prob
    )
    logger.info(f"[RESUBSTITUTION METRICS]: {resubstitution_metrics}")

    # Evaluate on the test set (generalization error)
    logger.info("[COMPUTING GENERALIZATION METRICS]...")
    start_score_time = time.time()
    y_test_pred = best_model.predict(X_test)
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]
    end_score_time = time.time()

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_prob)
    test_metrics["score_time"] = end_score_time - start_score_time
    logger.info(f"[GENERALIZATION METRICS]: {test_metrics}")

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
    logger: Any,
    n_iter: int,
    dsel_size: float = 0.2,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Pipeline,
    Dict[str, Any],
    Dict[str, Optional[float] | int],
    Dict[str, Optional[float] | int],
]:
    """
    Train and evaluate a Dynamic Ensemble Selection (DES) model on a single outer CV fold.

    This helper implements a leakage-safe two-stage DES workflow within one outer split:

    1) Split the outer training fold into a pool-training subset and a DSEL subset using a
       stratified ``train_test_split`` controlled by ``random_state``.

    2) Tune and refit the pool pipeline on the pool-training subset via
       :func:`run_randomized_search_cv`. The returned best pipeline is then used to compute
       pool resubstitution metrics on the pool-training subset.

    3) Fit the DES model on DSEL:
       - extract the fitted preprocessing part of the tuned pool pipeline by slicing it up to
         (but excluding) the step named ``"resampling"``,
       - transform ``X_dsel`` with the fitted preprocessing,
       - extract the fitted pool estimator from the tuned pipeline step named ``"classifier"``,
       - inject the fitted pool into a local copy of ``des_conf`` under ``"pool_classifiers"``,
         then call ``des_model.set_params(**...)`` and ``des_model.fit(...)`` on transformed DSEL.

    4) Build an inference pipeline (preprocessing -> DES) and evaluate on the outer test fold.
       Test-time predictions are timed and stored as ``"score_time"`` inside the returned
       ``test_metrics``. Probabilities are attempted via ``predict_proba``; if unavailable,
       probability-based metrics are returned as ``None`` by
       :func:`compute_classification_metrics`.

    Parameters
    ----------
    des_model : sklearn.base.BaseEstimator
        Unfitted DES estimator (typically from DESlib) implementing ``set_params``, ``fit``,
        and ``predict``. If the final inference pipeline exposes ``predict_proba``,
        probability-based metrics can be computed on the test fold.
    des_conf : Dict[str, Any]
        Configuration forwarded to ``des_model.set_params(**des_conf_local)``. This function
        creates a local shallow copy to inject the fitted pool under the ``"pool_classifiers"``
        key before calling ``set_params``.
    pool_classifiers : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Pool estimator to tune on the pool-training subset. In the current implementation this
        is expected to behave like a fitted pipeline after tuning (i.e., expose ``named_steps``
        and support slicing), and to contain:
        - a step named ``"resampling"`` (used only during training; excluded from inference),
        - a step named ``"classifier"`` (the fitted pool injected into the DES model).
    search_space : Dict[str, Any]
        Hyperparameter distributions or candidate lists used to tune ``pool_classifiers``.
        Keys must match valid parameter names of the pool pipeline using the double-underscore
        convention (e.g., ``"classifier__n_estimators"``, ``"feature_selection_filter__k"``).
    X_train : Union[pandas.DataFrame, numpy.ndarray]
        Features for the outer training fold of shape ``(n_train_samples, n_features)``.
    y_train : Union[pandas.Series, numpy.ndarray]
        Labels for the outer training fold of shape ``(n_train_samples,)``.
    X_test : Union[pandas.DataFrame, numpy.ndarray]
        Features for the outer test fold of shape ``(n_test_samples, n_features)``.
    y_test : Union[pandas.Series, numpy.ndarray]
        Labels for the outer test fold of shape ``(n_test_samples,)``.
    logger : Any
        Logger exposing ``.info(str)``.
    n_iter : int
        Number of hyperparameter configurations sampled during pool tuning.
    dsel_size : float, default=0.2
        Fraction of ``X_train`` reserved for DSEL (must satisfy ``0 < dsel_size < 1``).
        The split is stratified by ``y_train``.
    val_cv_split : int, default=5
        Number of folds for the inner CV used during pool tuning.
    scoring : str, default="f1"
        Scoring identifier used by the randomized CV search to select the best pool
        configuration (e.g., ``"f1"``, ``"roc_auc"``, ``"average_precision"``).
    random_state : int, default=42
        Random seed used for:
        - the pool-train vs DSEL split,
        - the inner CV splitter shuffling,
        - the randomized hyperparameter sampling.
    n_jobs : int, default=-1
        Number of parallel jobs used by the randomized CV search. To avoid nested parallelism,
        configure underlying pool estimators with ``n_jobs=1`` when appropriate.

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline used for test evaluation. It contains the fitted preprocessing
        steps extracted from the tuned pool pipeline (all steps before ``"resampling"``),
        followed by the fitted DES estimator as the final step named ``"classifier"``.
    tuning_results : Dict[str, Any]
        Tuning summary returned by :func:`run_randomized_search_cv` for the best pool candidate
        (e.g., CV mean/std scores, best params, tuning time).
    pool_resubstitution_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the pool-training subset using the tuned pool pipeline. This
        requires that the tuned pool pipeline implements ``predict_proba``.
    test_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the outer test fold using the final DES inference pipeline. The
        returned dictionary also includes ``"score_time"`` (seconds), measured around the test
        prediction calls in this function.

    Raises
    ------
    ValueError
        If ``dsel_size`` is not in ``(0, 1)``, if stratified splitting fails, if tuning fails
        due to invalid configuration (e.g., incompatible ``scoring`` or invalid parameter names
        in ``search_space``), or if the tuned pool pipeline does not include a step named
        ``"resampling"`` (required for preprocessing extraction).
    KeyError
        If the tuned pool pipeline does not contain a step named ``"classifier"`` when
        extracting the fitted pool.
    AttributeError
        If the tuned pool pipeline does not implement ``predict_proba`` (required for pool
        resubstitution metrics in this implementation), or if required pipeline-like attributes
        (e.g., ``named_steps``) are not available.
    Exception
        Any exception raised by the underlying estimators/pipeline during fitting, transformation,
        or prediction may propagate.

    Notes
    -----
    - No outer-test leakage: the outer test fold is used only for final evaluation; pool tuning
      and DES fitting occur exclusively within the outer training fold.
    - Train-time-only resampling: the pool pipeline step named ``"resampling"`` is excluded from
      the final inference pipeline.
    - Binary classification convention: when available, the positive-class probability is taken
      as ``predict_proba(X)[:, 1]``.
    - Test probabilities are optional: if ``final_des_pipeline.predict_proba`` is not available,
      probabilities are set to ``None`` and probability-based metrics are returned as ``None`` by
      :func:`compute_classification_metrics`.

    Examples
    --------
    >>> final_pipe, tuning, pool_resub, test_metrics = train_and_evaluate_one_fold_des_model(des_model=des_model, des_conf=des_conf, pool_classifiers=pool_pipeline, search_space=pool_space, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, logger=logger, n_iter=30, dsel_size=0.2, val_cv_split=5, scoring="average_precision", random_state=42, n_jobs=-1)
    >>> test_metrics["score_time"]
    """

    # Split TRAIN into pool-training and DSEL
    X_train_pool, X_dsel, y_train_pool, y_dsel = train_test_split(
        X_train,
        y_train,
        test_size=dsel_size,
        stratify=y_train,
        random_state=random_state,
    )

    # Run RandomizedSearchCV
    best_pipe_pool_classifiers, tuning_results = run_randomized_search_cv(
        estimator=pool_classifiers,
        search_space=search_space,
        X_train=X_train_pool,
        y_train=y_train_pool,
        n_iter=n_iter,
        val_cv_split=val_cv_split,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # --- Pool resubstitution metrics (only for the tuned pool pipeline) ---
    logger.info("[COMPUTING POOL RESUBSTITUTION METRICS]...")
    y_train_pool_pred = best_pipe_pool_classifiers.predict(X_train_pool)

    # Consistent with the STATIC reference function: require predict_proba
    y_train_pool_pred_prob = best_pipe_pool_classifiers.predict_proba(X_train_pool)[:, 1]

    pool_resubstitution_metrics = compute_classification_metrics(
        y_train_pool, y_train_pool_pred, y_train_pool_pred_prob
    )
    logger.info(f"[POOL RESUBSTITUTION METRICS]: {pool_resubstitution_metrics}")

    # Extract the preprocessing pipeline (fitted)
    # We skip resampling step since it is required just at training time
    resampling_idx = list(best_pipe_pool_classifiers.named_steps.keys()).index("resampling")
    fitted_preproc = best_pipe_pool_classifiers[:resampling_idx]

    # Apply the preprocessing steps on the DSEL dataset
    X_dsel_trans = fitted_preproc.transform(X_dsel)

    # Extract the fitted pool of classifiers
    fitted_pool = best_pipe_pool_classifiers.named_steps["classifier"]

    # Add the trained pool of classifiers to the DES model config
    des_conf_local = dict(des_conf)  # shallow copy is enough
    des_conf_local["pool_classifiers"] = fitted_pool

    # Fit DES model on DSEL in transformed space
    logger.info("[FITTING DSEL METHOD]...")
    des_model.set_params(**des_conf_local)
    des_model.fit(X_dsel_trans, y_dsel)

    # Final inference pipeline: preprocessing -> DES
    final_des_pipeline = Pipeline(fitted_preproc.steps + [("classifier", des_model)])

    # Evaluate on the test set (generalization error)
    logger.info("[COMPUTING GENERALIZATION METRICS]...")
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
    logger.info(f"[GENERALIZATION METRICS]: {test_metrics}")

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
    static_ensemble_models: Sequence[str],
    static_ensemble_pools: Sequence[str],
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
    logger: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train and evaluate all STATIC, STATIC-ENSEMBLE, and DES models on a single outer CV fold.

    This helper orchestrates the end-to-end workflow for one outer split of a repeated
    stratified cross-validation experiment. Given global feature/label arrays and the current
    outer train/test indices, it:

    1) builds the outer fold datasets and logs class distributions,
    2) trains and evaluates each STATIC model listed in ``static_models``,
    3) trains and evaluates each STATIC ENSEMBLE model listed in ``static_ensemble_models``,
       using the shared base-model pool ``static_ensemble_pools``,
    4) trains and evaluates each DES model listed in ``des_models`` using a two-stage DES
       workflow (pool tuning + DSEL fitting).

    For each trained model, this function collects standardized reporting rows for both:
    - training-side evaluation (resubstitution or pool-resubstitution), and
    - test-side evaluation (generalization metrics on the outer test fold),

    by delegating row construction to :func:`collect_fold_reports`.

    Parameters
    ----------
    run_id : int
        Global counter for the current outer split, typically produced by
        ``enumerate(cv_outer.split(X, y))``. Used to diversify tuning randomness as
        ``random_state + run_id``.
    iteration_idx : int
        0-based repetition index. Stored as 1-based in output rows via ``iteration_idx + 1``.
    fold_idx : int
        0-based fold index within the repetition. Stored as 1-based in output rows via
        ``fold_idx + 1``.
    train_idx : numpy.ndarray
        Indices of samples used as training data for this outer fold.
    test_idx : numpy.ndarray
        Indices of samples used as test data for this outer fold.
    X : numpy.ndarray of shape (n_samples, n_features)
        Full feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Full target vector.
    experiment_name : str
        Experiment identifier stored in every output row.
    idx_num_features_to_standardize : Sequence[int]
        Column indices standardized by the preprocessing step inside
        :func:`build_model_pipeline`.
    transformed_feature_names : Sequence[str]
        Feature names aligned with the input to the ``"feature_selection_filter"`` step
        (i.e., after the ``"preprocessor"`` transformation). Used to map selected feature
        indices back to names via :func:`get_final_selected_features`.
    static_models : Sequence[str]
        Names of static (single-estimator) models to train (e.g., ``["SVC", "RandomForestClassifier"]``).
    static_ensemble_models : Sequence[str]
        Names of static ensemble types to train (e.g., ``["VotingClassifier", "StackingClassifier"]``).
        Forwarded as ``ensemble_type`` to :func:`get_static_ensemble_model_and_search_space`.
    static_ensemble_pools : Sequence[str]
        Pool of base-model names used to build each static ensemble instance in this fold
        (e.g., ``["SVC", "XGBClassifier"]``). The same pool is reused for all ensemble types
        listed in ``static_ensemble_models``.
    des_models : Sequence[str]
        Names of DES models to train (e.g., ``["KNORAU", "METADES"]``). Forwarded to
        :func:`get_des_model`.
    fs_k_best_to_keep : int or str
        Default ``k`` used when constructing the ``SelectKBest(k=...)`` step inside
        :func:`build_model_pipeline`. If the tuning search space includes
        ``"feature_selection_filter__k"``, the fitted value may differ from this default.
    fs_k_best_candidates : Sequence[int | str] or None
        Optional candidate values for ``SelectKBest.k`` to be explored during tuning. When
        provided, candidates are injected into the relevant search spaces by adding
        ``"feature_selection_filter__k": list(fs_k_best_candidates)``.
    use_cost_sensitive_learning : bool
        Whether to enable cost-sensitive behavior (e.g., class weights) in STATIC models,
        STATIC ensembles, and DES pools. Forwarded to the model factory functions.
    resampling_method : str or None
        Canonical name of the resampling strategy (e.g., ``"SMOTE"``, ``"RandomUnderSampler"``,
        ``"SMOTEENN"``) or ``None`` to disable resampling.
    resampling_params : Dict[str, Any] or None
        Extra keyword arguments forwarded to the sampler constructor when resampling is enabled.
    tuning_n_iter : int
        Number of parameter settings sampled in the randomized hyperparameter search.
    tuning_cv_inner_n_splits : int
        Number of stratified folds for the inner CV used during hyperparameter tuning.
    tuning_scoring : str
        Scoring metric used by the hyperparameter search (e.g., ``"f1"``,
        ``"average_precision"``, ``"roc_auc"``).
    tuning_n_jobs : int
        Number of parallel jobs used during hyperparameter tuning.
    dsel_size : float
        Fraction of the outer training set reserved for the DSEL subset used to fit DES
        competence models (must satisfy ``0 < dsel_size < 1``).
    random_state : int
        Base random seed forwarded to factories and splitters. Tuning calls use
        ``random_state + run_id`` to diversify randomized search sampling across outer folds.
    logger : Any
        Logger instance exposing ``.info(str)``.

    Returns
    -------
    resubstitution_rows : List[Dict[str, Any]]
        Metrics rows on the training side of the current outer fold.
        - STATIC models: resubstitution metrics on ``(X_train, y_train)``.
        - STATIC ENSEMBLES: resubstitution metrics on ``(X_train, y_train)``.
        - DES models: pool-resubstitution metrics computed for the tuned/fitted pool on the
          pool-training subset used inside :func:`train_and_evaluate_one_fold_des_model`.
    generalization_rows : List[Dict[str, Any]]
        Metrics rows on the test side of the current outer fold.
        - STATIC models: test metrics on ``(X_test, y_test)``.
        - STATIC ENSEMBLES: test metrics on ``(X_test, y_test)``.
        - DES models: test metrics produced by the final DES inference pipeline on
          ``(X_test, y_test)``.

    Raises
    ------
    ValueError
        If any factory function receives an unsupported model key, or if downstream tuning/evaluation
        helpers raise due to invalid CV configuration, invalid parameter names, incompatible scoring,
        or invalid ``dsel_size``.
    KeyError
        If downstream helpers expect specific pipeline step names that are missing (e.g.,
        ``"feature_selection_filter"``, ``"resampling"``, ``"classifier"``).
    AttributeError
        If downstream evaluation requires probability estimates but the fitted estimator/pipeline does
        not expose ``predict_proba``.
    Exception
        Any exception raised by underlying estimators, samplers, or scikit-learn model-selection
        routines may propagate.

    Notes
    -----
    - Output rows use 1-based ``iteration`` and ``fold`` indices (``iteration_idx + 1``,
      ``fold_idx + 1``).
    - Feature-selection tuning via ``"feature_selection_filter__k"`` assumes the pipeline step is
      named exactly ``"feature_selection_filter"``.
    - STATIC evaluation via :func:`train_and_evaluate_one_fold_static_model` requires ``predict_proba``.
      Therefore, STATIC ENSEMBLES must be configured to expose probabilities (e.g., soft voting).
    - To avoid nested parallelism on HPC, align estimator-level ``n_jobs`` to 1 and control parallelism
      primarily via the tuning layer (``tuning_n_jobs``).

    Examples
    --------
    Typical usage inside an outer CV loop::

        from loguru import logger
        from sklearn.model_selection import RepeatedStratifiedKFold

        cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

        resub_rows_all = []
        gen_rows_all = []

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
                static_ensemble_models=["VotingClassifier", "StackingClassifier"],
                static_ensemble_pools=["SVC", "XGBClassifier"],
                des_models=["KNORAU", "METADES"],
                fs_k_best_to_keep=20,
                fs_k_best_candidates=[10, 20, 30, "all"],
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

            resub_rows_all.extend(res_rows)
            gen_rows_all.extend(gen_rows)
    """

    resubstitution_rows: List[Dict[str, Any]] = []
    generalization_rows: List[Dict[str, Any]] = []

    # Split the data into training set (9 training folds) and test set (1 test fold)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Report class balance statistics for iteration
    for name, target_arr in zip(["train dataset", "test dataset"], [y_train, y_test]):
        unique, frequency = np.unique(target_arr, return_counts=True)
        logger.info(
            f"Class distribution ({name} statistics) [class, frequency]: {(unique, frequency)}"
        )

    logger.info(f"[ITERATION {iteration_idx + 1:2} - FOLD {fold_idx + 1:2} - RUN_ID {run_id:3}]")

    # ----- Start training STATIC MODELS -----
    for static_model_name in static_models:
        print("-" * 165)
        logger.info(f"Training STATIC model: {static_model_name}")

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
        best_static_model, tuning_results, resubstitution_metrics, test_metrics = (
            train_and_evaluate_one_fold_static_model(
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
                logger=logger,
            )
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_static_model,
            feature_names=transformed_feature_names,
        )

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_name,
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=static_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,  # keep tuning info on resub row
            selected_features_indices=selected_indices,
            selected_features_names=selected_names,
        )

    # ----- Start training STATIC ENSEMBLE MODELS -----
    for static_ensemble_model_name in static_ensemble_models:
        print("-" * 165)
        logger.info(f"Training STATIC ENSEMBLE model: {static_ensemble_model_name}")

        # Get the static ensemble model estimator with its hyperparameter search space
        static_ensemble_model_estimator, static_ensemble_model_search_space = (
            get_static_ensemble_model_and_search_space(
                ensemble_type=static_ensemble_model_name,
                model_pool=static_ensemble_pools,
                random_state=random_state,
                use_cost_sensitive_learning=use_cost_sensitive_learning,
            )
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            static_ensemble_model_search_space["feature_selection_filter__k"] = list(
                fs_k_best_candidates
            )

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
        static_ensemble_model_pipeline = build_model_pipeline(
            estimator=static_ensemble_model_estimator,
            numerical_features_to_standardize=idx_num_features_to_standardize,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=resampling_method,
            resampling_params=resampling_params,
        )

        # Tune the static model, fit on the training folds and evaluate on the test fold
        best_static_ensemble_model, tuning_results, resubstitution_metrics, test_metrics = (
            train_and_evaluate_one_fold_static_model(
                base_model=static_ensemble_model_pipeline,
                search_space=static_ensemble_model_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_n_iter,
                val_cv_split=tuning_cv_inner_n_splits,
                scoring=tuning_scoring,
                random_state=random_state + run_id,
                n_jobs=tuning_n_jobs,
                logger=logger,
            )
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_static_ensemble_model,
            feature_names=transformed_feature_names,
        )

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_name,
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=static_ensemble_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,  # keep tuning info on resub row
            selected_features_indices=selected_indices,
            selected_features_names=selected_names,
        )

    # ----- Start training DES MODELS -----
    for des_model_name in des_models:
        print("-" * 165)
        logger.info(f"Training DES model: {des_model_name}")

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
        best_des_model, tuning_results, resubstitution_metrics, test_metrics = (
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
                logger=logger,
            )
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_des_model,
            feature_names=transformed_feature_names,
        )

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_name,
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=des_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,
            selected_features_indices=selected_indices,
            selected_features_names=selected_names,
        )

    logger.info(
        f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]"
    )

    return resubstitution_rows, generalization_rows
