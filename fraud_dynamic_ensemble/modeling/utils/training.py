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
    Run hyperparameter tuning with :class:`sklearn.model_selection.RandomizedSearchCV` and
    return the best refit estimator along with a standardized tuning summary.

    This utility performs a randomized hyperparameter search over ``search_space`` using an
    inner stratified K-fold cross-validation splitter. The search is fit on
    ``(X_train, y_train)`` and refits the best configuration on the full training set.

    It returns:

    1. ``best_model``: the best estimator found by the search, refit on the full training set.
    2. ``tuning_results``: a compact dictionary summarizing the best candidate's inner-CV
       train/validation performance, best hyperparameters, and wall-clock tuning time.

    The inner CV splitter is:

    - ``StratifiedKFold(n_splits=val_cv_split, shuffle=True, random_state=random_state)``

    Train/validation score statistics are extracted from ``search.cv_results_`` at
    ``search.best_index_``. Because ``return_train_score=True`` is enabled, both training and
    validation statistics for the inner CV are available.

    Parameters
    ----------
    estimator : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Estimator or pipeline to tune. Must be compatible with scikit-learn's model selection
        API (i.e., implement ``fit`` and expose tunable parameters via ``get_params``).
    search_space : Dict[str, Any]
        Hyperparameter search space forwarded to ``param_distributions`` in
        :class:`sklearn.model_selection.RandomizedSearchCV`.

        Keys must match valid parameter names for ``estimator`` (e.g., ``"C"`` for a bare
        estimator, or ``"classifier__C"`` for a pipeline step named ``"classifier"``).
        Values may be:
        - explicit candidate lists, and/or
        - distribution objects compatible with RandomizedSearchCV sampling.
    X_train : Union[pandas.DataFrame, numpy.ndarray]
        Training features of shape ``(n_samples, n_features)``.
    y_train : Union[pandas.Series, numpy.ndarray]
        Training labels/targets of shape ``(n_samples,)``.
    n_iter : int
        Number of hyperparameter configurations sampled in the randomized search.
    val_cv_split : int
        Number of folds for the inner stratified K-fold cross-validation (must be >= 2).
    scoring : str
        Scoring identifier used by RandomizedSearchCV (e.g., ``"f1"``, ``"roc_auc"``,
        ``"average_precision"``). Must be a valid scikit-learn scoring identifier.
    random_state : int
        Random seed used for both:
        - shuffling in the inner StratifiedKFold splitter, and
        - randomized sampling of hyperparameters in RandomizedSearchCV.
    n_jobs : int
        Number of parallel jobs used by RandomizedSearchCV. Use ``-1`` to use all available
        cores.
    verbose : int, default=3
        Verbosity level forwarded to RandomizedSearchCV.

    Returns
    -------
    best_model : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Best estimator found by RandomizedSearchCV, refit on the full training set.
    tuning_results : Dict[str, Any]
        Standardized tuning summary for the best candidate with keys:

        - ``"cv_tuning_mean_train_score"`` : float
            Mean training score (inner CV) at the best candidate.
        - ``"cv_tuning_std_train_score"`` : float
            Standard deviation of training score (inner CV) at the best candidate.
        - ``"cv_tuning_mean_val_score"`` : float
            Mean validation score (inner CV) at the best candidate.
        - ``"cv_tuning_std_val_score"`` : float
            Standard deviation of validation score (inner CV) at the best candidate.
        - ``"best_params"`` : Dict[str, Any]
            Best hyperparameters found (as returned by ``search.best_params_``).
        - ``"tuning_time"`` : float
            Wall-clock time in seconds spent inside ``search.fit``.

    Raises
    ------
    ValueError
        If ``val_cv_split`` is invalid, if ``scoring`` is invalid, or if ``search_space``
        contains invalid parameter names (raised by scikit-learn during initialization
        or fitting).
    Exception
        Any exception raised during fitting is propagated. In particular, because
        ``error_score="raise"`` is set, any estimator failure inside CV will raise
        immediately rather than being converted to a score.

    Notes
    -----
    - The "train" and "val" scores in ``tuning_results`` refer to the *inner-CV* splits used
      by RandomizedSearchCV (not to any outer evaluation protocol).
    - Progress is printed to stdout (tuning settings and best parameters). If you need
      structured logging, call this function from a wrapper that redirects prints to a
      logger.
    - To avoid nested parallelism, consider setting internal estimator ``n_jobs`` to 1 when
      RandomizedSearchCV uses ``n_jobs > 1``.

    Examples
    --------
    Tune a pipeline with a classifier step named ``"classifier"``::

        estimator, search_space = get_static_model_and_search_space("SVC", random_state=42)

        best_model, tuning_summary = run_randomized_search_cv(
            estimator=estimator,
            search_space=search_space,
            X_train=X_train,
            y_train=y_train,
            n_iter=30,
            val_cv_split=5,
            scoring="f1",
            random_state=42,
            n_jobs=-1,
            verbose=2,
        )

        print(tuning_summary["best_params"])
    """

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

    This helper runs inner-CV hyperparameter tuning on the provided outer training split,
    refits the best configuration on the full outer training data, and then reports
    performance on both the training (resubstitution) and held-out outer test split.

    Workflow
    --------
    1. **Hyperparameter tuning (inner CV)**:
       Uses :func:`run_randomized_search_cv` to perform randomized search over ``search_space``
       on ``(X_train, y_train)`` with a stratified K-fold splitter.

    2. **Resubstitution evaluation (train metrics)**:
       Computes metrics on ``(X_train, y_train)`` using the refit best estimator.

    3. **Generalization evaluation (test metrics)**:
       Computes metrics on ``(X_test, y_test)`` using the refit best estimator and appends
       ``"score_time"`` measuring the wall-clock time spent generating both hard predictions
       and probabilities on ``X_test`` (i.e., it wraps ``predict`` + ``predict_proba`` in this
       implementation).

    All metrics are computed via :func:`compute_classification_metrics`. This implementation
    assumes binary classification and extracts positive-class probabilities as
    ``predict_proba(X)[:, 1]``.

    Parameters
    ----------
    base_model : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Estimator or pipeline to tune and evaluate. Must implement ``fit`` and ``predict``.
        This implementation also requires ``predict_proba`` to compute probability-based
        metrics on both train and test splits.

        If ``base_model`` is a pipeline, parameter names in ``search_space`` must follow the
        scikit-learn double-underscore convention (e.g., ``"classifier__C"``,
        ``"feature_selection_filter__k"``).
    search_space : Dict[str, Any]
        Hyperparameter distributions or explicit candidate lists used during randomized tuning.
        Keys must correspond to valid parameters of ``base_model``.
    X_train : Union[pandas.DataFrame, numpy.ndarray]
        Training features for the outer fold, shape ``(n_train_samples, n_features)``.
    y_train : Union[pandas.Series, numpy.ndarray]
        Training labels for the outer fold, shape ``(n_train_samples,)``.
    X_test : Union[pandas.DataFrame, numpy.ndarray]
        Test features for the outer fold, shape ``(n_test_samples, n_features)``.
    y_test : Union[pandas.Series, numpy.ndarray]
        Test labels for the outer fold, shape ``(n_test_samples,)``.
    logger : Any
        Logger exposing ``.info(str)``.
    n_iter : int
        Number of sampled hyperparameter configurations used by the randomized search.
    val_cv_split : int, default=5
        Number of folds for the inner stratified CV used during tuning.
    scoring : str, default="f1"
        Scoring identifier used to select the best hyperparameter configuration during tuning
        (e.g., ``"f1"``, ``"average_precision"``, ``"roc_auc"``).
    random_state : int, default=42
        Random seed forwarded to the inner CV splitter (shuffling) and to the randomized
        hyperparameter sampling.
    n_jobs : int, default=-1
        Parallelism level used during hyperparameter tuning. Use ``-1`` to use all available
        cores. To avoid nested parallelism, configure the underlying estimator's own ``n_jobs``
        appropriately (often ``1`` on HPC).

    Returns
    -------
    best_model : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Best estimator found by the randomized search, refit on the full outer training split.
    tuning_results : Dict[str, Any]
        Tuning summary returned by :func:`run_randomized_search_cv` for the best candidate
        (e.g., CV mean/std scores, best params, tuning time).
    resubstitution_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the training split via :func:`compute_classification_metrics`.
    test_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the test split via :func:`compute_classification_metrics`, plus:
        - ``"score_time"`` : float
          Wall-clock time (seconds) measured around the test-time prediction calls in this
          function (``predict`` and ``predict_proba``).

    Raises
    ------
    AttributeError
        If the refit ``best_model`` does not expose ``predict_proba`` (required by this
        implementation).
    ValueError
        If tuning fails due to invalid ``search_space`` keys, incompatible ``scoring``, or an
        invalid CV configuration (raised by scikit-learn inside :func:`run_randomized_search_cv`).
    Exception
        Any exception raised by the underlying estimator/pipeline during fitting or prediction
        may propagate (in particular, because the tuning helper uses ``error_score="raise"``).

    Notes
    -----
    - **Binary classification convention:** probabilities are extracted as
      ``predict_proba(X)[:, 1]``.
    - ``score_time`` as implemented includes both ``predict`` and ``predict_proba`` on the
      test split. If you want to time only hard predictions, move the timer so it wraps only
      ``predict``.
    - Any preprocessing, feature selection, and/or resampling should be encapsulated inside
      ``base_model`` when it is a pipeline; when used in CV, this is what ensures leakage-safe
      fitting of those steps.

    Examples
    --------
    >>> best_model, tuning_results, resubstitution_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
    ...     base_model=base_model,
    ...     search_space={"classifier__C": [0.1, 1.0, 10.0]},
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     logger=logger,
    ...     n_iter=10,
    ...     val_cv_split=5,
    ...     scoring="average_precision",
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )
    >>> tuning_results["best_params"]
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

    This helper implements a leakage-safe, two-stage DES workflow within one outer split:

    1. **Split outer-train into pool-train and DSEL**
       The outer training fold ``(X_train, y_train)`` is stratified-split into:
       - a *pool-training* subset (used to tune/fit the pool pipeline), and
       - a *DSEL* subset (used to fit the DES competence model).

    2. **Tune + refit the pool pipeline (inner CV)**
       The pool pipeline (``pool_classifiers``) is tuned on the pool-training subset via a
       randomized CV search (delegated to :func:`run_randomized_search_cv`). The best
       configuration is refit on the full pool-training subset, producing a fitted pipeline
       that is then used to:
       - compute pool resubstitution metrics on the pool-training subset, and
       - provide the fitted preprocessing and fitted pool estimator required by DES.

    3. **Fit the DES model on DSEL**
       The fitted preprocessing portion of the tuned pool pipeline (all steps before the
       ``"resampling"`` step) is applied to DSEL via ``transform``. The fitted pool estimator
       is extracted from the tuned pipeline's ``"classifier"`` step and injected into the DES
       configuration under ``pool_classifiers``. The DES estimator is then fitted on the
       transformed DSEL data.

    4. **Build inference pipeline and evaluate on outer-test**
       A final inference pipeline is created as ``preprocessing -> DES`` (no resampling).
       Predictions are generated on ``X_test``; probabilities are used when available.
       Metrics are computed via :func:`compute_classification_metrics`, and a ``"score_time"``
       field is added to the test metrics.

    Parameters
    ----------
    des_model : sklearn.base.BaseEstimator
        Unfitted DES estimator (typically from DESlib) implementing ``fit`` and ``predict``.
        If the final pipeline exposes ``predict_proba``, probability-based metrics can be
        computed on the test set.
    des_conf : Dict[str, Any]
        Parameter dictionary forwarded to ``des_model.set_params(**...)``.
        This function does **not** mutate ``des_conf``. A local shallow copy is created
        internally to inject the fitted pool as ``pool_classifiers`` before calling
        ``set_params``.
    pool_classifiers : Union[imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline, sklearn.base.BaseEstimator]
        Pool pipeline (or estimator) to be tuned on the pool-training subset.
        In the current implementation, it is expected to be a pipeline containing:
        - a step named ``"resampling"`` (used only during training; excluded from inference),
        - a step named ``"classifier"`` (the fitted pool to inject into DES).
        All steps before ``"resampling"`` must support ``transform``.
    search_space : Dict[str, Any]
        Hyperparameter distributions or candidate lists used to tune ``pool_classifiers``.
        Keys must match valid parameter names of the pool pipeline using the double-underscore
        convention (e.g., ``"classifier__n_estimators"``, ``"feature_selection_filter__k"``).
    X_train : Union[pandas.DataFrame, numpy.ndarray]
        Features for the outer training fold, shape ``(n_train_samples, n_features)``.
    y_train : Union[pandas.Series, numpy.ndarray]
        Labels for the outer training fold, shape ``(n_train_samples,)``.
    X_test : Union[pandas.DataFrame, numpy.ndarray]
        Features for the outer test fold, shape ``(n_test_samples, n_features)``.
    y_test : Union[pandas.Series, numpy.ndarray]
        Labels for the outer test fold, shape ``(n_test_samples,)``.
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
        configure the underlying pool estimators with ``n_jobs=1`` when appropriate.

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline used for test evaluation. It contains:
        - the fitted preprocessing steps extracted from the tuned pool pipeline
          (all steps before ``"resampling"``), followed by
        - the fitted DES estimator as the final step named ``"classifier"``.
    tuning_results : Dict[str, Any]
        Tuning summary returned by :func:`run_randomized_search_cv` for the best candidate
        (e.g., CV mean/std scores, best params, tuning time).
    pool_resubstitution_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the pool-training subset using the tuned pool pipeline.
        This requires that the tuned pool pipeline implements ``predict_proba``.
    test_metrics : Dict[str, Optional[float] | int]
        Metrics computed on the outer test fold using the final DES inference pipeline, plus:
        - ``"score_time"`` : float
          Wall-clock time (seconds) measured around the test-time prediction calls in this
          function (``predict`` plus the attempted ``predict_proba``).

    Raises
    ------
    ValueError
        If ``dsel_size`` is not in ``(0, 1)``, if stratified splitting fails, or if tuning
        fails due to invalid configuration (e.g., incompatible ``scoring`` or invalid
        parameter names in ``search_space``).
    KeyError
        If the tuned pool pipeline does not contain required steps (notably ``"resampling"``
        and/or ``"classifier"``).
    AttributeError
        If the tuned pool pipeline does not implement ``predict_proba`` (required for pool
        resubstitution metrics in this implementation).
    Exception
        Any exception raised by the underlying estimators/pipeline during fitting,
        transformation, or prediction may propagate.

    Notes
    -----
    - **No outer-test leakage:** the outer test fold is used only for final evaluation.
      Pool tuning and DES fitting occur exclusively within the outer training fold.
    - **Train-time-only resampling:** any ``"resampling"`` step in the pool pipeline is
      excluded from the final inference pipeline.
    - **Binary classification convention:** when probabilities are available, the positive
      class probability is taken as ``predict_proba(X)[:, 1]``.
    - **Test probabilities are optional:** if ``final_des_pipeline.predict_proba`` is not
      available, ``y_test_pred_proba`` is set to ``None`` and probability-based metrics are
      returned as ``None`` by :func:`compute_classification_metrics`.

    Examples
    --------
    >>> final_pipe, tuning, pool_resub, test_metrics = train_and_evaluate_one_fold_des_model(
    ...     des_model=des_model,
    ...     des_conf=des_conf,
    ...     pool_classifiers=pool_pipeline,
    ...     search_space=pool_space,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     logger=logger,
    ...     n_iter=30,
    ...     dsel_size=0.2,
    ...     val_cv_split=5,
    ...     scoring="average_precision",
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )
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
    Train and evaluate all STATIC and DES models on a single outer CV fold.

    This helper encapsulates the full workflow for one outer split of a
    ``RepeatedStratifiedKFold`` / ``RepeatedStratifiedKFold``-style experiment.
    Given the global feature/label arrays and the current train/test indices, it:

    1. Builds the outer fold datasets:
       - ``(X_train, y_train) = (X[train_idx], y[train_idx])``
       - ``(X_test,  y_test)  = (X[test_idx],  y[test_idx])``
       and logs basic class-distribution statistics for both.

    2. For each **STATIC** model name in ``static_models``:
       - Retrieves the estimator and its *estimator-level* hyperparameter search space
         via :func:`get_static_model_and_search_space`.
       - Optionally injects a pipeline-level tuning dimension for SelectKBest by adding
         ``"feature_selection_filter__k": list(fs_k_best_candidates)`` to the search space
         when ``fs_k_best_candidates`` is provided.
       - Builds the end-to-end pipeline via :func:`build_model_pipeline`
         (preprocessing → SelectKBest → optional resampling → classifier).
       - Tunes/refits and evaluates the model via :func:`train_and_evaluate_one_fold_static_model`,
         producing:
           * the best fitted pipeline,
           * a tuning summary dict (e.g., best params / CV info),
           * resubstitution (train) metrics,
           * generalization (test) metrics.
       - Extracts the final selected feature indices/names via :func:`get_final_selected_features`.
       - Appends standardized report rows to ``resubstitution_rows`` and
         ``generalization_rows`` via :func:`collect_fold_reports`.

    3. For each **DES** model name in ``des_models``:
       - Retrieves:
           * the bagging/ensemble pool estimator,
           * the pool *estimator-level* search space,
           * the (unfitted) DESlib competence model,
           * a DES configuration dict
         via :func:`get_des_model`.
       - Optionally injects ``"feature_selection_filter__k"`` candidates into the pool
         search space (same logic as STATIC).
       - Builds the pool pipeline via :func:`build_model_pipeline`
         (preprocessing → SelectKBest → optional resampling → pool estimator).
       - Tunes the pool, fits the DES competence model on a DSEL split, and evaluates the
         final pipeline via :func:`train_and_evaluate_one_fold_des_model`, producing:
           * a fitted pipeline returned as ``best_des_model`` (as defined by the helper),
           * a tuning summary dict,
           * resubstitution metrics (for the tuned/fitted pool on the training side),
           * generalization metrics (for DES inference on the outer test set).
       - Extracts the final selected feature indices/names via :func:`get_final_selected_features`.
       - Appends standardized report rows via :func:`collect_fold_reports`.

    The function returns two lists of dict rows that are ready to be aggregated and
    persisted at experiment level.

    Parameters
    ----------
    run_id : int
        Global counter for the current outer split, typically coming from
        ``enumerate(cv_outer.split(X, y))``.
    iteration_idx : int
        0-based outer repetition index (0..n_repeats-1). Stored as 1-based in output rows via ``iteration_idx + 1``.
    fold_idx : int
        0-based fold index within the repetition (0..n_splits-1). Stored as 1-based in output rows via ``fold_idx + 1``.
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
        Column indices to be standardized by the preprocessing step.
    transformed_feature_names : Sequence[str]
        Feature names aligned with the input of the ``"feature_selection_filter"`` step
        (i.e., after the ``"preprocessor"`` transformation). Used to map selected feature
        indices back to names.
    static_models : Sequence[str]
        Names of static models to train (e.g., ``["SVC", "RandomForestClassifier"]``).
    des_models : Sequence[str]
        Names of DES models to train (e.g., ``["KNORAU", "METADES"]``).
    fs_k_best_to_keep : int | str
        Default ``k`` used when constructing the ``SelectKBest(k=...)`` step inside
        :func:`build_model_pipeline`. If the tuning search space includes
        ``"feature_selection_filter__k"``, the fitted value may differ from this default.
    fs_k_best_candidates : Sequence[int | str] | None
        Optional candidate values for ``SelectKBest.k`` to be explored during tuning.
        When provided, candidates are injected by adding::

            "feature_selection_filter__k": list(fs_k_best_candidates)

        into the corresponding hyperparameter search space.
    use_cost_sensitive_learning : bool
        Whether to enable cost-sensitive behaviour (e.g., class weights) in both STATIC
        models and DES pools. Forwarded to the model factory functions.
    resampling_method : str | None
        Canonical name of the resampling strategy (e.g., ``"SMOTE"``,
        ``"RandomUnderSampler"``, ``"SMOTEENN"``) or ``None`` to disable resampling.
    resampling_params : Dict[str, Any] | None
        Extra keyword arguments for the sampler constructor (if resampling is enabled).
    tuning_n_iter : int
        Number of parameter settings sampled in the randomized hyperparameter search.
    tuning_cv_inner_n_splits : int
        Number of stratified folds for the inner CV used during hyperparameter tuning.
    tuning_scoring : str
        Scoring metric used by the hyperparameter search (e.g., ``"f1"``,
        ``"average_precision"``).
    tuning_n_jobs : int
        Number of parallel jobs used during hyperparameter tuning.
    dsel_size : float
        Fraction of the outer training set reserved for the DSEL subset used to fit DES
        competence models (``0 < dsel_size < 1``).
    random_state : int
        Base random seed forwarded to models, inner CV splitters, and the DSEL split.
        The tuning calls use ``random_state + run_id`` to diversify the randomized search
        across outer folds.
    logger : Any
        Logger instance exposing ``.info(str)`` (e.g., Loguru logger). This argument is
        required and is used for fold-level diagnostics.

    Returns
    -------
    resubstitution_rows : List[Dict[str, Any]]
        Metrics rows on the **training** side of the current outer fold.
        - STATIC models: resubstitution metrics computed on ``(X_train, y_train)``.
        - DES models: resubstitution metrics computed for the tuned/fitted **pool**
          (i.e., the ensemble used by the DES model).
    generalization_rows : List[Dict[str, Any]]
        Metrics rows on the **test** side of the current outer fold.
        - STATIC models: test metrics computed on ``(X_test, y_test)``.
        - DES models: test metrics computed by the final DES inference pipeline on
          ``(X_test, y_test)``.

    Notes
    -----
    - Output rows use 1-based ``iteration`` and ``fold`` indices (``iteration_idx + 1``,
      ``fold_idx + 1``) to simplify downstream reporting.
    - Feature-selection tuning via ``"feature_selection_filter__k"`` assumes the pipeline
      step is named exactly ``"feature_selection_filter"``.
    - Tuning metadata (``tuning_results``) is passed to :func:`collect_fold_reports`; by
      design, it is typically stored on the resubstitution row (not necessarily on the
      test row), depending on the implementation of :func:`collect_fold_reports`.
    - For DES models, the meaning of ``best_des_model`` and the exact definition of
      resubstitution vs generalization metrics follow :func:`train_and_evaluate_one_fold_des_model`.

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
