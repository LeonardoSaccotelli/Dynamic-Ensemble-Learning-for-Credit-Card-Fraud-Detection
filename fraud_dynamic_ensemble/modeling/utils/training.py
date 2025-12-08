from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple, Union
import warnings

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    HalvingRandomSearchCV,
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
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
    factor: int = 3,
    min_resources: int | str = "smallest",
    max_resources: int | str = "auto",
    aggressive_elimination: bool = False,
    resource: str = "n_samples",
    n_candidates: int | str = "exhaust",
) -> tuple[
    Union[ImbPipeline, Pipeline, BaseEstimator],
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float],
]:
    """
    Tune a model (or pipeline) with ``HalvingRandomSearchCV`` on the training set,
    refit the best configuration, and report metrics on both train and test.

    The procedure uses a stratified CV splitter during the hyperparameter search,
    then evaluates the selected estimator on:
      1) the **training** set (resubstitution error), and
      2) the **held-out test** set (generalization).

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        Estimator/pipeline to optimize. Must implement ``fit`` and ``predict``. If you
        rely on probability-based metrics (e.g., ROC-AUC, Average Precision), it should
        also implement ``predict_proba`` (or you must adapt the code to use
        ``decision_function``).
    search_space : dict
        Parameter distributions for ``HalvingRandomSearchCV``. Keys must match the
        estimator (or pipeline step) names, e.g. ``"classifier__C"``,
        ``"classifier__max_depth"``.
    X_train, X_test : array-like of shape (n_samples, n_features)
        Training and test features.
    y_train, y_test : array-like of shape (n_samples,)
        Training and test labels.
    val_cv_split : int, default=5
        Number of stratified folds used during the hyperparameter search
        (inner CV for model selection).
    scoring : str, default="f1"
        Optimization metric passed to ``HalvingRandomSearchCV`` (e.g., ``"f1"``,
        ``"average_precision"``, ``"roc_auc"``).
    random_state : int, default=42
        Random seed for both the ``StratifiedKFold`` splitter (with shuffling) and
        the ``HalvingRandomSearchCV`` search process.
    n_jobs : int, default=-1
        Number of parallel jobs for the halving search (``-1`` uses all available
        cores). The underlying estimators should typically use ``n_jobs=1`` to
        avoid nested parallelism.
    factor : int, default=3
        The halving parameter. At each iteration, only a fraction ``1 / factor``
        of the candidates is selected to continue to the next round with
        increased resources (e.g., ``factor=3`` keeps the best third).
    min_resources : {"exhaust", "smallest"} or int, default="smallest"
        Minimum amount of resource (e.g. number of samples) that any candidate
        is allowed to use at the first iteration. Equivalently, this defines the
        resources ``r0`` allocated per candidate in the first round. If
        ``min_resources="exhaust"``, enough resources are used so that only a
        single iteration is run (subject to ``max_resources`` and ``factor``).
    max_resources : int or "auto", default="auto"
        Maximum amount of resources that any candidate is allowed to use. When
        ``resource="n_samples"`` (default) and ``max_resources="auto"``, this is
        set to ``n_samples`` (the size of the training set). If ``resource`` is
        set to another estimator parameter (e.g. ``"n_estimators"``), then
        ``max_resources`` must be an integer and **not** ``"auto"``.
    aggressive_elimination : bool, default=False
        Controls how aggressively candidates are eliminated when there are not
        enough resources to reduce the pool to at most ``factor`` candidates at
        the last iteration. If ``True``, the search may replay earlier
        iterations to further prune the pool; if ``False``, the last iteration
        may evaluate more than ``factor`` candidates.
    resource : {"n_samples"} or str, default="n_samples"
        Name of the resource that increases with each iteration. By default this
        is ``"n_samples"`` (number of training samples). It can also be set to
        any integer-valued parameter of the base estimator (e.g. ``"n_estimators"``,
        ``"n_iterations"``). In that case, ``max_resources`` cannot be ``"auto"``
        and must be specified as an integer.
    n_candidates : int or "exhaust", default="exhaust"
        Number of candidate parameter settings to sample at the first iteration.
        If set to ``"exhaust"``, enough candidates are sampled so that the last
        iteration uses as many resources as possible, given
        ``min_resources``, ``max_resources`` and ``factor``. In this case,
        ``min_resources`` cannot be ``"exhaust"``.

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        The refit estimator corresponding to the best hyperparameter configuration
        found by ``HalvingRandomSearchCV``.
    tuning_results : dict
        Summary of the tuning at the best index, including:
        - ``cv_tuning_mean_train_score`` : float
        - ``cv_tuning_std_train_score`` : float
        - ``cv_tuning_mean_val_score`` : float
        - ``cv_tuning_std_val_score`` : float
        - ``best_params`` : dict of the best hyperparameters
        - ``tuning_time`` : float, total search time in seconds
    resubstitution_metrics : dict[str, float]
        Metrics on the training set computed via ``compute_classification_metrics``.
        Includes confusion-matrix counts and derived metrics (e.g., accuracy, f1,
        balanced accuracy, ROC-AUC, Average Precision, kappa, MCC, etc.).
    test_metrics : dict[str, float]
        Metrics on the test set computed via ``compute_classification_metrics``,
        plus:
        - ``"score_time"`` : float, seconds required to generate test predictions.

    Notes
    -----
    - The CV splitter used inside ``HalvingRandomSearchCV`` is
      ``StratifiedKFold(n_splits=val_cv_split, shuffle=True, random_state=random_state)``
      to preserve class proportions across folds.
    - Hyperparameter names in ``search_space`` must align with your pipeline
      step names and scikit-learn’s double-underscore convention.
    - ``HalvingRandomSearchCV`` performs a **successive halving** strategy:
      it starts with a larger set of random configurations evaluated on a
      subset of the data (``min_resources``) and keeps only the top fraction
      (controlled by ``factor``) while increasing the resources up to
      ``max_resources``.
    - For strict reproducibility, the combination of ``random_state`` for both
      the splitter and the halving search ensures deterministic behaviour given
      the same data and configuration.

    Examples
    --------
    Optimize a pipeline and evaluate:

    >>> splitter_metric = "average_precision"
    >>> best_model, tuning, resubstitution_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
    ...     base_model=base_model,            # e.g., ImbPipeline([... ('classifier', clf)])
    ...     search_space=search_space,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,   y_test=y_test,
    ...     val_cv_split=5,
    ...     scoring=splitter_metric,
    ...     random_state=42,
    ...     n_jobs=-1,
    ...     factor=3,
    ...     min_resources="smallest",
    ...     max_resources="auto",
    ...     aggressive_elimination=False,
    ...     resource="n_samples",
    ...     n_candidates="exhaust",
    ... )
    """
    splitter = StratifiedKFold(
        n_splits=val_cv_split,
        random_state=random_state,
        shuffle=True,
    )

    print(
        f"[HALVING SEARCH SETTINGS]:"
        f"\nscoring: {scoring}"
        f"\nrandom_state: {random_state}"
        f"\nn_jobs: {n_jobs}"
    )

    search = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        factor=factor,
        min_resources=min_resources,
        max_resources=max_resources,
        aggressive_elimination=aggressive_elimination,
        resource=resource,
        n_candidates=n_candidates,
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
    print(f"[HALVING SEARCH BEST PARAMS]: {tuning_results['best_params']}")

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
    dsel_size: float = 0.2,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
    factor: int = 3,
    min_resources: int | str = "smallest",
    max_resources: int | str = "auto",
    aggressive_elimination: bool = False,
    resource: str = "n_samples",
    n_candidates: int | str = "exhaust",
) -> Tuple[Pipeline, Dict[str, float | int]]:
    """
    Train a **Dynamic Ensemble Selection (DES)** model on one outer fold and
    evaluate it on the held-out test set.

    Workflow
    --------
    1) Split the provided outer training set into:
       - **pool-training** subset: to tune/fit the base ensemble
         (``pool_classifiers``) with ``HalvingRandomSearchCV``.
       - **DSEL** subset: to fit the DES competence model.
    2) Tune and refit ``pool_classifiers`` on the pool-training subset.
    3) From the best pool pipeline:
       - Extract the **fitted preprocessing** (e.g., preprocessor + feature selection),
         **excluding** the ``"resampling"`` and ``"classifier"`` steps.
       - Extract the tuned **classifier** step to be used as the **pool**.
    4) Transform the DSEL data with the fitted preprocessing and **fit the DES model**
       on ``(X_dsel_trans, y_dsel)`` with the injected pool.
    5) Build the **final inference pipeline** = ``preprocessing -> DES``.
    6) Evaluate on ``(X_test, y_test)`` via ``compute_classification_metrics``.

    Parameters
    ----------
    des_model : sklearn.base.BaseEstimator
        Unfitted DESlib estimator **instance** (e.g., ``KNORAE``, ``OLA``, ``METADES``).
        Must accept ``pool_classifiers=...`` in ``set_params`` and implement
        ``fit`` / ``predict`` (``predict_proba`` optional).
    des_conf : dict
        Hyperparameters for ``des_model`` (e.g., ``k``, ``DFP``, ``IH_rate``, ``voting``,
        ``n_jobs``). **Do not** include ``pool_classifiers`` here; it is injected internally.
        This dict is updated in-place before fitting.
    pool_classifiers : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        Base pool to be tuned (typically a pipeline with steps named:
        ``"preprocessor"``, ``"feature_selection_filter"``, ``"feature_selection_embedded"``,
        ``"resampling"``, ``"classifier"``). The tuned ``"classifier"`` step becomes the pool.
        The ``"resampling"`` step is used **only** during pool training and is **dropped**
        from the final inference pipeline.
    search_space : dict
        Hyperparameter distributions for ``HalvingRandomSearchCV``. Keys must match the pool's
        parameter naming (e.g., ``"classifier__n_estimators"``,
        ``"preprocessor__scaler__with_mean"``).
    X_train : pandas.DataFrame or numpy.ndarray
        Features of the **outer training** fold (split internally into pool-training and DSEL).
    y_train : pandas.Series or numpy.ndarray
        Labels of the outer training fold.
    X_test : pandas.DataFrame or numpy.ndarray
        Features of the **outer test** fold (never used in tuning or DSEL fitting).
    y_test : pandas.Series or numpy.ndarray
        Labels of the outer test fold.
    dsel_size : float, default=0.2
        Proportion of ``X_train`` reserved for DSEL (``0 < dsel_size < 1``).
    val_cv_split : int, default=5
        Number of stratified folds for the inner hyperparameter search.
    scoring : str, default="f1"
        Scoring metric passed to ``HalvingRandomSearchCV`` (e.g., ``"f1"``, ``"roc_auc"``,
        ``"average_precision"``).
    random_state : int, default=42
        Random seed used in the DSEL split, the inner CV splitter, and the halving search.
    n_jobs : int, default=-1
        Parallel jobs for ``HalvingRandomSearchCV`` (``-1`` uses all cores). The underlying
        estimators should typically use ``n_jobs=1`` to avoid nested parallelism.
    factor : int, default=3
        The halving parameter. At each iteration, only a fraction ``1 / factor`` of the
        candidates is selected to continue to the next round with increased resources
        (e.g., ``factor=3`` keeps the best third).
    min_resources : {"exhaust", "smallest"} or int, default="smallest"
        Minimum amount of resource (e.g. number of samples) that any candidate is allowed
        to use at the first iteration. Equivalently, this defines the resources ``r0``
        allocated per candidate in the first round.
    max_resources : int or "auto", default="auto"
        Maximum amount of resources that any candidate is allowed to use. When
        ``resource="n_samples"`` (default) and ``max_resources="auto"``, this is set to
        ``n_samples`` (the size of the training set). If ``resource`` is set to another
        estimator parameter (e.g. ``"n_estimators"``), then ``max_resources`` must be an
        integer and **not** ``"auto"``.
    aggressive_elimination : bool, default=False
        Controls how aggressively candidates are eliminated when there are not enough
        resources to reduce the pool to at most ``factor`` candidates at the last
        iteration. If ``True``, the search may replay earlier iterations to further
        prune the pool; if ``False``, the last iteration may evaluate more than
        ``factor`` candidates.
    resource : {"n_samples"} or str, default="n_samples"
        Name of the resource that increases with each iteration. By default this is
        ``"n_samples"`` (number of training samples). It can also be set to any
        integer-valued parameter of the base estimator (e.g. ``"n_estimators"``,
        ``"n_iterations"``). In that case, ``max_resources`` cannot be ``"auto"`` and
        must be specified as an integer.
    n_candidates : int or "exhaust", default="exhaust"
        Number of candidate parameter settings to sample at the first iteration.
        If set to ``"exhaust"``, enough candidates are sampled so that the last
        iteration uses as many resources as possible, given ``min_resources``,
        ``max_resources`` and ``factor``. In this case, ``min_resources`` cannot be
        ``"exhaust"``.

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline:
        ``[('preprocessor', ...), ('feature_selection_filter', ...),
          ('feature_selection_embedded', ...), ('classifier', DES)]``.
    test_metrics : dict[str, float | int]
        Metrics computed on the test set by ``compute_classification_metrics``
        (e.g., ``accuracy``, ``f1``, ``roc_auc``, ``average_precision``,
        ``tp``, ``tn``, ``fp``, ``fn``), plus ``"score_time"`` (seconds).

    Notes
    -----
    - **No leakage:** The outer test set is never used for tuning nor for DES fitting.
    - **Resampling at train-time only:** The pool's ``"resampling"`` step (if present)
      is not part of the final inference pipeline.
    - **Step assumptions:** This function expects the pool pipeline to have the named
      steps listed above and slices the first three steps as preprocessing
      (``best_pipe_pool_classifiers[:3]``). Adjust if your layout differs.
    - **Probabilities:** If ``predict_proba`` is unavailable, the code falls back to
      ``None`` for probabilities; ensure your ``compute_classification_metrics`` supports
      that or adapt it to ``decision_function``-based metrics.
    - ``HalvingRandomSearchCV`` performs a **successive halving** strategy for tuning
      the pool: it starts with a larger set of random configurations evaluated on a
      subset of the data (``min_resources``) and keeps only the top fraction
      (controlled by ``factor``) while increasing the resources up to ``max_resources``.

    Examples
    --------
    >>> final_pipe, test_metrics = train_and_evaluate_one_fold_des_model(
    ...     des_model=des_model,
    ...     des_conf=des_conf,                  # will be updated with the tuned pool
    ...     pool_classifiers=pool_classifiers,  # your training pipeline
    ...     search_space=search_space,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,   y_test=y_test,
    ...     dsel_size=0.2,
    ...     val_cv_split=5,
    ...     scoring="average_precision",
    ...     random_state=42,
    ...     n_jobs=-1,
    ...     factor=3,
    ...     min_resources="smallest",
    ...     max_resources="auto",
    ...     aggressive_elimination=False,
    ...     resource="n_samples",
    ...     n_candidates="exhaust",
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
        f"[HALVING SEARCH SETTINGS]:"
        f"\nscoring: {scoring}"
        f"\nrandom_state: {random_state}"
        f"\nn_jobs: {n_jobs}"
    )

    search = HalvingRandomSearchCV(
        estimator=pool_classifiers,
        param_distributions=search_space,
        factor=factor,
        min_resources=min_resources,
        max_resources=max_resources,
        aggressive_elimination=aggressive_elimination,
        resource=resource,
        n_candidates=n_candidates,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        cv=splitter,
        verbose=3,
        random_state=random_state,
        return_train_score=True,
    )

    search.fit(X_train_pool, y_train_pool)
    best_pipe_pool_classifiers = search.best_estimator_
    print(f"[HALVING SEARCH BEST PARAMS]: {search.best_params_}")

    # Extract the preprocessing pipeline (fitted)
    # We skip resampling step since it is required just at training time
    fitted_preproc = best_pipe_pool_classifiers[:3]

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

    return final_des_pipeline, test_metrics


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
    use_cost_sensitive_learning: bool,
    resampling_method: str | None,
    resampling_params: Dict[str, Any] | None,
    tuning_n_candidates: int,
    tuning_factor: int | float,
    tuning_min_resources: int | str,
    tuning_max_resources: int | str,
    tuning_aggressive_elimination: bool,
    tuning_cv_inner_n_splits: int,
    tuning_scoring: str,
    tuning_n_jobs: int,
    dsel_size: float,
    random_state: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train and evaluate all STATIC and DES models on a single outer CV fold.

    This helper encapsulates the full workflow for one outer split of a
    ``RepeatedStratifiedKFold`` experiment. Given the global feature/label
    arrays and the current train/test indices, it:

    1. Splits ``X`` and ``y`` into an outer training and test set.
    2. For each **STATIC** model in ``static_models``:
       - builds the pipeline (preprocessing → feature selection → resampling → classifier),
       - tunes it with :func:`train_and_evaluate_one_fold_static_model`
         using ``HalvingRandomSearchCV``,
       - extracts final selected features via :func:`get_final_selected_features`,
       - collects both resubstitution (train) and generalization (test) metrics.
    3. For each **DES** model in ``des_models``:
       - builds and tunes a bagging pool via :func:`train_and_evaluate_one_fold_des_model`,
       - fits the DES model on a DSEL subset of the outer training set,
       - collects generalization (test) metrics and selected features.
    4. Returns two lists of rows (dicts) ready to be aggregated and saved
       at experiment level.

    Parameters
    ----------
    run_id : int
        Global counter from ``enumerate(cv_outer.split(X, y))`` identifying the
        current outer split.
    iteration_idx : int
        0-based index of the outer repetition within ``RepeatedStratifiedKFold``.
    fold_idx : int
        0-based index of the fold within the current repetition.
    train_idx : numpy.ndarray
        Indices of samples used as training data for this outer fold.
    test_idx : numpy.ndarray
        Indices of samples used as test data for this outer fold.
    X : numpy.ndarray of shape (n_samples, n_features)
        Full feature matrix (already shuffled upstream).
    y : numpy.ndarray of shape (n_samples,)
        Full target vector (already shuffled upstream).
    experiment_name : str
        Experiment identifier stored in every metrics row.
    idx_num_features_to_standardize : sequence of int
        Column indices to be standardized by the preprocessing step.
    transformed_feature_names : sequence of str
        Feature names **after** the preprocessor, aligned with the input
        of the first feature-selection step. Used to map selected indices
        back to human-readable names.
    static_models : sequence of str
        List of static model names to be trained
        (e.g. ``["SVC", "RandomForestClassifier"]``).
    des_models : sequence of str
        List of DES model names to be trained
        (e.g. ``["KNORAE", "METADES"]``).
    fs_k_best_to_keep : int or {"all"}
        Number of features retained by ``SelectKBest``. Use ``"all"`` to
        keep all features.
    use_cost_sensitive_learning : bool
        Whether to enable cost-sensitive / imbalance-aware behaviour in both
        STATIC models and DES pools. This flag is forwarded to:

        - :func:`get_static_model_and_search_space` for each static model, and
        - :func:`get_des_model` for each DES pool.

        When ``True``, models that support it are configured with imbalance-aware
        defaults (e.g. ``class_weight='balanced'``, tuned ``scale_pos_weight`` in
        XGBoost, imbalanced-learn samplers/ensembles). When ``False``, those
        mechanisms are disabled where possible.
    resampling_method : str or None
        Canonical name of the resampling strategy (e.g. ``"SMOTE"``,
        ``"RandomUnderSampler"``, ``"SMOTEENN"``) or ``None``/``"none"``
        to disable resampling.
    resampling_params : dict or None
        Extra keyword arguments for the sampler constructor
        (e.g. ``sampling_strategy``, ``random_state``, ``k_neighbors``).
        If ``None``, no extra kwargs are passed.
    tuning_n_candidates : int
        Value passed as ``n_candidates`` to ``HalvingRandomSearchCV`` for
        both STATIC and DES pool tuning.
    tuning_factor : int or float
        ``factor`` parameter for ``HalvingRandomSearchCV`` (successive
        halving ratio).
    tuning_min_resources : int or {"smallest", "exhaust"}
        ``min_resources`` parameter for ``HalvingRandomSearchCV`` defining
        the initial resource level per candidate.
    tuning_max_resources : int or {"auto"}
        ``max_resources`` parameter for ``HalvingRandomSearchCV`` defining
        the maximum resource level per candidate.
    tuning_aggressive_elimination : bool
        ``aggressive_elimination`` parameter for ``HalvingRandomSearchCV``,
        controlling how aggressively candidates are discarded when the
        resource budget is tight.
    tuning_cv_inner_n_splits : int
        Number of stratified folds for the inner model-selection CV.
    tuning_scoring : str
        Scoring metric passed to ``HalvingRandomSearchCV``
        (e.g. ``"f1"``, ``"roc_auc"``, ``"average_precision"``).
    tuning_n_jobs : int
        Number of parallel jobs for the halving search (``-1`` uses all
        available cores).
    dsel_size : float
        Fraction of the outer training set reserved for the DSEL subset
        used to fit DES models (``0 < dsel_size < 1``).
    random_state : int
        Random seed forwarded to models, inner CV splitters and DSEL split
        within this fold.

    Returns
    -------
    resubstitution_rows : list of dict
        Metrics rows for all STATIC models on the **training** set
        (resubstitution error) of the current outer fold. Each row is
        compatible with :func:`collect_report_one_fold` output.
    generalization_rows : list of dict
        Metrics rows for all STATIC and DES models on the **test** set
        (generalization error) of the current outer fold.

    Notes
    -----
    - The ``iteration`` and ``fold`` fields written in the output rows are
      1-based (``iteration_idx + 1``, ``fold_idx + 1``) for readability in
      downstream reports.
    - This function is **config-free**: all required settings are passed as
      parameters, which makes it suitable for potential parallel execution
      of outer folds.

    Examples
    --------
    Typical usage inside the main outer CV loop::

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
                des_models=["KNORAE"],
                fs_k_best_to_keep=20,
                use_cost_sensitive_learning=True,
                resampling_method="SMOTE",
                resampling_params={"sampling_strategy": 0.2, "random_state": 42},
                tuning_n_candidates=35,
                tuning_factor=4,
                tuning_min_resources=1500,
                tuning_max_resources="auto",
                tuning_aggressive_elimination=False,
                tuning_cv_inner_n_splits=5,
                tuning_scoring="average_precision",
                tuning_n_jobs=-1,
                dsel_size=0.2,
                random_state=42,
            )

            resubstitution_metrics_summary.extend(res_rows)
            generalization_metrics_summary.extend(gen_rows)
    """

    resubstitution_rows: List[Dict[str, Any]] = []
    generalization_rows: List[Dict[str, Any]] = []

    # Split the data into training set (9 training folds) and test set (1 test fold)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Report class balance statistics for iteration
    for name, target_arr in zip(["train dataset", "test dataset"], [y_train, y_test]):
        unique, frequency = np.unique(target_arr, return_counts=True)
        print(f"Class distribution ({name} statistics) [class, frequency]: {unique, frequency}")

    # ----- Start training STATIC MODELS -----
    for static_model_name in static_models:
        print("-" * 165)
        print(f"Training STATIC model: {static_model_name}")

        # Get the static model estimator and with its hyperparameter search space
        static_model_estimator, static_model_search_space = get_static_model_and_search_space(
            static_model_name,
            random_state=random_state,
            use_cost_sensitive_learning=use_cost_sensitive_learning,
        )

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
            n_candidates=tuning_n_candidates,
            factor=tuning_factor,
            min_resources=tuning_min_resources,
            max_resources=tuning_max_resources,
            aggressive_elimination=tuning_aggressive_elimination,
            val_cv_split=tuning_cv_inner_n_splits,
            scoring=tuning_scoring,
            random_state=random_state,
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
        print(f"Training DES model: {des_model_name}")

        # Get the des model estimator and its configuration, with the
        # pool of classifiers and its hyperparameter search space
        pool_classifiers, pool_search_space, des_model_estimator, des_model_conf = get_des_model(
            des_model_name,
            random_state=random_state,
            use_cost_sensitive_learning=use_cost_sensitive_learning,
        )

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling + Classifier
        pool_classifiers_pipeline = build_model_pipeline(
            estimator=pool_classifiers,
            numerical_features_to_standardize=idx_num_features_to_standardize,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=resampling_method,
            resampling_params=resampling_params,
        )

        # Tune the des model, fit on the training folds and evaluate on the test fold
        best_des_model, test_metrics = train_and_evaluate_one_fold_des_model(
            des_model=des_model_estimator,
            des_conf=des_model_conf,
            pool_classifiers=pool_classifiers_pipeline,
            search_space=pool_search_space,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_candidates=tuning_n_candidates,
            factor=tuning_factor,
            min_resources=tuning_min_resources,
            max_resources=tuning_max_resources,
            aggressive_elimination=tuning_aggressive_elimination,
            dsel_size=dsel_size,
            val_cv_split=tuning_cv_inner_n_splits,
            scoring=tuning_scoring,
            random_state=random_state,
            n_jobs=tuning_n_jobs,
        )

        # Extract selected feature indices and names
        selected_indices, selected_names = get_final_selected_features(
            pipeline=best_des_model,
            feature_names=transformed_feature_names,
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

    print(f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]")

    return resubstitution_rows, generalization_rows
