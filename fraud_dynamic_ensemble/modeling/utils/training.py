from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Union

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from fraud_dynamic_ensemble.evaluation.metrics_evaluation import compute_classification_metrics


def train_and_evaluate_one_fold_static_model(
    base_model: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_iter: int = 50,
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
    Tune a model (or pipeline) with RandomizedSearchCV on the training set,
    refit the best configuration, and report metrics on both train and test.

    The procedure uses stratified CV during hyperparameter search, then evaluates
    the selected estimator on:
      1) the **training** set (resubstitution error), and
      2) the **held-out test** set (generalization).

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        Estimator/pipeline to optimize. Must implement ``fit`` and ``predict``. If you
        rely on probability-based metrics (e.g., ROC-AUC, AP), it should also implement
        ``predict_proba`` (or you must adapt the code to use ``decision_function``).
    search_space : dict
        Parameter distributions for ``RandomizedSearchCV``. Keys must match the estimator
        (or pipeline step) names, e.g. ``"classifier__C"``, ``"select__skb__k"``.
    X_train, X_test : array-like of shape (n_samples, n_features)
        Training and test features.
    y_train, y_test : array-like of shape (n_samples,)
        Training and test labels.
    n_iter : int, default=50
        Number of parameter settings sampled by ``RandomizedSearchCV``.
    val_cv_split : int, default=5
        Number of stratified folds used during the hyperparameter search.
    scoring : str, default="f1"
        Optimization metric passed to ``RandomizedSearchCV`` (e.g., ``"f1"``,
        ``"average_precision"``, ``"roc_auc"``).
    random_state : int, default=42
        Random seed for the ``StratifiedKFold`` splitter (shuffling enabled).
    n_jobs : int, default=-1
        Number of parallel jobs for the search (``-1`` uses all available cores).

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        The refit estimator corresponding to the best hyperparameter setting.
    tuning_results : dict
        Summary at the best index, including:
        - ``cv_tuning_mean_train_score``, ``cv_tuning_std_train_score``
        - ``cv_tuning_mean_val_score``, ``cv_tuning_std_val_score``
        - ``best_params`` (dict of the best hyperparameters)
        - ``tuning_time`` (seconds, float)
    resubstitution_metrics : dict[str, float]
        Metrics on the training set computed via ``compute_classification_metrics``.
    test_metrics : dict[str, float]
        Metrics on the test set computed via ``compute_classification_metrics``, plus
        ``"score_time"`` (seconds to generate test predictions).

    Notes
    -----
    - The CV splitter is ``StratifiedKFold(shuffle=True, random_state=random_state)`` to
      preserve class proportions across folds.
    - Ensure hyperparameter names align with your pipeline step names (scikit-learn
      double-underscore convention).
    - For full reproducibility of the *search* itself, consider setting
      ``random_state`` on ``RandomizedSearchCV`` as well.

    Examples
    --------
    Optimize a pipeline and evaluate:

    >>> splitter_metric = "average_precision"
    >>> best_model, tuning, resubstitution_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
    ...     base_model=base_model,            # e.g., ImbPipeline([... ('classifier', clf)])
    ...     search_space=search_space,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     n_iter=30,
    ...     val_cv_split=5,
    ...     scoring=splitter_metric,
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )
    """

    splitter = StratifiedKFold(n_splits=val_cv_split, random_state=random_state, shuffle=True)

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

    # Evaluate on the training set (resubstitution error)
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    resubstitution_metrics = compute_classification_metrics(
        y_train, y_train_pred, y_train_pred_prob
    )

    # Evaluate on the test set (generalization error)
    start_score_time = time.time()
    y_test_pred = best_model.predict(X_test)
    end_score_time = time.time()
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_prob)
    test_metrics["score_time"] = end_score_time - start_score_time

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
    n_iter: int = 50,
    dsel_size: float = 0.2,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, float | int]]:
    """
    Train a **Dynamic Ensemble Selection (DES)** model on one outer fold and
    evaluate it on the held-out test set.

    Workflow
    --------
    1) Split the provided outer training set into:
       - **pool-training** subset: to tune/fit the base ensemble
         (``pool_classifiers``) with ``RandomizedSearchCV``.
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
    pool_classifiers : imblearn.pipeline.Pipeline | sklearn.pipeline.Pipeline | BaseEstimator
        Base pool to be tuned (typically a pipeline with steps named:
        ``"preprocessor"``, ``"feature_selection_filter"``, ``"feature_selection_embedded"``,
        ``"resampling"``, ``"classifier"``). The tuned ``"classifier"`` step becomes the pool.
        The ``"resampling"`` step is used **only** during pool training and is **dropped**
        from the final inference pipeline.
    search_space : dict
        Hyperparameter distributions for ``RandomizedSearchCV``. Keys must match the pool's
        parameter naming (e.g., ``"classifier__n_estimators"``, ``"preprocessor__scaler__with_mean"``).
    X_train : pandas.DataFrame or numpy.ndarray
        Features of the **outer training** fold (split internally into pool-training and DSEL).
    y_train : pandas.Series or numpy.ndarray
        Labels of the outer training fold.
    X_test : pandas.DataFrame or numpy.ndarray
        Features of the **outer test** fold (never used in tuning or DSEL fitting).
    y_test : pandas.Series or numpy.ndarray
        Labels of the outer test fold.
    n_iter : int, default=50
        Number of parameter settings sampled by ``RandomizedSearchCV``.
    dsel_size : float, default=0.2
        Proportion of ``X_train`` reserved for DSEL (``0 < dsel_size < 1``).
    val_cv_split : int, default=5
        Number of stratified folds for the inner hyperparameter search.
    scoring : str, default="f1"
        Scoring metric passed to ``RandomizedSearchCV`` (e.g., ``"f1"``, ``"roc_auc"``,
        ``"average_precision"``).
    random_state : int, default=42
        Random seed used in the DSEL split and the inner CV splitter.
    n_jobs : int, default=-1
        Parallel jobs for ``RandomizedSearchCV`` (``-1`` uses all cores).

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline:
        ``[('preprocessor', ...), ('feature_selection_filter', ...),
          ('feature_selection_embedded', ...), ('classifier', DES)]``.
    test_metrics : dict[str, float | int]
        Metrics computed on the test set by ``compute_classification_metrics``
        (e.g., ``accuracy``, ``f1``, ``roc_auc``, ``average_precision``,
        ``tp``, ``tn``, ``fp``, ``fn``), plus ``'score_time'`` (seconds).

    Notes
    -----
    - **No leakage:** The outer test set is never used for tuning nor for DES fitting.
    - **Resampling at train-time only:** The pool's ``'resampling'`` step (if present)
      is not part of the final inference pipeline.
    - **Step assumptions:** This function expects the pool pipeline to have the named
      steps listed above and slices the first three steps as preprocessing
      (``best_pipe_pool_classifiers[:3]``). Adjust if your layout differs.
    - **Probabilities:** If ``predict_proba`` is unavailable, the code falls back to
      ``None`` for probabilities; ensure your ``compute_classification_metrics`` supports
      that or adapt it to ``decision_function``-based metrics.

    Examples
    --------
    >>> final_pipe, test_metrics = train_and_evaluate_one_fold_des_model(
    ...     des_model=des_model,
    ...     des_conf=des_conf,               # will be updated with the tuned pool
    ...     pool_classifiers=pool_classifiers,    # your training pipeline
    ...     search_space=search_space,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test,  y_test=y_test,
    ...     n_iter=30, dsel_size=0.2,
    ...     val_cv_split=5, scoring="average_precision",
    ...     random_state=42, n_jobs=-1,
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

    search.fit(X_train_pool, y_train_pool)
    best_pipe_pool_classifiers = search.best_estimator_

    # Extract the preprocessing pipeline (fitted)
    # We skip resampling step since it is required just at training time
    fitted_preproc = best_pipe_pool_classifiers[:3]

    # Apply the preprocessing steps on the dsel dataset
    X_dsel_trans = fitted_preproc.transform(X_dsel)

    # Extract the fitted pool of classifiers
    fitted_pool = best_pipe_pool_classifiers.named_steps["classifier"]

    # Add the trained pool of classifiers to the des model config
    des_conf["pool_classifiers"] = fitted_pool

    # Fit DES model on DSEL in transformed space
    des_model.set_params(**des_conf)
    des_model.fit(X_dsel_trans, y_dsel)

    # Final inference pipeline: preprocessing -> DES
    final_des_pipeline = Pipeline(fitted_preproc.steps + [("classifier", des_model)])

    # Evaluation on test set
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

    return final_des_pipeline, test_metrics
