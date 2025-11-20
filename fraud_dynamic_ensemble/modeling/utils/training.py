import time
from typing import Any, Dict, Tuple, Union

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """
    Fit a model/pipeline with RandomizedSearchCV and report tuning, train, and test metrics.

    The function performs hyperparameter optimization with stratified CV on ``X_train, y_train``,
    refits the best configuration on the full training set, and then computes metrics on both
    the training (resubstitution) and test sets.

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline | sklearn.pipeline.Pipeline | BaseEstimator
        The estimator or pipeline to optimize (must support ``fit`` and ``predict`` and, for
        probability-based metrics, ``predict_proba``).
    search_space : dict
        Parameter distributions for ``RandomizedSearchCV`` (keys should match step names,
        e.g. ``"classifier__C"``).
    X_train, X_test : array-like of shape (n_samples, n_features)
        Training and test features.
    y_train, y_test : array-like of shape (n_samples,)
        Training and test labels.
    n_iter : int, default=50
        Number of RandomizedSearchCV iterations.
    val_cv_split : int, default=5
        Number of stratified folds for validation during tuning.
    scoring : str, default="f1"
        Optimization metric passed to ``RandomizedSearchCV``.
    random_state : int, default=42
        Random seed for the CV splitter and (passed) search randomness.
    n_jobs : int, default=-1
        Parallel jobs for the hyperparameter search.

    Returns
    -------
    tuning_results : dict
        Summary of tuning, including mean/std train/val scores for the best index, best params,
        and total tuning time (seconds).
    resubstitution_metrics : dict[str, float]
        Metrics computed on the training set using the best model.
    test_metrics : dict[str, float]
        Metrics computed on the test set using the best model; includes ``'score_time'`` (seconds).

    Notes
    -----
    - Assumes the best estimator exposes ``predict_proba``. If not, adapt the metric function
      or convert decision scores accordingly.
    - Uses ``StratifiedKFold`` with shuffling for stable class proportions across folds.
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
        random_state=None,
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

    return tuning_results, resubstitution_metrics, test_metrics
