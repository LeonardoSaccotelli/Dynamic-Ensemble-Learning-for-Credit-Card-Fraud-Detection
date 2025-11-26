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
    Tune a model (or pipeline) with RandomizedSearchCV on the training set,
    refit the best configuration, and report metrics on both train and test.

    The procedure uses stratified CV during hyperparameter search, then evaluates
    the selected estimator on:
      1) the **training** set (resubstitution error) and
      2) the **held-out test** set (generalization).

    Parameters
    ----------
    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or BaseEstimator
        Estimator/pipeline to optimize. Must implement ``fit`` and ``predict``. If you
        rely on probability-based metrics (e.g., ROC-AUC, AP), it should also implement
        ``predict_proba`` (or you must adapt the code to use ``decision_function``).
    search_space : dict
        Parameter distributions for ``RandomizedSearchCV``. Keys must match the estimator
        (or pipeline step) names, e.g., ``"classifier__C"``, ``"select__skb__k"``.
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
    tuning_results : dict
        Summary at the best index, including:
        - ``cv_tuning_mean_train_score``, ``cv_tuning_std_train_score``
        - ``cv_tuning_mean_val_score``, ``cv_tuning_std_val_score``
        - ``best_params`` (dict)
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
    - If your estimator lacks ``predict_proba``, adapt the evaluation to use decision
      scores or choose metrics compatible with hard labels only.

    Examples
    --------
    Optimize a pipeline and evaluate:

    >>> splitter_metric = "average_precision"
    >>> tuning, train_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
    ...     base_model=pipe,               # e.g., ImbPipeline([... ('classifier', clf)])
    ...     search_space=param_distributions,
    ...     X_train=X_tr, y_train=y_tr,
    ...     X_test=X_te, y_test=y_te,
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
