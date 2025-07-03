import time
import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.modeling.metrics_utils import compute_classification_metrics


def train_and_evaluate_base_model(
    base_model: Union[Pipeline, BaseEstimator],
    search_space: dict,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple:
    """
    Train and evaluate a model or pipeline using RandomizedSearchCV.

    Parameters
    ----------
    base_model : Pipeline or BaseEstimator
        The model or pipeline to train and optimize.

    search_space : dict
        Hyperparameter search space.

    X_train : {array-like, pd.DataFrame}, shape (n_samples, n_features)
        Training features.

    y_train : {array-like, pd.Series}, shape (n_samples,)
        Training labels.

    X_test : {array-like, pd.DataFrame}, shape (n_samples, n_features)
        Test features.

    y_test : {array-like, pd.Series}, shape (n_samples,)
        Test labels.

    n_iter : int, optional
        Number of iterations for RandomizedSearchCV. Default is 50.

    cv : int, optional
        Number of cross-validation folds. Default is 5.

    scoring : str, optional
        Scoring metric for optimization. Default is 'f1'.

    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 (use all processors).

    Returns
    -------
    tuple
        - fitted_model : estimator
            The model or pipeline fitted on training data.
        - tuning_results : dict
            Dictionary with tuning summary including best parameters and scores.
        - resubstitution_metrics : dict
            Metrics on training data.
        - test_metrics : dict
            Metrics on test data.
    """

    cvs = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cvs,
        verbose=3,
        n_jobs=n_jobs,
        random_state=None,
        refit=True,
        return_train_score=True
    )

    start_tuning_time = time.time()
    search.fit(X_train, y_train)
    end_tuning_time = time.time()
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

    # Evaluate on the training set
    y_train_pred = best_model.predict(X_train)
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    resubstitution_metrics = compute_classification_metrics(y_train, y_train_pred, y_train_pred_prob)

    # Evaluate on the test set
    start_score_time = time.time()
    y_test_pred = best_model.predict(X_test)
    end_score_time = time.time()
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_prob)
    test_metrics["score_time"] = end_score_time - start_score_time

    return best_model, tuning_results, resubstitution_metrics, test_metrics


def train_and_evaluate_ensemble_model(
    ensemble_model: Union[Pipeline, BaseEstimator],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Tuple[BaseEstimator, dict]:
    """
    Train and evaluate an ensemble model (static or dynamic) using a fixed training set.

    Parameters
    ----------
    ensemble_model : Pipeline or BaseEstimator
        The ensemble model to train (e.g., VotingClassifier, DES model).

    X_train : {array-like, pd.DataFrame}, shape (n_samples, n_features)
        Training features (can be standard training for static ensemble or DSEL for DES models).

    y_train : {array-like, pd.Series}, shape (n_samples,)
        Training labels.

    X_test : {array-like, pd.DataFrame}, shape (n_samples, n_features)
        Test features.

    y_test : {array-like, pd.Series}, shape (n_samples,)
        Test labels.

    Returns
    -------
    tuple
        - fitted_model : estimator
            The ensemble model fitted on training data.
        - test_metrics : dict
            Dictionary with classification metrics on the test set.
    """
    # Fit ensemble on its corresponding data (train for static ensemble, or DSEL for dynamic ensemble)
    ensemble_model.fit(X_train, y_train)

    # Predict and evaluate on a test set
    start_score_time = time.time()
    y_test_pred = ensemble_model.predict(X_test)
    end_score_time = time.time()
    y_test_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_pred_proba)
    test_metrics["score_time"] = end_score_time - start_score_time

    return ensemble_model, test_metrics