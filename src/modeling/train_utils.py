from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, cohen_kappa_score, confusion_matrix)

def train_and_evaluate_base_model(
    base_model: Union[Pipeline, BaseEstimator],
    search_space: dict,
    X_train,
    y_train,
    X_test,
    y_test,
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

    X_train : array-like
        Training features.

    y_train : array-like
        Training labels.

    X_test : array-like
        Test features.

    y_test : array-like
        Test labels.

    n_iter : int, optional
        Number of iterations for RandomizedSearchCV. Default is 50.

    cv : int, optional
        Number of cross-validation folds. Default is 5.

    scoring : str, optional
        Scoring metric for optimization. Default is 'f1'.

    random_state : int, optional
        Random state for reproducibility. Default is 42.

    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1.

    Returns
    -------
    tuple
        (fitted_model: The model or pipeline fitted on training data,
        tuning_results: dict with tuning results - best_params with mean and std deviation of cross-validation score,
        resubstitution_metrics: dict with metrics on training data,
        test_metrics: dict with metrics on test data)
    """

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Retrieve best search info
    tuning_results = {
        "mean_val_score": search.cv_results_["mean_test_score"][search.best_index_],
        "std_val_score": search.cv_results_["std_test_score"][search.best_index_],
        "best_params": search.best_params_
    }

    # Evaluate on the training set
    y_train_pred = best_model.predict(X_train)
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    resubstitution_metrics = compute_classification_metrics(y_train, y_train_pred, y_train_prob)

    # Evaluate on the test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob)

    return best_model, tuning_results, resubstitution_metrics, test_metrics


def compute_classification_metrics(y_true, y_pred, y_proba):
    """
    Compute a set of classification metrics based on true and predicted labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.

    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.

    y_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary with classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
