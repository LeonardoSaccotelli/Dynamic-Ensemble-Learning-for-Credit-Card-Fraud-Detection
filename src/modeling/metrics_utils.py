from typing import List, Tuple, Union, Dict, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score, \
    average_precision_score, cohen_kappa_score
from sklearn.pipeline import Pipeline


def compute_classification_metrics(y_true, y_pred, y_pred_proba):
    """
    Compute a set of classification metrics based on true and predicted labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.

    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.

    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary with classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tnr) if (fp + tnr) > 0 else 0.0

    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "average_precision": average_precision_score(y_true, y_pred_proba),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "balanced_accuracy": (recall + tnr) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tnr": tnr,
        "fpr": fpr,
        "tpr": tpr
    }


def compute_voting_weights_from_dsel(
    pool: List[Tuple[str, Union[Pipeline, BaseEstimator]]],
    X_dsel: np.ndarray,
    y_dsel: np.ndarray,
    metric: str = "f1",
    log_scores: bool = False,
) -> List[float]:
    """
    Compute voting weights for pre-fitted classifiers using DSEL performance.

    Parameters
    ----------
    pool : list of (str, BaseEstimator or Pipeline)
        List of (name, fitted classifier) tuples.

    X_dsel : np.ndarray
        Features from DSEL set.

    y_dsel : np.ndarray
        Labels from DSEL set.

    metric : str, default="f1"
        Metric to use for scoring ("f1", "roc_auc", "accuracy", etc.).

    log_scores : bool, default=False
        If True, print model scores to stdout.

    Returns
    -------
    list of float
        Normalized voting weights based on the selected metric.
    """
    weights = []

    for name, model in pool:
        y_pred = model.predict(X_dsel)
        y_pred_proba = model.predict_proba(X_dsel)[:, 1]
        metrics = compute_classification_metrics(y_dsel, y_pred, y_pred_proba)
        score = metrics.get(metric, 0.0)

        weights.append(score)
        if log_scores:
            print(f"[DSEL Score] {name}: {metric} = {score:.4f}")

    total = sum(weights)
    if total > 0:
        return [w / total for w in weights]
    else:
        # Fallback: equal weights
        return [1.0 / len(pool)] * len(pool)



def append_metrics(
    store: List[Dict[str, Any]],
    *,
    iteration: int,
    fold: int,
    pool_name: str,
    model: str,
    metrics: Dict[str, float],
    data_split: str,
    **extra
) -> None:
    """
    Push one metrics row into `store` without repeating boilerplate.

    Parameters
    ----------
    store        : list receiving the row
    iteration    : 0-based outer repetition index
    fold         : 0-based fold index
    pool_name    : the name given to the pool of classifiers
    model        : model name
    metrics      : {'accuracy': …, 'f1': …, …}
    data_split   : 'resub' | 'test' | 'dsel' … (whatever label you prefer)
    extra        : any extra key-values (e.g., tuning results,
                   selected_features, fold_size …)
    """
    row = {
        "iteration": iteration,
        "fold": fold,
        "pool_name": pool_name,
        "model": model,
        "split": data_split,             # identifies resubstitution vs. test at a glance
        **metrics,
        **extra
    }
    store.append(row)