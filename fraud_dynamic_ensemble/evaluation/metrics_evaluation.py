from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from imblearn.metrics import (
    geometric_mean_score,
    specificity_score,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_pred_proba: Sequence[float] | np.ndarray | None = None,
) -> Dict[str, Optional[float] | int]:
    """
    Compute a standard set of binary classification metrics.

    This helper aggregates confusion-matrix counts (TP/TN/FP/FN), threshold-based
    metrics (accuracy, precision, recall, F1), imbalance-oriented metrics
    (specificity, balanced accuracy, geometric mean, MCC, Cohen's kappa), and
    optional ranking/probability metrics (ROC-AUC, average precision) when
    predicted probabilities/scores are provided. Non-computable probability
    metrics (e.g., single-class ``y_true``) are returned as ``None``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 for negative class, 1 for positive class).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1), typically produced by ``estimator.predict``.
    y_pred_proba : array-like of shape (n_samples,) or (n_samples, n_classes), optional
        Predicted probability or score for the positive class. If a 2D array is
        provided, column 1 (``[:, 1]``) is interpreted as the positive-class
        probability/score when available. If ``None``, ROC-AUC and average
        precision are not computed and are returned as ``None``.

    Returns
    -------
    metrics : dict
        Dictionary of metrics with keys:
        - ``'tp'``, ``'tn'``, ``'fp'``, ``'fn'`` : int
        - ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'`` : float
        - ``'specificity'``, ``'fpr'``, ``'balanced_accuracy'``, ``'geometric_mean'`` : float
        - ``'mcc'``, ``'kappa'`` : float
        - ``'roc_auc'``, ``'average_precision'`` : float or None

    Notes
    -----
    - Confusion-matrix counts are computed with ``labels=[0, 1]`` to enforce a 2×2
      shape even when one class is absent in predictions.
    - ``precision``, ``recall``, and ``f1`` use ``zero_division=0`` to handle
      degenerate cases (e.g., no positive predictions) without raising.
    - ``fpr`` is computed as ``1.0 - specificity``.
    - ROC-AUC and average precision require both classes to be present in ``y_true``.
      When undefined, they are returned as ``None``.

    Examples
    --------
    >>> y_true = [0, 0, 0, 1, 1]
    >>> y_pred = [0, 0, 1, 1, 0]
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
    (1, 1, 1, 2)

    >>> import numpy as np
    >>> y_pred_proba = np.array([0.10, 0.35, 0.60, 0.80, 0.40])
    >>> metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)
    >>> metrics["roc_auc"] is not None
    True
    """

    # Convert to numpy arrays for safety
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # 1. Confusion Matrix (forced labels to ensure 2x2 shape)
    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    # 2. Imbalanced-learn optimized metrics
    #    Specificity: TN / (TN + FP)
    spec = specificity_score(y_true_arr, y_pred_arr, average="binary")

    #    Geometric mean: sqrt(Sensitivity * Specificity)
    g_mean = geometric_mean_score(y_true_arr, y_pred_arr, average="binary")

    # 3. Probabilistic metrics
    roc_auc = None
    avg_precision = None
    if y_pred_proba is not None:
        proba_arr = np.asarray(y_pred_proba)

        # Accept both (n_samples,) and (n_samples, 2) formats
        if proba_arr.ndim == 2:
            if proba_arr.shape[1] < 2:
                # Fallback: treat as (n_samples,)
                proba_pos = proba_arr.ravel()
            else:
                # Assume positive class is column 1
                proba_pos = proba_arr[:, 1]
        else:
            proba_pos = proba_arr.ravel()

        try:
            roc_auc = roc_auc_score(y_true_arr, proba_pos)
            avg_precision = average_precision_score(y_true_arr, proba_pos)
        except ValueError:
            # Edge cases where y_true has only one class → metrics undefined
            roc_auc = None
            avg_precision = None

    return {
        # --- Raw counts ---
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # --- Standard classification metrics ---
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        # Sensitivity / True Positive Rate
        "recall": recall_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        # --- Imbalance & robustness ---
        "specificity": spec,
        "fpr": 1.0 - spec,
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "geometric_mean": g_mean,
        "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
        "kappa": cohen_kappa_score(y_true_arr, y_pred_arr),
        # --- Probabilistic / ranking ---
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }


def collect_single_report_one_fold(
    store: List[Dict[str, Any]],
    *,
    experiment_name: str,
    iteration: int,
    fold: int,
    model: str,
    metrics: Mapping[str, Optional[float] | int],
    data_split: str,
    **extra: Any,
) -> None:
    """
    Append one standardized metrics row for a single CV fold.

    This helper builds a single dictionary row with common identifiers
    (experiment name, iteration, fold, model, and split label) and merges them
    with scalar metrics plus any additional user-provided fields. The constructed
    row is appended to ``store`` in-place.

    Parameters
    ----------
    store : list of dict
        Mutable list that receives the constructed row (side effect: append).
    experiment_name : str
        Experiment identifier written into the row.
    iteration : int
        Outer repetition index for the current run.
    fold : int
        Fold index within the current iteration.
    model : str
        Model (or pipeline) identifier written into the row.
    metrics : Mapping[str, float or int or None]
        Mapping of metric names to scalar values (e.g., ``{"f1": 0.81, "tp": 10}``).
    data_split : str
        Split label written into the row under the ``"split"`` key (e.g.,
        ``"resubstitution"``, ``"test"``).
    **extra : Any
        Additional key-value pairs to merge into the row (e.g., hyperparameters,
        timings, selected feature metadata).

    Returns
    -------
    None
        This function operates by side effect.

    Notes
    -----
    The row is created via ``{**metrics, **extra}`` merging. To prevent accidental
    overwrites of the identifier fields (``experiment_name``, ``iteration``, ``fold``,
    ``model``, ``split``), consider validating that ``metrics`` and ``extra`` do not
    contain these reserved keys.

    Examples
    --------
    >>> rows = []
    >>> collect_single_report_one_fold(
    ...     rows,
    ...     experiment_name="baseline-v1",
    ...     iteration=1,
    ...     fold=3,
    ...     model="RandomForest",
    ...     metrics={"f1": 0.812, "roc_auc": 0.972},
    ...     data_split="test",
    ...     best_params={"n_estimators": 200},
    ...     tuning_time=12.4,
    ... )
    >>> len(rows)
    1
    """

    row = {
        "experiment_name": experiment_name,
        "iteration": iteration,
        "fold": fold,
        "model": model,
        "split": data_split,
        **metrics,
        **extra,
    }
    store.append(row)


def collect_fold_reports(
    *,
    resubstitution_rows: List[Dict[str, Any]],
    generalization_rows: List[Dict[str, Any]],
    experiment_name: str,
    iteration: int,
    fold: int,
    model_name: str,
    resubstitution_metrics: Mapping[str, Optional[float] | int],
    test_metrics: Mapping[str, Optional[float] | int],
    fold_size_train: int,
    fold_size_test: int,
    selected_features_indices: Sequence[int],
    selected_features_names: Sequence[str],
    tuning_results: Dict[str, Any] | None = None,
) -> None:
    """
    Append per-fold report rows for resubstitution and generalization splits.

    This helper writes two standardized rows into the provided output buffers:
    one row for metrics computed on the training (resubstitution) split and one
    row for metrics computed on the held-out test (generalization) split. Row
    construction is delegated to :func:`collect_single_report_one_fold`, and
    feature-selection metadata is recorded consistently for both splits. If
    provided, ``tuning_results`` is attached only to the resubstitution row to
    avoid duplicating tuning metadata.

    Parameters
    ----------
    resubstitution_rows : list of dict
        Output buffer that receives the resubstitution row.
    generalization_rows : list of dict
        Output buffer that receives the generalization row.
    experiment_name : str
        Experiment identifier stored in both rows.
    iteration : int
        Outer repetition index stored in both rows.
    fold : int
        Outer fold index stored in both rows.
    model_name : str
        Model identifier stored in both rows.
    resubstitution_metrics : Mapping[str, float or int or None]
        Metrics computed on the training split.
    test_metrics : Mapping[str, float or int or None]
        Metrics computed on the outer test split.
    fold_size_train : int
        Number of samples in the training split; stored as ``fold_size`` in the
        resubstitution row.
    fold_size_test : int
        Number of samples in the test split; stored as ``fold_size`` in the
        generalization row.
    selected_features_indices : Sequence[int]
        Indices of selected features for this fold. Stored in both rows.
    selected_features_names : Sequence[str]
        Names of selected features for this fold, aligned with
        ``selected_features_indices``. Stored in both rows.
    tuning_results : dict or None, optional
        Optional tuning summary to attach to the resubstitution row (e.g., best
        parameters, inner-CV scores, tuning time). If ``None``, no tuning fields
        are added.

    Returns
    -------
    None
        This function operates by side effect (appends to both output buffers).

    Notes
    -----
    - ``tuning_results`` is attached only to the resubstitution row to avoid
      duplicating tuning metadata in the generalization row.
    - Both rows always include selected feature indices and names to support
      downstream feature-selection stability analyses.
    - The final row schema and merge behavior are governed by
      :func:`collect_single_report_one_fold`.

    Examples
    --------
    >>> resub_rows, gen_rows = [], []
    >>> collect_fold_reports(
    ...     resubstitution_rows=resub_rows,
    ...     generalization_rows=gen_rows,
    ...     experiment_name="COST_SENSITIVE_NO_RESAMPLING",
    ...     iteration=1,
    ...     fold=3,
    ...     model_name="SVC",
    ...     resubstitution_metrics={"f1": 0.95, "roc_auc": 0.99},
    ...     test_metrics={"f1": 0.82, "roc_auc": 0.93},
    ...     fold_size_train=18_000,
    ...     fold_size_test=2_000,
    ...     selected_features_indices=[0, 2, 5],
    ...     selected_features_names=["V1", "V3", "Amount_log1p"],
    ...     tuning_results={"best_params": {"classifier__C": 1.0}, "tuning_time": 12.4},
    ... )
    >>> (len(resub_rows), len(gen_rows))
    (1, 1)
    """

    tuning_kwargs = tuning_results or {}

    collect_single_report_one_fold(
        resubstitution_rows,
        experiment_name=experiment_name,
        iteration=iteration,
        fold=fold,
        model=model_name,
        metrics=resubstitution_metrics,
        data_split="resubstitution",
        fold_size=fold_size_train,
        **tuning_kwargs,
        selected_features_indices=selected_features_indices,
        selected_features_names=selected_features_names,
    )

    collect_single_report_one_fold(
        generalization_rows,
        experiment_name=experiment_name,
        iteration=iteration,
        fold=fold,
        model=model_name,
        metrics=test_metrics,
        data_split="generalization",
        fold_size=fold_size_test,
        selected_features_indices=selected_features_indices,
        selected_features_names=selected_features_names,
    )
