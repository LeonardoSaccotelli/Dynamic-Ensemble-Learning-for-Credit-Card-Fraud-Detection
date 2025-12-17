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
    Compute a comprehensive set of binary classification metrics.

    This helper aggregates:
    - raw confusion-matrix counts (TP/TN/FP/FN),
    - thresholded (label-based) metrics from scikit-learn,
    - imbalance-oriented metrics from imbalanced-learn, and
    - optional probability/ranking metrics (ROC-AUC, Average Precision) when predicted
      probabilities are provided.

    The function is designed to be robust to common edge cases:
    - If ``y_pred_proba`` is ``None``, probabilistic metrics are returned as ``None``.
    - If ``y_true`` contains only one class, ROC-AUC / Average Precision are undefined and
      returned as ``None`` (handled via ``ValueError``).

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (0 for negative class, 1 for positive class).
    y_pred : Sequence[int]
        Predicted binary labels (0 or 1), typically produced by ``estimator.predict``.
    y_pred_proba : Sequence[float] | numpy.ndarray | None, default=None
        Predicted probabilities or scores for the positive class. The function accepts:
        - shape ``(n_samples,)``: interpreted as positive-class probability/score,
        - shape ``(n_samples, 2)`` (or more columns): column 1 is interpreted as the
          positive-class probability (``[:, 1]``).

        If ``None``, probability-based metrics (ROC-AUC, Average Precision) are not computed
        and are returned as ``None``.

    Returns
    -------
    Dict[str, Optional[float] | int]
        Dictionary containing the following keys:

        Confusion-matrix counts
            - ``"tp"`` : int
            - ``"tn"`` : int
            - ``"fp"`` : int
            - ``"fn"`` : int

        Thresholded (label-based) metrics
            - ``"accuracy"`` : float
            - ``"precision"`` : float
            - ``"recall"`` : float
            - ``"f1"`` : float

        Imbalance / robustness metrics
            - ``"specificity"`` : float
            - ``"fpr"`` : float
            - ``"balanced_accuracy"`` : float
            - ``"geometric_mean"`` : float
            - ``"mcc"`` : float
            - ``"kappa"`` : float

        Probability / ranking metrics (optional)
            - ``"roc_auc"`` : float | None
            - ``"average_precision"`` : float | None

    Notes
    -----
    - Confusion-matrix counts are computed with ``labels=[0, 1]`` to ensure a 2×2 shape,
      even when one class is absent from predictions.
    - ``specificity`` is computed via :func:`imblearn.metrics.specificity_score`.
    - ``geometric_mean`` is computed via :func:`imblearn.metrics.geometric_mean_score`.
    - ``fpr`` is computed as ``1.0 - specificity``.
    - ``precision``, ``recall`` and ``f1`` use ``zero_division=0`` to avoid exceptions in
      degenerate cases (e.g., no positive predictions).
    - ``roc_auc`` and ``average_precision`` require both classes to be present in
      ``y_true``; when undefined they are returned as ``None``.

    Examples
    --------
    >>> y_true = [0, 0, 0, 1, 1]
    >>> y_pred = [0, 0, 1, 1, 0]
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    (1, 1, 1, 2)
    >>> metrics["roc_auc"] is None and metrics["average_precision"] is None
    True

    >>> import numpy as np
    >>> y_true = np.array([0, 0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 1, 0])
    >>> y_pred_proba = np.array([0.10, 0.35, 0.60, 0.80, 0.40])
    >>> metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)
    >>> metrics["roc_auc"] is not None and metrics["average_precision"] is not None
    True

    >>> y_pred_proba_2d = np.array([
    ...     [0.90, 0.10],
    ...     [0.65, 0.35],
    ...     [0.40, 0.60],
    ...     [0.20, 0.80],
    ...     [0.60, 0.40],
    ... ])
    >>> metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba_2d)
    >>> metrics["roc_auc"] is not None and metrics["average_precision"] is not None
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
    Append one metrics row (dict) to ``store`` for a single CV fold/run.

    This helper builds a standardized row containing the common identifiers
    (experiment name, iteration, fold, model, and split label) and merges them with
    the provided metric values. Additional user-defined fields can be attached via
    ``**extra`` (e.g., tuning summaries, selected features, timings).

    The function appends the constructed row to ``store`` in-place.

    Parameters
    ----------
    store : List[Dict[str, Any]]
        Mutable list that will receive the constructed row (side effect: append).
    experiment_name : str
        Experiment identifier written into the row (e.g., ``"baseline-v1"``).
    iteration : int
        1-based outer repetition index.
    fold : int
        1-based fold index within the current iteration.
    model : str
        Model (or pipeline) identifier written into the row.
    metrics : Mapping[str, Optional[float] | int]
        Mapping of scalar metric values to merge into the row
        (e.g., ``{"accuracy": 0.97, "f1": 0.83, "tp": 10}``).
    data_split : str
        Split label written into the row under the ``"split"`` key
        (e.g., ``"resubstitution"``, ``"test"``, ``"val"``).
    **extra : Any
        Optional additional key-value pairs to merge into the row
        (e.g., ``best_params=...``, ``tuning_time=...``, ``selected_features_names=...``).

    Returns
    -------
    None
        The function operates by side effect (appends to ``store``).

    Raises
    ------
    ValueError
        If any key in ``metrics`` or ``extra`` collides with reserved identifiers:
        ``{"experiment_name", "iteration", "fold", "model", "split"}``.

    Notes
    -----
    - Keys from ``metrics`` and ``extra`` are merged into the final row. To prevent
      accidental overwrites, collisions with the reserved identifiers should be
      validated by the caller or enforced by an explicit check in this function.

    Examples
    --------
    >>> rows = []
    >>> collect_single_report_one_fold(
    ...     rows,
    ...     experiment_name="baseline-v1",
    ...     iteration=0,
    ...     fold=3,
    ...     model="RandomForest",
    ...     metrics={"f1": 0.812, "roc_auc": 0.972},
    ...     data_split="test",
    ...     best_params={"n_estimators": 200},
    ...     tuning_time=12.4,
    ... )
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
    Append standardized per-fold reporting rows for resubstitution and test splits.

    This helper writes two report rows (one for the training/resubstitution split and
    one for the held-out test/generalization split) into the provided row buffers.
    It delegates the actual row construction to :func:`collect_single_report_one_fold`
    and ensures consistent metadata and feature-selection fields across both splits.

    If ``tuning_results`` is provided, its key/value pairs are appended only to the
    **resubstitution** row (i.e., the row where ``data_split="resubstitution"``),
    so that inner-CV tuning summaries are stored once per fold/model.

    Parameters
    ----------
    resubstitution_rows : List[Dict[str, Any]]
        Mutable list acting as an output buffer that will receive one row for the
        resubstitution split.
    generalization_rows : List[Dict[str, Any]]
        Mutable list acting as an output buffer that will receive one row for the
        generalization (outer test) split.
    experiment_name : str
        Experiment identifier stored in both rows.
    iteration : int
        1-based (recommended) outer repetition index stored in both rows.
    fold : int
        1-based (recommended) outer fold index stored in both rows.
    model_name : str
        Model identifier stored in both rows (e.g., ``"SVC"``, ``"METADES"``).
    resubstitution_metrics : Mapping[str, Optional[float] | int]
        Metrics computed on the training split. Forwarded as ``metrics=...`` when writing
        the resubstitution row.
    test_metrics : Mapping[str, Optional[float] | int]
        Metrics computed on the outer test split. Forwarded as ``metrics=...`` when writing
        the generalization row.
    fold_size_train : int
        Number of samples in the training split. Stored in the resubstitution row as
        ``fold_size``.
    fold_size_test : int
        Number of samples in the test split. Stored in the generalization row as
        ``fold_size``.
    selected_features_indices : Sequence[int]
        Indices of selected features for this fold (post-preprocessing / feature selection).
        Stored in both rows.
    selected_features_names : Sequence[str]
        Names of selected features for this fold, aligned with ``selected_features_indices``.
        Stored in both rows.
    tuning_results : Dict[str, Any] | None, default=None
        Optional tuning summary (e.g., best params, inner-CV mean/std, tuning time).
        If provided, it is expanded into keyword arguments and written **only** into the
        resubstitution row.

    Returns
    -------
    None
        Mutates ``resubstitution_rows`` and ``generalization_rows`` in place.

    Raises
    ------
    TypeError
        If ``tuning_results`` contains keys that conflict with explicit keyword arguments
        passed to :func:`collect_single_report_one_fold`, or if the downstream reporter
        cannot handle the provided values.

    Notes
    -----
    - ``tuning_results`` is attached only to the resubstitution row to avoid duplicating
      tuning metadata in the test row.
    - Both rows always include the selected feature indices/names to support downstream
      feature-selection frequency and stability analyses.
    - The exact final schema of each row is determined by :func:`collect_single_report_one_fold`
      (reserved key policy, merge behaviour, etc.).

    Examples
    --------
    >>> resubstitution_rows, generalization_rows = [], []
    >>> collect_fold_reports(
    ...     resubstitution_rows=resubstitution_rows,
    ...     generalization_rows=generalization_rows,
    ...     experiment_name="COST_SENSITIVE_NO_RESAMPLING",
    ...     iteration=0,
    ...     fold=3,
    ...     model_name="SVC",
    ...     resubstitution_metrics={"f1": 0.95, "roc_auc": 0.99},
    ...     test_metrics={"f1": 0.82, "roc_auc": 0.93, "score_time": 0.12},
    ...     fold_size_train=18000,
    ...     fold_size_test=2000,
    ...     selected_features_indices=[0, 2, 5],
    ...     selected_features_names=["V1", "V3", "Amount_log1p"],
    ...     tuning_results={"best_params": {"classifier__C": 1.0}, "tuning_time": 12.4},
    ... )
    >>> len(resubstitution_rows), len(generalization_rows)
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
