from __future__ import annotations

from typing import Any, Dict, List, Sequence

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
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of binary classification metrics.

    This function aggregates raw confusion-matrix counts, standard metrics
    from scikit-learn, and imbalance-oriented metrics from imbalanced-learn.
    Probabilistic metrics (ROC-AUC, Average Precision) are computed only if
    predicted probabilities are provided.

    Parameters
    ----------
    y_true : sequence of int
        True binary labels (0 for negative, 1 for positive).
    y_pred : sequence of int
        Predicted binary labels (0 or 1).
    y_pred_proba : array-like of shape (n_samples,) or (n_samples, 2), optional
        Predicted probabilities for the positive class. If a 2D array is
        passed, the second column (``[:, 1]``) is assumed to correspond to
        the positive class. If ``None``, probabilistic metrics are returned
        as ``None``.

    Returns
    -------
    dict
        Dictionary with the following entries:

        Raw counts
            - ``tp`` : int
            - ``tn`` : int
            - ``fp`` : int
            - ``fn`` : int

        Standard metrics
            - ``accuracy`` : float
            - ``precision`` : float
            - ``recall`` : float
            - ``f1`` : float

        Imbalance-aware / robust metrics
            - ``specificity`` : float
            - ``fpr`` : float
            - ``balanced_accuracy`` : float
            - ``geometric_mean`` : float
            - ``mcc`` : float
            - ``kappa`` : float

        Probabilistic / ranking metrics
            - ``roc_auc`` : float or None
            - ``average_precision`` : float or None

        For degenerate cases where only one class is present in ``y_true``,
        some metrics (e.g., ROC-AUC, Average Precision) may be undefined and
        are returned as ``None``.

    Notes
    -----
    - ``specificity`` is computed via ``imblearn.metrics.specificity_score``.
    - ``geometric_mean`` is ``sqrt(sensitivity * specificity)`` computed
      via ``imblearn.metrics.geometric_mean_score``.
    - ``mcc`` (Matthews correlation coefficient) and ``kappa`` (Cohen’s kappa)
      are often more informative than plain accuracy on imbalanced data.
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
    metrics: Dict[str, float],
    data_split: str,
    **extra: Any,
) -> None:
    """
    Append one metrics row (dict) to ``store`` for a single CV fold/run.

    This helper centralizes the common fields (experiment/run identifiers and
    split label) and merges them with the provided metric values. Extra
    user-defined key–values can be included via ``**extra`` (e.g., tuning
    summaries, selected features).

    Parameters
    ----------
    store : list of dict
        Mutable list that will receive the constructed row (in-place append).
    experiment_name : str
        Name/identifier of the experiment (e.g., ``"baseline-v1"``).
    iteration : int
        1-based outer repetition index.
    fold : int
        1-based fold index within the current iteration.
    model : str
        Model (or pipeline) name.
    metrics : dict[str, float]
        Dictionary of scalar metrics (e.g., ``{"accuracy": 0.97, "f1": 0.83}``).
    data_split : str
        Split label (e.g., ``"resub"``, ``"test"``, ``"val"``).
    **extra : Any
        Optional additional fields to include (e.g., ``best_params=...``,
        ``tuning_time=...``, ``selected_features=...``).

    Returns
    -------
    None
        Operates by side effect (appends to ``store``).

    Raises
    ------
    ValueError
        If any key in ``metrics`` or ``extra`` collides with reserved fields:
        {``"experiment_name"``, ``"iteration"``, ``"fold"``, ``"model"``, ``"split"``}.

    Notes
    -----
    - Keys from ``metrics`` and ``extra`` are merged into the final row. To
      prevent accidental overwrites, collisions with reserved identifiers are
      disallowed.

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
    resubstitution_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
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
    It delegates the actual row construction to ``collect_single_report_one_fold`` and
    ensures consistent metadata and feature-selection fields across both splits.

    If ``tuning_results`` is provided, its key/value pairs are appended only to the
    **resubstitution** row (i.e., the row where ``data_split="resubstitution"``),
    enabling a single place to store inner-CV tuning summaries alongside the
    training-set metrics.

    Parameters
    ----------
    resubstitution_rows : List[Dict[str, Any]]
        Mutable list acting as an output buffer that will receive a single row
        dictionary for the resubstitution split.

    generalization_rows : List[Dict[str, Any]]
        Mutable list acting as an output buffer that will receive a single row
        dictionary for the test/generalization split.

    experiment_name : str
        Experiment identifier (e.g., a label encoding resampling policy, cost-sensitive
        learning setting, or any other experimental condition). Stored in both rows.

    iteration : int
        Outer repetition index (e.g., in a repeated cross-validation scheme). Stored
        in both rows.

    fold : int
        Outer fold index. Stored in both rows.

    model_name : str
        Model identifier used for reporting (e.g., ``"SVC"``, ``"RandomForestClassifier"``,
        ``"METADES"``). Stored in both rows.

    resubstitution_metrics : Dict[str, Any]
        Metrics dictionary computed on the training split (resubstitution).
        Passed through to ``collect_single_report_one_fold`` as ``metrics=...`` for the
        resubstitution row.

    test_metrics : Dict[str, Any]
        Metrics dictionary computed on the held-out test split (generalization).
        Passed through to ``collect_single_report_one_fold`` as ``metrics=...`` for the
        test row.

    fold_size_train : int
        Number of samples in the training split for this outer fold. Stored in the
        resubstitution row as ``fold_size``.

    fold_size_test : int
        Number of samples in the test split for this outer fold. Stored in the test
        row as ``fold_size``.

    selected_features_indices : Sequence[int]
        Indices of the selected features for this fold (after preprocessing / feature
        selection). Stored in both rows to enable later feature-selection frequency
        analysis.

    selected_features_names : Sequence[str]
        Names of the selected features for this fold (aligned with
        ``selected_features_indices``). Stored in both rows.

    tuning_results : Dict[str, Any] | None, default=None
        Optional tuning summary dictionary (e.g., from RandomizedSearchCV or similar).
        If provided, its items are expanded into keyword arguments and forwarded only
        to the resubstitution row writer. If ``None``, no tuning fields are added.

        Typical keys include (but are not limited to): ``best_params``, CV mean/std
        scores, and tuning wall-clock time.

    Returns
    -------
    None
        This function mutates ``resubstitution_rows`` and ``generalization_rows`` in place.

    Raises
    ------
    TypeError
        If ``tuning_results`` contains keys that conflict with explicit keyword
        arguments of ``collect_single_report_one_fold`` (e.g., overlapping names such as
        ``experiment_name``), or if the values cannot be serialized/handled by the
        downstream reporter.

    Notes
    -----
    - ``tuning_results`` is applied only to the resubstitution row to avoid duplicating
      tuning metadata in the test row.
    - Both rows always include feature-selection fields (indices and names) so that
      downstream analyses can align performance metrics with selected feature subsets.
    - The exact schema of each row is determined by ``collect_single_report_one_fold``.

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
