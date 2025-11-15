from __future__ import annotations

from typing import Optional, Union

from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def get_feature_selection(
    *,
    k: Union[int, str] = 20,
    score: str = "f_classif",
    combine: str = "parallel",
    C: float = 0.1,
    solver: str = "liblinear",
    class_weight: Optional[str] = "balanced",
    max_iter: int = 2000,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Union[FeatureUnion, Pipeline]:
    """
    Build a feature-selection transform that uses **two selectors**:
    (1) a filter method (``SelectKBest(score=...)``) and
    (2) an embedded method (``SelectFromModel(LogisticRegression(penalty='l1', ...)``).

    You can combine them either:
      - in **parallel** with a ``FeatureUnion`` (concatenates features kept by *either* branch), or
      - in **sequence** with a ``Pipeline`` (the L1 selector runs on the output of SelectKBest).

    Parameters
    ----------
    k : int or {"all"}, default=20
        Number of top features to keep in the filter step. Use ``"all"`` to pass all features
        through the filter branch.
    score : {"f_classif", "mutual_info_classif", "chi2"}, default="f_classif"
        Scoring function for ``SelectKBest``:
        - ``"f_classif"`` → ANOVA F-test (fast, linear association).
        - ``"mutual_info_classif"`` → Mutual information (non-linear, slower).
        - ``"chi2"`` → Chi-square (**requires non-negative features**).
    combine : {"parallel", "sequential"}, default="parallel"
        - ``"parallel"``: return a ``FeatureUnion([('skb', ...), ('l1', ...)])``.
        - ``"sequential"``: return a ``Pipeline([('skb', ...), ('l1', ...)])``.
    C : float, default=0.1
        Inverse regularization strength for the L1-penalized Logistic Regression inside
        ``SelectFromModel`` (smaller → stronger sparsity).
    solver : {"liblinear", "saga"}, default="liblinear"
        Logistic Regression solver supporting ``penalty='l1'``.
    class_weight : {"balanced", None}, optional, default="balanced"
        Class weighting for Logistic Regression (use ``"balanced"`` for strong imbalance).
    max_iter : int, default=2000
        Maximum iterations for Logistic Regression.
    n_jobs : int or None, optional, default=None
        Parallelism for ``FeatureUnion`` (ignored for the sequential ``Pipeline``).
    random_state : int or None, optional, default=None
        Random seed for Logistic Regression (useful with ``solver='saga'``).

    Returns
    -------
    sklearn.pipeline.FeatureUnion or sklearn.pipeline.Pipeline
        - ``FeatureUnion`` when ``combine='parallel'`` (concatenates both branches).
        - ``Pipeline`` when ``combine='sequential'`` (L1 selector runs after SelectKBest).

    Notes
    -----
    - **Scaling**: Put a ``StandardScaler`` **before** this selector to stabilize L1 behavior.
      Do **not** standardize if using ``score='chi2'``; instead normalize to non-negative
      (e.g., ``MinMaxScaler``).
    - **Duplicates (parallel only)**: Union concatenates outputs; overlapping choices from both
      branches will duplicate columns (usually harmless; you can deduplicate later).
    - **Imbalance**: If you resample (SMOTE/undersampling), do so **inside** CV and **before**
      selection to avoid leakage.

    Examples
    --------
    Parallel union (default):
    >>> selector = get_feature_selection(k=20, score="f_classif", combine="parallel")

    Sequential composition:
    >>> selector = get_feature_selection(k=30, score="mutual_info_classif", combine="sequential")

    Chi-square branch (ensure non-negative inputs):
    >>> selector = get_feature_selection(k=25, score="chi2", combine="parallel")

    Grid-search knobs (if the step is named ``select``):
    - ``select__skb__k``
    - ``select__l1__estimator__C``, ``select__l1__estimator__class_weight``
    - ``select__l1__estimator__solver``  (must be one of ``{'liblinear','saga'}``)
    """
    # Map score string to callable
    if score == "chi2":
        score_func = chi2
    elif score == "f_classif":
        score_func = f_classif
    elif score == "mutual_info_classif":
        score_func = mutual_info_classif
    else:
        raise ValueError(
            f"Unsupported score '{score}'. "
            "Choose from {'chi2', 'f_classif', 'mutual_info_classif'}."
        )

    # Validate solver for L1
    if solver not in {"liblinear", "saga"}:
        raise ValueError("Solver must be one of {'liblinear', 'saga'} for L1 penalty.")

    skb = SelectKBest(score_func=score_func, k=k)

    l1_est = LogisticRegression(
        penalty="l1",
        solver=solver,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    l1_sfm = SelectFromModel(estimator=l1_est)

    combine_norm = (combine or "parallel").lower()
    if combine_norm == "parallel":
        return FeatureUnion(
            transformer_list=[
                ("skb", skb),
                ("l1", l1_sfm),
            ],
            n_jobs=n_jobs,
        )
    elif combine_norm == "sequential":
        # In sequence: L1 runs on the reduced space output by SelectKBest
        return Pipeline(
            [
                ("skb", skb),
                ("l1", l1_sfm),
            ]
        )
    else:
        raise ValueError("Parameter 'combine' must be either 'parallel' or 'sequential'.")
