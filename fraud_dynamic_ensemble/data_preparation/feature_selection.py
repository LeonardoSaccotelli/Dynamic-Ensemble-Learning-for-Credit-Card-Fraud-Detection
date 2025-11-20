from __future__ import annotations

from typing import Optional, Union, Tuple

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
    C: float = 0.1,
    solver: str = "liblinear",
    class_weight: Optional[str] = "balanced",
    max_iter: int = 2000,
    n_jobs: Optional[int] = None,       # kept for API compatibility (ignored)
    random_state: Optional[int] = None,
) -> Tuple[SelectKBest, SelectFromModel]:
    """
    Build **two separate** feature selectors and return them uncombined.

    This factory returns:
      1) a filter selector: ``SelectKBest(score=...)``, and
      2) an embedded selector: ``SelectFromModel(LogisticRegression(penalty='l1', ...)``

    Composition (parallel union vs sequential pipeline) is intentionally **not**
    performed here; you can assemble them later as needed.

    Parameters
    ----------
    k : int or {"all"}, default=20
        Number of top features to keep in the filter step. Use ``"all"`` to pass all
        features through the filter branch.
    score : {"f_classif", "mutual_info_classif", "chi2"}, default="f_classif"
        Scoring function for ``SelectKBest``:
        - ``"f_classif"`` → ANOVA F-test (fast, linear association).
        - ``"mutual_info_classif"`` → Mutual information (captures non-linear dependence).
        - ``"chi2"`` → Chi-square (**requires non-negative features**).
    C : float, default=0.1
        Inverse regularization strength for the L1-penalized Logistic Regression used
        by ``SelectFromModel`` (smaller → stronger sparsity).
    solver : {"liblinear", "saga"}, default="liblinear"
        Logistic Regression solver supporting ``penalty='l1'``.
    class_weight : {"balanced", None}, optional, default="balanced"
        Class weighting for Logistic Regression (use ``"balanced"`` for strong imbalance).
    max_iter : int, default=2000
        Maximum iterations for Logistic Regression.
    n_jobs : int or None, optional, default=None
        **Ignored.** (Previously used for parallel ``FeatureUnion``.)
    random_state : int or None, optional, default=None
        Random seed for Logistic Regression (useful with ``solver='saga'``).

    Returns
    -------
    skb : sklearn.feature_selection.SelectKBest
        The filter-based selector configured with the requested scoring function.
    l1_sfm : sklearn.feature_selection.SelectFromModel
        The embedded-model selector wrapping an L1-penalized Logistic Regression.

    Notes
    -----
    - **Scaling**: Place a ``StandardScaler`` **before** the L1 selector for stable paths.
      Do **not** standardize if using ``score='chi2'``; use a non-negative scaling (e.g.,
      ``MinMaxScaler`` to [0, 1]).

    Examples
    --------
    >>> skb, l1 = get_feature_selection(k=20, score="f_classif", C=0.1, solver="liblinear")
    >>> from sklearn.pipeline import FeatureUnion
    >>> union = FeatureUnion([('skb', skb), ('l1', l1)])

    >>> skb, l1 = get_feature_selection(k=30, score="mutual_info_classif", C=0.2, solver="saga")
    >>> from sklearn.pipeline import Pipeline
    >>> seq = Pipeline([('skb', skb), ('l1', l1)])
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
        n_jobs=n_jobs,
    )
    l1_sfm = SelectFromModel(estimator=l1_est)

    return skb, l1_sfm