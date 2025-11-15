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
from sklearn.pipeline import FeatureUnion


def get_feature_selection(
    *,
    k: Union[int, str] = 20,
    score: str = "f_classif",
    C: float = 0.1,
    solver: str = "liblinear",
    class_weight: Optional[str] = "balanced",
    max_iter: int = 2000,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> FeatureUnion:
    """
    Build a feature-selection transformer that **unions** two selectors:
    1) a filter method: ``SelectKBest(score=...)``
    2) a wrapper method: ``SelectFromModel(LogisticRegression(penalty='l1', solver=...))``

    At transform time, the outputs of both branches are concatenated horizontally
    via a ``FeatureUnion`` (i.e., features kept by either method are included).

    Parameters
    ----------
    k : int or {"all"}, default=20
        Number of top features to keep in the filter step (``SelectKBest``).
        Use ``"all"`` to pass all features through the filter branch.
    score : {"f_classif", "mutual_info_classif", "chi2"}, default="f_classif"
        Scoring function for ``SelectKBest``:
        - ``"f_classif"``: ANOVA F-test (linear association, very fast).
        - ``"mutual_info_classif"``: Mutual information (captures non-linear dependence).
        - ``"chi2"``: Chi-square test (**requires non-negative features**).
    C : float, default=0.1
        Inverse regularization strength for the L1-penalized Logistic Regression
        used inside ``SelectFromModel``. Smaller values ⇒ stronger sparsity.
    solver : {"liblinear", "saga"}, default="liblinear"
        Logistic Regression solver supporting ``penalty='l1'``.
    class_weight : {"balanced", None}, optional, default="balanced"
        Class weighting for Logistic Regression. Use ``"balanced"`` for heavily
        imbalanced datasets.
    max_iter : int, default=2000
        Maximum iterations for Logistic Regression.
    n_jobs : int or None, optional, default=None
        Parallelism for the ``FeatureUnion`` (its branches can be fit in parallel).
    random_state : int or None, optional, default=None
        Random seed for Logistic Regression (useful with stochastic solvers like ``"saga"``).

    Returns
    -------
    sklearn.pipeline.FeatureUnion
        A transformer suitable for inclusion in a scikit-learn ``Pipeline``.
        During ``fit``, each branch selects features independently;
        during ``transform``, their outputs are concatenated.

    Notes
    -----
    - **Scaling**: Place a ``StandardScaler`` **before** this selector to stabilize
      the L1 path of Logistic Regression. **Do not standardize** if you use
      ``score="chi2"``; ``chi2`` requires non-negative features (use ``MinMaxScaler``
      to [0, 1] instead).
    - **Duplicates**: Since outputs are concatenated, if both branches choose the same
      original columns you will effectively duplicate them in the union. This is usually
      harmless; you can deduplicate later if desired.
    - **Imbalance**: If you apply resampling (e.g., SMOTE/undersampling), place it
      **inside** the CV pipeline and **before** this selector to avoid leakage.

    Examples
    --------
    ANOVA F-test + L1-Logistic, with standardization:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> selector = get_feature_selection(k=20, score="f_classif", C=0.1, solver="liblinear")
    >>> pipe = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("select", selector),
    ...     ("clf", LogisticRegression(max_iter=2000))
    ... ])

    Mutual information + L1-Logistic (tune width and sparsity):

    >>> selector = get_feature_selection(k=30, score="mutual_info_classif", C=0.2, solver="saga")

    Chi-square (ensure non-negative inputs, e.g., via MinMax scaling):

    >>> from sklearn.preprocessing import MinMaxScaler
    >>> pipe = Pipeline([
    ...     ("minmax", MinMaxScaler()),
    ...     ("select", get_feature_selection(k=25, score="chi2", C=0.1)),
    ...     ("clf", LogisticRegression(max_iter=2000))
    ... ])

    Hyperparameters to consider in grid search (assuming the union is named ``select``):
    - Filter width: ``select__skb__k``
    - L1 strength/weights: ``select__l1__estimator__C``, ``select__l1__estimator__class_weight``
    - Solver choice: ``select__l1__estimator__solver`` (must be one of ``{"liblinear","saga"}``)
    """
    # map score string to callable
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

    # validate solver for L1
    if solver not in {"liblinear", "saga"}:
        raise ValueError(
            f"Solver '{solver}' does not support L1 penalty. Use 'liblinear' or 'saga'."
        )

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

    selector = FeatureUnion(
        transformer_list=[
            ("skb", skb),
            ("l1", l1_sfm),
        ],
        n_jobs=n_jobs,
    )
    return selector
