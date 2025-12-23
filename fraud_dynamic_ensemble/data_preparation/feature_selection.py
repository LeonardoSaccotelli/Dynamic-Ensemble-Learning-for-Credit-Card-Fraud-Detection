from __future__ import annotations

from typing import Union

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


def get_feature_selection(
    *,
    k: Union[int, str] = 20,
    score: str = "f_classif",
) -> SelectKBest:
    """
    Build a filter-based feature selector using SelectKBest.

    This factory returns a :class:`sklearn.feature_selection.SelectKBest`
    configured with the requested scoring function. It is designed to be used
    as a pipeline step named ``"feature_selection_filter"`` so that ``k`` can be
    tuned via ``feature_selection_filter__k`` in CV searches.

    Parameters
    ----------
    k : int or {'all'}, default 20
        Number of top features to keep. Use ``'all'`` to keep all features.
    score : {'f_classif', 'mutual_info_classif', 'chi2'}, default 'f_classif'
        Scoring function used to rank features.
        If ``'f_classif'``, use the ANOVA F-test.
        If ``'mutual_info_classif'``, use mutual information.
        If ``'chi2'``, use chi-square (requires non-negative features).

    Returns
    -------
    sklearn.feature_selection.SelectKBest
        A SelectKBest selector configured with the chosen scoring function and ``k``.

    Raises
    ------
    ValueError
        If ``score`` is not one of ``{'chi2', 'f_classif', 'mutual_info_classif'}``.

    Notes
    -----
    When using ``score='chi2'``, ensure all input features are non-negative (e.g.,
    by using MinMax scaling). Standardization around zero is typically incompatible
    with chi-square feature selection.

    Examples
    --------
    >>> skb = get_feature_selection(k=20, score="f_classif")

    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([("feature_selection_filter", skb)])
    """

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

    return SelectKBest(score_func=score_func, k=k)
