from __future__ import annotations

from typing import Union

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


def get_feature_selection(
    *,
    k: Union[int, str] = 20,
    score: str = "f_classif",
) -> SelectKBest:
    """
    Build a **single** filter-based feature selector (``SelectKBest``).

    This factory returns a filter selector configured with the requested
    scoring function, intended to be used as a pipeline step named
    ``"feature_selection_filter"`` (so it can be tuned via
    ``feature_selection_filter__k`` in CV searches).

    Parameters
    ----------
    k : int or {"all"}, default=20
        Number of top features to keep. Use ``"all"`` to keep all features.
        This parameter can be tuned inside a CV search by setting
        ``feature_selection_filter__k`` in ``param_distributions``.
    score : {"f_classif", "mutual_info_classif", "chi2"}, default="f_classif"
        Scoring function for ``SelectKBest``:
        - ``"f_classif"`` → ANOVA F-test (fast, captures linear association).
        - ``"mutual_info_classif"`` → Mutual information (captures non-linear dependence).
        - ``"chi2"`` → Chi-square (**requires non-negative features**).

    Returns
    -------
    skb : sklearn.feature_selection.SelectKBest
        A ``SelectKBest`` selector configured with the requested scoring function
        and ``k`` value.

    Raises
    ------
    ValueError
        If ``score`` is not one of ``{"chi2", "f_classif", "mutual_info_classif"}``.

    Notes
    -----
    - If using ``score="chi2"``, ensure features are non-negative. In practice,
      this often implies a non-negative scaling strategy (e.g., MinMax scaling
      to [0, 1]) rather than standardization around zero.

    Examples
    --------
    Basic usage:

    >>> skb = get_feature_selection(k=20, score="f_classif")

    Typical pipeline usage (step name matches CV tuning keys):

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
