from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def ks_two_sample(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
) -> Dict[str, float | int]:
    """
    Kolmogorov–Smirnov two-sample test (global CDF difference).

    Parameters
    ----------
    x, y : array-like of shape (n_samples,)
        Numeric samples. NaN/Inf are removed internally.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Defines the alternative hypothesis.
    method : {'auto', 'exact', 'asymp'}, default 'auto'
        Algorithm used by SciPy; 'auto' chooses a suitable method by sample size.

    Returns
    -------
    dict
        {'D': float, 'p': float, 'n_x': int, 'n_y': int}

    Raises
    ------
    ValueError
        If either sample is empty after filtering invalid values.

    Notes
    -----
    - Sensitive to differences in location, scale, and shape (global distribution).
    - With very large n, tiny distribution shifts can yield very small p-values,
      so report/inspect both D and p.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([1, 2, 3, 4, 5])
    >>> y = pd.Series([1, 2, 2, 3, 4, 6])
    >>> out = ks_two_sample(x, y)
    >>> out['p']
    1.0
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()

    # Drop non-finite values
    xv = xv[np.isfinite(xv)]
    yv = yv[np.isfinite(yv)]

    n_x = int(xv.size)
    n_y = int(yv.size)
    if n_x == 0 or n_y == 0:
        raise ValueError("KS test requires non-empty samples after filtering NaN/Inf.")

    D, p = stats.ks_2samp(xv, yv, alternative=alternative, method=method)
    return {"D": float(D), "p": float(p), "n_x": n_x, "n_y": n_y}
