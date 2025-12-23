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
    Compute the two-sample Kolmogorov–Smirnov test statistic and p-value.

    This function runs SciPy's two-sample KS test to quantify the maximum absolute
    difference between the empirical CDFs of two samples. Non-finite values
    (NaN/Inf) are removed prior to testing.

    Parameters
    ----------
    x, y : array-like of shape (n_samples,)
        Numeric samples. Non-finite values (NaN/Inf) are removed internally.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis passed to SciPy.
    method : {'auto', 'exact', 'asymp'}, default 'auto'
        Computation method used by SciPy. ``'auto'`` selects a suitable method
        based on sample size.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``'D'`` : float
            KS statistic (maximum CDF distance).
        - ``'p'`` : float
            P-value for the chosen alternative.
        - ``'n_x'`` : int
            Number of valid (finite) observations used from ``x``.
        - ``'n_y'`` : int
            Number of valid (finite) observations used from ``y``.

    Raises
    ------
    ValueError
        If either sample is empty after filtering non-finite values.

    Notes
    -----
    - The KS test is sensitive to differences in location, scale, and distribution shape.
    - With large sample sizes, small distribution differences may yield very small
      p-values; interpret the p-value jointly with the effect size ``D``.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([1, 2, 3, 4, 5])
    >>> y = pd.Series([1, 2, 2, 3, 4, 6])
    >>> ks_two_sample(x, y)["D"] >= 0.0
    True
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
