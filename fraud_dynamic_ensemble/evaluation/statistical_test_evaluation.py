from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats
from scipy.stats import t


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


def corrected_resampled_ttest(metric_model_a, metric_model_b, n_train, n_test):
    """
    Compute the corrected resampled t-test (Nadeau and Bengio) to compare two models.

    This function implements the corrected resampled t-test for paired performance
    estimates obtained from repeated resampling (e.g., repeated cross-validation).
    It computes per-split score differences, estimates their mean and variance, and
    applies the Nadeau–Bengio correction to the standard error to account for the
    dependence between resamples induced by training set overlap.

    The p-value is computed via the survival function (:meth:`scipy.stats.t.sf`) for
    improved numerical precision in the tails. A small-denominator guard is used to
    avoid division by (near) zero when the variance of differences is negligible; in
    that case the models are treated as effectively identical.

    Parameters
    ----------
    metric_model_a : array-like
        Per-resample metric scores for Model A. Must be 1D and have the same length as
        ``metric_model_b``.
    metric_model_b : array-like
        Per-resample metric scores for Model B. Must be 1D and have the same length as
        ``metric_model_a``.
    n_train : int
        Number of samples used for training in each resample/split.
    n_test : int
        Number of samples used for testing in each resample/split.

    Returns
    -------
    t_stat : float
        Corrected resampled t-statistic.
    p_value : float
        Two-tailed p-value computed from a t-distribution with ``df = n - 1``, where
        ``n`` is the number of paired resample scores.

    Notes
    -----
    - Differences are computed as ``diffs = metric_model_a - metric_model_b``.
    - The corrected standard error uses::

          correction_factor = (1 / n) + (n_test / n_train)

      where ``n`` is the number of paired resamples.
    - If the corrected denominator is smaller than ``1e-10``, the function returns
      ``(0.0, 1.0)`` to indicate no detectable difference.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import t
    >>> a = np.array([0.91, 0.89, 0.90, 0.92, 0.88])
    >>> b = np.array([0.90, 0.88, 0.90, 0.91, 0.88])
    >>> t_stat, p_value = corrected_resampled_ttest(a, b, n_train=20000, n_test=5000)
    >>> (np.isfinite(t_stat) and 0.0 <= p_value <= 1.0)
    True
    """

    # 1. Calculate differences
    diffs = np.array(metric_model_a) - np.array(metric_model_b)
    n = len(diffs)

    # 2. Calculate Mean and Variance
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)

    # 3. Compute Corrected Standard Error
    correction_factor = (1 / n) + (n_test / n_train)
    denominator = np.sqrt(var_diff * correction_factor)

    # 4. Safety Check: Avoid division by zero (or near-zero)
    # If the variance is tiny, models are effectively identical -> p-value = 1.0
    if denominator < 1e-10:
        return 0.0, 1.0

    # 5. Compute t-statistic
    t_stat = mean_diff / denominator

    # 6. Compute p-value (Two-tailed)
    # t.sf(x) is the Survival Function (1 - cdf).
    # It is more precise for calculating small tail probabilities.
    df = n - 1
    p_value = 2 * t.sf(np.abs(t_stat), df)

    return t_stat, p_value
