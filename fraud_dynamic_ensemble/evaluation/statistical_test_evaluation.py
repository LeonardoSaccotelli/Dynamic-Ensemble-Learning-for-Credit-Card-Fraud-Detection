from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t

from fraud_dynamic_ensemble.evaluation.visual_evaluation import (
    plot_ranking_with_significance,
    plot_significance_heatmap,
)


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


def corrected_resampled_ttest(
    metric_model_a: np.ndarray | Sequence[float],
    metric_model_b: np.ndarray | Sequence[float],
    n_train: int | float,
    n_test: int | float,
) -> tuple[float, float]:
    """
    Compute the corrected resampled t-test (Nadeau–Bengio) to compare two models.

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
    metric_model_a : numpy.ndarray or Sequence[float]
        Per-resample metric scores for Model A. Must be 1D and have the same length as
        ``metric_model_b``.
    metric_model_b : numpy.ndarray or Sequence[float]
        Per-resample metric scores for Model B. Must be 1D and have the same length as
        ``metric_model_a``.
    n_train : int or float
        Number of samples used for training in each resample/split. Must be positive.
    n_test : int or float
        Number of samples used for testing in each resample/split. Must be positive.

    Returns
    -------
    t_stat : float
        Corrected resampled t-statistic.
    p_value : float
        Two-tailed p-value computed from a t-distribution with ``df = n - 1``, where
        ``n`` is the number of paired resample scores.

    Raises
    ------
    ValueError
        If ``n_train`` or ``n_test`` is not positive, if the input vectors have different
        lengths, or if fewer than two paired scores are provided (variance with ``ddof=1``
        is undefined).
    TypeError
        If inputs cannot be converted to numeric arrays.

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


def compute_pairwise_corrected_resampled_ttest(
    df: pd.DataFrame,
    metric_name: str,
    n_train: int | float,
    n_test: int | float,
    save_path: Path,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute ordered pairwise corrected resampled t-tests (Nadeau–Bengio) for a single metric and save outputs.

    This function performs an all-vs-all **ordered** pairwise comparison across models for one
    metric column. For each ordered pair (``Model_A``, ``Model_B``) with ``Model_A != Model_B``,
    it extracts the per-model metric vectors from ``df``, applies :func:`corrected_resampled_ttest`,
    and records the resulting t-statistic and p-value.

    The function also computes the per-model means and their difference (A − B), derives a
    significance flag based on ``alpha``, and assigns a qualitative label:

    - ``"Better"``: Model_A significantly outperforms Model_B (p < alpha and t-stat > 0)
    - ``"Worse"``:  Model_A significantly underperforms Model_B (p < alpha and t-stat < 0)
    - ``"No Diff"``: no statistically significant difference (p >= alpha)

    After computing all comparisons, the function saves a metric-specific CSV under ``save_path``
    and triggers two plots:
    - :func:`plot_significance_heatmap` (p-value heatmap), and
    - :func:`plot_ranking_with_significance` (ranked barplot with clique lines).

    Parameters
    ----------
    df : pandas.DataFrame
        Input results DataFrame. Must contain:
        - ``"model"``: model identifier, and
        - ``metric_name``: numeric column holding the metric scores to compare.

        The function sorts ``df`` in-place by ``"model"`` to enforce consistent model ordering.
    metric_name : str
        Metric column name to compare across models (e.g., ``"roc_auc"``, ``"f1"``).
        Must exist in ``df`` and be numeric.
    n_train : int or float
        Number of samples used for training in each resample/split. Forwarded to
        :func:`corrected_resampled_ttest`. Must be positive.
    n_test : int or float
        Number of samples used for testing in each resample/split. Forwarded to
        :func:`corrected_resampled_ttest`. Must be positive.
    save_path : pathlib.Path
        Output directory where the CSV and figures will be written. The directory must exist
        and be writable.
    alpha : float, default=0.05
        Significance threshold used to set ``is_significant`` and the ``result`` label.
        Must be in ``(0, 1)``.

    Returns
    -------
    pandas.DataFrame
        Pairwise comparison table with one row per **ordered** model pair (A, B), containing:

        - ``"Metric"``: the metric name (constant = ``metric_name``),
        - ``"Model_A"``, ``"Model_B"``: ordered model pair,
        - ``"Mean_Diff"``: mean(metric_A) − mean(metric_B),
        - ``"t-stat"``: corrected resampled t-statistic,
        - ``"p-value"``: two-tailed p-value,
        - ``"is_significant"``: boolean (p-value < alpha),
        - ``"result"``: categorical label in {``"Better"``, ``"Worse"``, ``"No Diff"``}.

    Raises
    ------
    KeyError
        If ``"model"`` or ``metric_name`` is not present in ``df``.
    ValueError
        If ``alpha`` is not in ``(0, 1)``, or if ``n_train``/``n_test`` are not positive.
    PermissionError
        If the CSV or figures cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during file writing.

    Notes
    -----
    - The pairwise loop is **ordered**: A vs B and B vs A are both computed; self-comparisons
      are skipped.
    - Vectors are extracted as::

          vec_a = df[df["model"] == model_a][metric_name].values
          vec_b = df[df["model"] == model_b][metric_name].values

      If ``len(vec_a) != len(vec_b)``, the pair is skipped and a warning is printed. This
      typically indicates missing rows for one model (e.g., missing folds/iterations).
    - Results are saved as::

          corrected_resampled_ttest_<metric_name>.csv

      Plots are generated via::

          plot_significance_heatmap(df_comparisons, metric_name, save_path)
          plot_ranking_with_significance(df, df_comparisons, metric_name, save_path)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df = pd.DataFrame(
    ...     {
    ...         "model": ["A", "A", "B", "B"],
    ...         "roc_auc": [0.91, 0.89, 0.93, 0.92],
    ...     }
    ... )
    >>> out = compute_pairwise_corrected_resampled_ttest(
    ...     df=df,
    ...     metric_name="roc_auc",
    ...     n_train=20000,
    ...     n_test=5000,
    ...     save_path=Path("."),
    ...     alpha=0.05,
    ... )
    >>> {"Model_A", "Model_B", "p-value", "result"}.issubset(out.columns)
    True
    """

    print(
        f"\n{'=' * 80}\nCORRECTED RESAMPLED t-TEST ANALYSIS (Pairwise): {metric_name.upper()}\n{'=' * 80}"
    )

    # 1. Sort Models for consistent ordering
    df.sort_values(by=["model"], inplace=True)
    unique_models = df["model"].unique()

    comparison_records = []

    # 2. Compute Statistics (All vs All)
    for model_a in unique_models:
        for model_b in unique_models:
            if model_a == model_b:
                continue

            # Extract vectors for the specific metric
            vec_a = df[df["model"] == model_a][metric_name].values
            vec_b = df[df["model"] == model_b][metric_name].values

            # Sanity check
            if len(vec_a) != len(vec_b):
                print(f"Warning: Size mismatch {model_a} vs {model_b}")
                continue

            # Perform Statistical Test
            t_stat, p_val = corrected_resampled_ttest(vec_a, vec_b, n_train, n_test)

            # Calculate Means (for printing and magnitude)
            mean_a = np.mean(vec_a)
            mean_b = np.mean(vec_b)
            mean_diff = mean_a - mean_b

            # Determine Relationship Label
            is_significant = p_val < alpha

            if is_significant and t_stat > 0:
                relationship = "Better"  # A is significantly better than B
            elif is_significant and t_stat < 0:
                relationship = "Worse"  # A is significantly worse than B
            else:
                relationship = "No Diff"  # Statistical Tie

            # --- DETAILED PRINTING ---
            sig_label = "Statistically SIGNIFICANT" if is_significant else "NOT significant"
            print(f"Comparing '{model_a}' (mean={mean_a:.3f}) vs '{model_b}' (mean={mean_b:.3f})")
            print(f"  p-value: {p_val:.4f} (t={t_stat:.4f}) -> {sig_label}")
            print("-" * 80)

            comparison_records.append(
                {
                    "Metric": metric_name,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "Mean_Diff": mean_diff,
                    "t-stat": t_stat,
                    "p-value": p_val,
                    "is_significant": is_significant,
                    "result": relationship,
                }
            )

    # 3. Create DataFrame
    df_comparisons = pd.DataFrame(comparison_records)

    # 4. Save Single Metric CSV
    csv_filename = f"corrected_resampled_ttest_{metric_name}.csv"
    df_comparisons.to_csv(save_path / csv_filename, index=False)

    # 5. Generate Visualization (Heatmap)
    plot_significance_heatmap(df_comparisons, metric_name, save_path)
    plot_ranking_with_significance(df, df_comparisons, metric_name, save_path)

    return df_comparisons
