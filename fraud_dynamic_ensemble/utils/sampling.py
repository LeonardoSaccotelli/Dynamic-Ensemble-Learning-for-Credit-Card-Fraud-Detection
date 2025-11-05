from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


def get_class_stats(df: pd.DataFrame, target: str) -> tuple[pd.Series, pd.Series]:
    """
    Compute per-class counts and percentages for a target column.

    This utility aggregates the frequency of each class label found in ``df[target]``
    and returns both the absolute counts and their percentages over the whole
    dataframe. Although typically used for binary targets (e.g., fraud vs. non-fraud),
    it also works for multi-class targets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the target column.
    target : str
        Name of the target column (e.g., ``"Class"``).

    Returns
    -------
    counts : pandas.Series
        Absolute counts per class. The index contains the class labels (sorted),
        and the values are integer counts.
    perc : pandas.Series
        Class percentages over all rows in ``df``. The index matches ``counts``,
        and values are rounded to 6 decimals.

    Raises
    ------
    KeyError
        If ``target`` is not a column of ``df``.

    Notes
    -----
    - Sorting by index ensures a deterministic class order in logs/plots.
    - When ``df`` is empty, both returned Series are empty.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Class": [0, 0, 1, 0, 1]})
    >>> counts, perc = get_class_stats(df, "Class")
    >>> counts.to_dict()
    {0: 3, 1: 2}
    >>> perc.to_dict()
    {0: 0.6, 1: 0.4}
    """
    # Count occurrences per class label and sort by the label for deterministic order
    counts = df[target].value_counts().sort_index()

    # Convert absolute counts to percentages over the full dataframe size
    perc = (counts / len(df)).round(6)

    return counts, perc


def derive_target_size(
    total_rows: int,
    *,
    policy: Literal["random", "stratified", "keep_all_minority"],
    n_rows: int | None,
    frac: float | None,
    minority_count: int,
) -> int:
    """
    Convert a size request (``n_rows`` or ``frac``) into an integer target sample size.

    The function enforces policy-specific constraints and clamps the computed size to
    feasible bounds given the dataset size and, when applicable, the minority-class
    count.

    Parameters
    ----------
    total_rows : int
        Total number of rows in the full dataset (must be >= 1).
    policy : {'random', 'stratified', 'keep_all_minority'}
        Sampling policy. If ``'keep_all_minority'`` is selected, the target size
        must be at least the number of minority samples.
    n_rows : int or None
        Absolute number of rows requested. Mutually exclusive with ``frac``.
    frac : float or None
        Fraction of the dataset requested, in the interval ``(0, 1]``. Mutually
        exclusive with ``n_rows``.
    minority_count : int
        Number of minority-class rows in the dataset. Used only to validate
        the ``'keep_all_minority'`` policy.

    Returns
    -------
    int
        Target sample size as a positive integer. For ``'keep_all_minority'``,
        clamped to ``[minority_count, total_rows]``; otherwise clamped to
        ``[1, total_rows]``.

    Raises
    ------
    ValueError
        - If both or neither of ``n_rows`` and ``frac`` are provided.
        - If ``frac`` is not in ``(0, 1]``.
        - If ``n_rows`` is not a positive integer.
        - If ``policy='keep_all_minority'`` and the requested size is smaller
          than ``minority_count``.

    Notes
    -----
    - When ``frac`` is used, the size is computed as ``int(total_rows * frac)``,
      i.e., flooring by design for determinism.
    - The caller is responsible for ensuring ``minority_count`` is correct for the
      same dataset of size ``total_rows``.

    Examples
    --------
    >>> derive_target_size(total_rows=270000, policy="random", n_rows=50000, frac=None, minority_count=492)
    50000
    >>> derive_target_size(total_rows=270000, policy="stratified", n_rows=None, frac=0.10, minority_count=492)
    27000
    >>> derive_target_size(total_rows=270000, policy="keep_all_minority", n_rows=80000, frac=None, minority_count=492)
    80000
    """
    # Enforce mutual exclusivity between absolute and fractional requests.
    if (n_rows is not None) and (frac is not None):
        raise ValueError("Provide exactly one of --n-rows or --frac (not both).")

    # Resolve the requested size from frac or n_rows.
    if frac is not None:
        if not (0.0 < frac <= 1.0):
            raise ValueError("--frac must be in the interval (0, 1].")
        requested = int(total_rows * frac)  # floor by design
    elif n_rows is not None:
        if n_rows <= 0:
            raise ValueError("--n-rows must be a positive integer.")
        requested = n_rows
    else:
        # Neither n_rows nor frac provided → cannot derive a size.
        raise ValueError("You must provide either --n-rows or --frac.")

    # Clamp to dataset bounds (never more than total_rows, never less than 1).
    requested = min(requested, total_rows)
    requested = max(requested, 1)

    # Additional constraint for keep_all_minority: ensure all minority rows can be kept.
    if policy == "keep_all_minority" and requested < minority_count:
        raise ValueError(
            f"Requested target size ({requested}) is smaller than the minority count "
            f"({minority_count}). With --policy keep_all_minority you must request at "
            "least the number of minority samples."
        )

    return requested


def sample_random(df: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    """
    Draw a non-stratified random sample of exactly ``n`` rows from a DataFrame.

    The function samples rows uniformly at random (without replacement), shuffles
    them, and resets the index. Class proportions (if any) are **not** preserved.
    For reproducibility, a fixed ``seed`` is used to set the random state.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to sample from.
    n : int
        Number of rows to sample. Must be a positive integer. If ``n`` exceeds
        ``len(df)``, it is clamped to ``len(df)``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A randomly sampled DataFrame with exactly ``min(n, len(df))`` rows,
        shuffled and with a zero-based, consecutive index.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    - Sampling is performed **without replacement**.
    - This method does **not** preserve class ratios. If you need to preserve
      class proportions, use ``sample_stratified`` instead.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(5)})
    >>> sample_random(df, n=3, seed=42).shape[0]
    3
    """
    # Validate requested size
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    # Clamp to available rows (uniform intent: “take as many as possible up to n”)
    if n > len(df):
        n = len(df)

    # Uniform random sample without replacement; reset index for cleanliness
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def sample_stratified(
    df: pd.DataFrame,
    *,
    target: str,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Draw a stratified random sample that preserves class proportions in ``target``.

    This function uses scikit-learn's ``train_test_split`` with the ``stratify`` flag
    to select exactly ``target_size`` rows while maintaining (as closely as possible)
    the original distribution of classes in ``df[target]``. Rows are returned shuffled
    with a reset, consecutive index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset that contains the target column.
    target : str
        Name of the (binary or multiclass) target column to stratify on.
    target_size : int
        Exact number of rows to select. Must satisfy ``1 ≤ target_size ≤ len(df)`` and
        also ``target_size ≥ n_classes`` for stratification to be feasible.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Stratified sample of exact size ``target_size`` (or a shuffled full copy when
        ``target_size >= len(df)``), with rows shuffled and index reset.

    Raises
    ------
    ValueError
        If ``target`` is missing, there are fewer than two classes, or
        ``target_size < n_classes``.

    Notes
    -----
    - When ``target_size >= len(df)``, ``train_test_split`` would fail because the
      complementary split would be empty; in that case we simply return a shuffled
      copy of the full dataset.
    - scikit-learn requires at least one instance per class in the stratified split;
      hence the ``target_size ≥ n_classes`` guard.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
    >>> df = pd.DataFrame(X).assign(Class=y)
    >>> out = sample_stratified(df, target="Class", target_size=200, seed=42)
    >>> len(out)
    200
    """
    # Basic validations
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = df[target]
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError("Stratified sampling requires at least two classes.")

    # If the user asks for all rows, just return a shuffled copy
    if target_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # scikit-learn needs at least one sample per class in the stratified split
    if target_size < n_classes:
        raise ValueError(
            f"`target_size` ({target_size}) must be at least the number of classes ({n_classes}) "
            "for stratified sampling."
        )

    # Perform stratified sampling with an exact integer train_size
    sample_df, _ = train_test_split(
        df,
        train_size=target_size,  # int is supported by scikit-learn
        stratify=y,
        random_state=seed,
        shuffle=True,
    )
    return sample_df.reset_index(drop=True)


def sample_keep_all_minority(
    df: pd.DataFrame,
    *,
    target: str,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Build a sample that **keeps all minority-class rows** and fills the remainder
    with randomly selected majority-class rows (without replacement) until
    ``target_size`` is reached, then shuffles the result.

    This is useful when the minority class is tiny, and you want to
    guarantee its full presence in the working subset while constraining the
    overall dataset size.

    Parameters
    ----------
    df : pandas.DataFrame
        Full input dataset.
    target : str
        Binary target column name. The function infers the minority class as the
        label with the smallest count.
    target_size : int
        Desired final sample size. Must be **>=** the minority-class count; if larger
        than the total available rows, the function caps at the maximum feasible size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of up to ``target_size`` rows containing **all** minority rows
        plus enough majority rows to reach ``target_size`` (if available). Rows are
        shuffled and the index is reset.

    Raises
    ------
    ValueError
        - If ``target`` is missing in ``df``.
        - If ``target`` is not binary (i.e., number of distinct classes != 2).
        - If ``target_size`` is smaller than the number of minority rows.

    Notes
    -----
    - Majority rows are drawn **without replacement**.
    - When ``target_size`` exceeds the total number of rows, the output will simply
      contain **all** rows (shuffled).
    - If you need *exact* class counts, consider using
      ``imblearn.under_sampling.RandomUnderSampler``. This pure-pandas version
      matches the intent and keeps dependencies minimal.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(10), "Class": [0]*9 + [1]})
    >>> out = sample_keep_all_minority(df, target="Class", target_size=6, seed=7)
    >>> out["Class"].value_counts().to_dict()  # keeps the single '1' and adds 5 zeros
    {0: 5, 1: 1}
    """
    # Validate the target column is present
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Count class frequencies and require a binary target
    counts = df[target].value_counts()
    if counts.size != 2:
        raise ValueError(
            f"'keep_all_minority' requires a binary target (found {counts.size} classes)."
        )

    # Identify minority and majority labels by frequency
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    # Split the dataframe by class
    df_min = df[df[target] == minority_label]
    df_maj = df[df[target] == majority_label]

    n_min = len(df_min)
    n_maj = len(df_maj)

    # The requested size must be at least the number of minority rows
    if target_size < n_min:
        raise ValueError(
            f"target_size ({target_size}) < minority_count ({n_min}); cannot keep all minority."
        )

    # Compute how many majority rows we need; clamp to availability
    majority_needed = target_size - n_min
    majority_needed = min(majority_needed, n_maj)

    # Draw the requested number of majority rows (without replacement) and combine
    if majority_needed > 0:
        df_maj_sample = df_maj.sample(n=majority_needed, random_state=seed, replace=False)
        out = pd.concat([df_min, df_maj_sample], axis=0)
    else:
        # Corner case: requested size equals the minority count
        out = df_min.copy()

    # Final shuffle for neutrality and reset index for cleanliness
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def apply_sampling(
    df: pd.DataFrame,
    *,
    policy: Literal["random", "stratified", "keep_all_minority"],
    target: str,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Apply the selected sampling policy and return the sampled DataFrame.

    This dispatcher routes to one of the concrete sampling strategies:
    ``sample_random`` (uniform, non-stratified), ``sample_stratified`` (preserve
    class proportions in ``target``), or ``sample_keep_all_minority`` (keep all
    minority rows and fill with majority up to ``target_size``).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to be sampled.
    policy : {'random', 'stratified', 'keep_all_minority'}
        Sampling policy to apply.
    target : str
        Name of the target column (used by 'stratified' and 'keep_all_minority').
        Kept for a uniform signature even when ``policy='random'``.
    target_size : int
        Exact number of rows requested for the sample. Assumed validated upstream
        (e.g., via ``derive_target_size``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Sampled DataFrame according to the chosen policy.

    Raises
    ------
    ValueError
        If an unknown ``policy`` is provided.

    Notes
    -----
    - For ``policy='random'``, class proportions are not preserved.
    - For ``policy='stratified'``, scikit-learn's stratified split is used.
    - For ``policy='keep_all_minority'``, all minority rows are included by design.

    Examples
    --------
    >>> sample = apply_sampling(df, policy="random", target="Class", target_size=50000, seed=42)
    >>> sample = apply_sampling(df, policy="stratified", target="Class", target_size=50000, seed=42)
    >>> sample = apply_sampling(df, policy="keep_all_minority", target="Class", target_size=80000, seed=42)
    """
    # Route to the concrete implementation based on the selected policy.
    if policy == "random":
        return sample_random(df, n=target_size, seed=seed)
    elif policy == "stratified":
        return sample_stratified(df, target=target, target_size=target_size, seed=seed)
    elif policy == "keep_all_minority":
        return sample_keep_all_minority(df, target=target, target_size=target_size, seed=seed)
    else:
        # Defensive guard: enforce explicit failure on unsupported options.
        raise ValueError(f"Unknown policy '{policy}'.")
