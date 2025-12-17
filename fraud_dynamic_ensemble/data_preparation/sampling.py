from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split


def get_class_stats(df: pd.DataFrame, target: str) -> tuple[Series[Any], Series[Any], int, int]:
    """
    Compute per-class counts, percentages, and dataframe shape.

    This utility aggregates the frequency of each class label found in ``df[target]``
    and returns both the absolute counts and their percentages, along with the
    total number of rows and columns. Although typically used for binary targets
    (e.g., fraud vs. non-fraud), it also works for multi-class targets.

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
    n_rows : int
        Total number of rows in ``df``.
    n_cols : int
        Total number of columns in ``df``.

    Raises
    ------
    KeyError
        If ``target`` is not a column of ``df``.

    Notes
    -----
    - Sorting by index ensures a deterministic class order in logs/plots.
    - When ``df`` is empty, both returned Series are empty, and ``n_rows`` is 0.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Class": [0, 0, 1, 0, 1], "Other": [1,2,3,4,5]})
    >>> counts, perc, rows, cols = get_class_stats(df, "Class")
    >>> counts.to_dict()
    {0: 3, 1: 2}
    >>> perc.to_dict()
    {0: 0.6, 1: 0.4}
    >>> rows
    5
    >>> cols
    2
    """
    # Get dataframe dimensions
    n_rows, n_cols = df.shape

    # Count occurrences per class label and sort by the label for deterministic order
    counts = df[target].value_counts().sort_index()

    # Convert absolute counts to percentages over the full dataframe size
    perc = (counts / len(df)).round(6)

    return counts, perc, n_rows, n_cols


def random_sampling(
    df: pd.DataFrame,
    n_rows: int | None = None,
    frac: float | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Perform simple random sampling **without replacement**.

    Exactly one of ``n_rows`` or ``frac`` must be provided. Sampling is
    reproducible when ``seed`` is set (forwarded to ``random_state``).
    The returned sample is shuffled and its index reset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to sample from.
    n_rows : int or None, optional
        Absolute number of rows to sample. Mutually exclusive with ``frac``.
        If larger than ``len(df)``, it is clamped to ``len(df)``.
    frac : float or None, optional
        Fraction of rows to sample, strictly in ``(0, 1]``. Mutually exclusive
        with ``n_rows``.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Randomly sampled DataFrame with a consecutive, zero-based index.

    Raises
    ------
    ValueError
        If **both or neither** of ``n_rows`` and ``frac`` are provided;
        if ``n_rows`` is not positive; or if ``frac`` is not in ``(0, 1]``.

    Notes
    -----
    - Sampling is performed **without replacement**.
    - With ``frac``, pandas determines the exact sample size via internal
      rounding. If you need an exact size, compute it and pass via ``n_rows``.

    Examples
    --------
    Basic usage with a fixed count:

    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(10)})
    >>> out = random_sampling(df, n_rows=3, seed=42)
    >>> len(out)
    3

    Fractional sampling (0.5 → 5 rows for a 10-row DataFrame):

    >>> out = random_sampling(df, frac=0.5, seed=0)
    >>> len(out)
    5
    """
    using_n = n_rows is not None
    using_f = frac is not None

    # Enforce mutual exclusivity between absolute and fractional requests.
    if using_n == using_f:
        raise ValueError("Provide exactly one of 'n_rows' or 'frac' (not both or neither).")

    if using_n:
        # Validate n_rows
        if n_rows is None or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")
        size = min(n_rows, len(df))
        sample_df = df.sample(n=size, replace=False, random_state=seed)
    else:
        # Validate frac
        if not (0.0 < frac <= 1.0):
            raise ValueError("frac must be in the interval (0, 1].")
        sample_df = df.sample(frac=frac, replace=False, random_state=seed)

    return sample_df.reset_index(drop=True)


def stratified_random_sampling(
    df: pd.DataFrame,
    stratify_by: str,
    n_rows: int | None = None,
    frac: float | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Perform stratified random sampling on ``df`` using ``stratify_by`` as the stratum key.

    Exactly one of ``n_rows`` or ``frac`` must be provided. Internally, a single call to
    ``sklearn.model_selection.train_test_split`` is used after computing a unified
    ``train_size`` parameter. The result is shuffled and its index reset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset that contains the stratification column.
    stratify_by : str
        Column name to use for stratification (binary or multiclass).
    n_rows : int or None, optional
        Absolute number of rows requested (mutually exclusive with ``frac``).
        If ``n_rows >= len(df)``, a shuffled full copy is returned.
    frac : float or None, optional
        Fraction in ``(0, 1]`` of rows requested (mutually exclusive with ``n_rows``).
        If ``frac == 1.0``, a shuffled full copy is returned.
    seed : int or None, optional
        Random seed forwarded to scikit-learn's ``random_state``.

    Returns
    -------
    pandas.DataFrame
        Stratified sample (rows shuffled, index reset).

    Raises
    ------
    ValueError
        If the stratification column is missing, there are fewer than two classes,
        if both or neither of ``n_rows``/``frac`` are provided, if ``n_rows`` is not
        positive, or if ``frac`` is not in ``(0, 1]``. Also raised if the requested
        train size would contain fewer samples than the number of classes.

    Notes
    -----
    - scikit-learn requires **at least one sample per class** in the train subset.
    - When the requested size equals or exceeds ``len(df)``, a shuffled copy of the
      full dataset is returned (no split).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(100), "y": [0]*90 + [1]*10})
    >>> out = stratified_random_sampling(df, stratify_by="y", n_rows=20, seed=42)
    >>> len(out)
    20
    >>> out = stratified_random_sampling(df, stratify_by="y", frac=0.2, seed=42)
    >>> len(out)
    20
    """
    if stratify_by not in df.columns:
        raise ValueError(f"Stratification column '{stratify_by}' not found in DataFrame.")

    y = df[stratify_by]
    n_classes = int(y.nunique())
    if n_classes < 2:
        raise ValueError("Stratified sampling requires at least two classes.")

    using_n = n_rows is not None
    using_f = frac is not None
    if using_n == using_f:
        raise ValueError("Provide exactly one of 'n_rows' or 'frac' (not both or neither).")

    n_total = len(df)

    # Compute unified train_size parameter (int or float) and handle "all rows" early.
    if using_n:
        if n_rows is None or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")
        if n_rows >= n_total:
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sample_size: int | float = n_rows
        train_count = n_rows
    else:
        assert frac is not None  # type narrowing
        if not (0.0 < frac <= 1.0):
            raise ValueError("frac must be in the interval (0, 1].")
        if frac == 1.0:
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sample_size = float(frac)
        train_count = max(1, int(n_total * sample_size))  # ensure ≥ one per class

    if train_count < n_classes:
        raise ValueError(
            f"Requested train size ({train_count}) is smaller than the number of classes "
            f"({n_classes}); increase 'n_rows' or 'frac'."
        )

    sample_df, _ = train_test_split(
        df,
        train_size=sample_size,  # int or float
        stratify=y,
        random_state=seed,
        shuffle=True,
    )
    return sample_df.reset_index(drop=True)


def keep_all_minority_random_sampling(
    df: pd.DataFrame,
    target: str,
    n_rows: int | None = None,
    frac: float | None = None,
    *,
    ratio: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Keep **all** minority-class rows in ``target`` and add a random subset of majority rows
    **without replacement** to form the final sample.

    Exactly **one** of ``n_rows``, ``frac``, or ``ratio`` must be provided:

    - **Size-driven** (``n_rows``): final size is ``min(n_rows, len(df))``. All minority rows are
      included, and the remaining quota is filled by majority rows.
    - **Fraction-driven** (``frac``): final size is ``int(len(df) * frac)`` (floored), clamped to
      ``[1, len(df)]``. Minority is fully included; majority fills the remainder.
    - **Ratio-driven** (``ratio``): include ``ratio`` majority rows **per** minority row, i.e.
      majority count = ``min(ratio * n_minority, n_majority)``. Final size =
      ``n_minority + majority_taken``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str
        Binary target column name. The **minority** class is inferred as the label with the
        fewest rows.
    n_rows : int or None, optional
        Requested final sample size (absolute). Mutually exclusive with ``frac`` and ``ratio``.
        Must be a positive integer. Values larger than ``len(df)`` are clamped.
    frac : float or None, optional
        Requested final sample size as a fraction of the dataset, strictly in ``(0, 1]``.
        Mutually exclusive with ``n_rows`` and ``ratio``. Effective size is
        ``int(len(df) * frac)``.
    ratio : int or None, optional
        Majority-per-minority ratio ``R`` (e.g., ``50`` → **1:50**). Mutually exclusive with
        ``n_rows`` and ``frac``. Must be a non-negative integer.
    seed : int or None, optional
        Random seed for deterministic majority sampling and final shuffle.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing **all** minority rows plus a random subset of majority rows,
        shuffled and with a zero-based, consecutive index.

    Raises
    ------
    ValueError
        - If ``target`` is missing or not binary.
        - If **none** or **more than one** of ``n_rows``, ``frac``, ``ratio`` is provided.
        - If ``n_rows`` ≤ 0, or ``frac`` ∉ ``(0, 1]``, or ``ratio`` < 0.
        - If the requested size (in size/fraction mode) is **smaller** than the minority count.

    Notes
    -----
    - Majority sampling is performed **without replacement**.
    - When the majority quota exceeds availability, the function **caps** at the number of
      available majority rows (no error).
    - The final output is **fully shuffled** for neutrality and then index-reset.

    Examples
    --------
    Absolute size (80k rows total, keep-all-minority):
    >>> out = keep_all_minority_random_sampling(df, target="Class", n_rows=80_000, seed=42)

    Fractional size (10% of the dataset, keep-all-minority):
    >>> out = keep_all_minority_random_sampling(df, target="Class", frac=0.10, seed=42)

    Ratio mode (1:50 majority-to-minority):
    >>> out = keep_all_minority_random_sampling(df, target="Class", ratio=50, seed=42)
    """
    # Validate 'target' column and binary nature
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # value_counts() yields counts per label; require exactly two classes for binary targets
    counts = df[target].value_counts()
    if counts.size != 2:
        raise ValueError("'keep_all_minority_random_sampling' requires a binary target.")

    # Identify minority/majority labels by frequency
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    # Partition the dataframe once (used by all modes)
    df_min = df[df[target] == minority_label]
    df_maj = df[df[target] == majority_label]

    n_min = len(df_min)  # number of minority rows (must all be kept)
    n_maj = len(df_maj)  # number of majority rows available
    n_total = n_min + n_maj  # total rows in the dataset

    # Validate *exactly one* mode selector is provided
    using_n = n_rows is not None
    using_f = frac is not None
    using_r = ratio is not None
    if (using_n + using_f + using_r) != 1:
        raise ValueError("Provide exactly one of 'n_rows', 'frac', or 'ratio'.")

    # We'll compute two key quantities:
    #   - 'majority_take': how many majority rows to sample
    #   - 'sample_size' : the implied total size of the final sample
    # Both depend on the chosen mode and must respect availability and invariants.

    # RATIO-DRIVEN MODE
    if using_r:
        # Validate the ratio
        if ratio is None or ratio < 0:
            raise ValueError("'ratio' must be a non-negative integer.")

        # Compute how many majority rows are *needed* by the ratio,
        # then cap it to the available majority rows.
        majority_needed = ratio * n_min
        majority_take = min(majority_needed, n_maj)

        # Implied final size = all minority + taken majority
        sample_size = n_min + majority_take

    # SIZE-DRIVEN MODE (absolute 'n_rows')
    elif using_n:
        # Validate 'n_rows'
        if n_rows is None or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")

        # Clamp requested size to the dataset bounds
        sample_size = min(n_rows, n_total)

        # Invariant: we must be able to keep *all* minority rows
        if sample_size < n_min:
            raise ValueError(
                f"Requested size ({sample_size}) is smaller than the minority count ({n_min}); "
                "cannot keep all minority."
            )

        # Majority quota is whatever remains after placing all minority rows
        majority_take = min(sample_size - n_min, n_maj)

    # FRACTION-DRIVEN MODE ('frac')
    else:
        # Validate 'frac'
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError("frac must be in the interval (0, 1].")

        # Convert fraction to an integer size (floor), then clamp to [1, n_total]
        sample_size = int(n_total * float(frac))
        sample_size = min(max(sample_size, 1), n_total)

        # Must still be able to include all minority rows
        if sample_size < n_min:
            raise ValueError(
                f"Requested size via frac ({sample_size}) is smaller than the minority count ({n_min}); "
                "increase 'frac'."
            )

        # Majority quota: remaining capacity after placing all minority rows
        majority_take = min(sample_size - n_min, n_maj)

    # Draw the majority sample (without replacement)
    # If no majority rows are needed (e.g., ratio=0 or sample_size==n_min), we skip sampling.
    if majority_take > 0:
        df_maj_sample = df_maj.sample(n=majority_take, replace=False, random_state=seed)
        # Concatenate all minority rows with the sampled majority rows
        out = pd.concat([df_min, df_maj_sample], axis=0)
    else:
        # Corner case: final sample equals the minority set only
        out = df_min.copy()

    # Final shuffle and index reset for downstream neutrality
    # Shuffling the *entire* sample (not just majority) prevents any ordering bias.
    out = out.sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)

    return out


def apply_sampling(
    df: pd.DataFrame,
    *,
    policy: str,
    target: str = "Class",
    n_rows: int | None = None,
    frac: float | None = None,
    ratio: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Dispatch a sampling request to the selected strategy.

    This is a thin, case-sensitive router that forwards arguments to one of:
    - ``random_sampling`` (uniform, without replacement)
    - ``stratified_random_sampling`` (preserve class proportions of ``target``)
    - ``keep_all_minority_random_sampling`` (include **all** minority rows, add majority)

    Detailed validation (e.g., mutual exclusivity of ``n_rows``/``frac``/``ratio``,
    target presence, binary checks) is handled by the concrete strategy functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to sample from.
    policy : str
        Sampling policy (case-sensitive). Accepted values:
        ``'random'``, ``'stratified'``, ``'keep_all_minority'``.
    target : str, default 'Class'
        Target/stratification column. Used by ``'stratified'`` and
        ``'keep_all_minority'``; ignored by ``'random'`` (kept for a uniform API).
    n_rows : int or None, optional
        Absolute requested sample size (delegated to the concrete function).
        Mutually exclusive with ``frac`` for ``'random'`` and ``'stratified'``.
        Mutually exclusive with both ``frac`` and ``ratio`` for ``'keep_all_minority'``.
    frac : float or None, optional
        Fractional requested sample size in ``(0, 1]`` (delegated).
        Mutually exclusive with ``n_rows`` for ``'random'`` and ``'stratified'``.
        Mutually exclusive with both ``n_rows`` and ``ratio`` for
        ``'keep_all_minority'``.
    ratio : int or None, optional
        **Only for** ``'keep_all_minority'``. Majority-per-minority ratio
        (e.g., ``50`` → **1:50**). Mutually exclusive with ``n_rows`` and ``frac``.
    seed : int or None, optional
        Random seed for reproducibility (forwarded to underlying functions).

    Returns
    -------
    pandas.DataFrame
        Sampled DataFrame with rows shuffled and index reset.

    Raises
    ------
    ValueError
        If ``policy`` is unknown. (Other errors are raised by the underlying
        strategy functions when inputs are invalid.)

    Notes
    -----
    - All strategies sample **without replacement**.
    - Use ``'stratified'`` if you need to preserve the class prior in the sample.
    - Use ``'keep_all_minority'`` to guarantee all minority cases are present and
      top up with majority rows via absolute size (``n_rows``), fraction (``frac``),
      or ratio (``ratio`` majority per minority).

    Examples
    --------
    Basic setup:
    >>> import pandas as pd
    >>> rng = pd.Series(range(1000))
    >>> df = pd.DataFrame({
    ...     "x": rng,
    ...     "Class": [0]*950 + [1]*50,  # imbalanced binary target
    ... })

    1) Random sampling (absolute size):
    >>> out = apply_sampling(df, policy="random", n_rows=200, seed=42)
    >>> len(out)
    200

    2) Stratified sampling (fractional size, preserves class proportions):
    >>> out = apply_sampling(df, policy="stratified", target="Class", frac=0.2, seed=42)
    >>> len(out)
    200
    >>> out["Class"].value_counts(normalize=True).round(3).to_dict()  # roughly ~0.95 / ~0.05
    {0: ... , 1: ...}

    3) Keep-all-minority (absolute size):
    >>> out = apply_sampling(df, policy="keep_all_minority", target="Class", n_rows=300, seed=42)
    >>> out["Class"].value_counts().to_dict()  # includes all 50 minority rows
    {0: ..., 1: 50}

    4) Keep-all-minority (fractional size):
    >>> out = apply_sampling(df, policy="keep_all_minority", target="Class", frac=0.3, seed=42)
    >>> len(out)
    300

    5) Keep-all-minority (ratio mode, e.g., 1:10 majority per minority):
    >>> out = apply_sampling(df, policy="keep_all_minority", target="Class", ratio=10, seed=42)
    >>> out["Class"].value_counts().to_dict()  # ~ 50 minority + min(50*10, 950) majority
    {0: ..., 1: 50}
    """
    if policy == "random":
        return random_sampling(df, n_rows=n_rows, frac=frac, seed=seed)

    if policy == "stratified":
        return stratified_random_sampling(
            df, stratify_by=target, n_rows=n_rows, frac=frac, seed=seed
        )

    if policy == "keep_all_minority":
        return keep_all_minority_random_sampling(
            df, target=target, n_rows=n_rows, frac=frac, ratio=ratio, seed=seed
        )

    raise ValueError(f"Unknown policy '{policy}'.")


def get_resampling_pipeline(
    strategy_name: Optional[str],
    **kwargs: Any,
) -> Union[BaseSampler, str]:
    """
    Return an **imblearn sampler instance** for the chosen resampling strategy
    (not a Pipeline). Use this directly inside your pipeline as a step.

    Accepted canonical names (case-sensitive)
    ------------------------------------------------
    Undersampling:
      'RandomUnderSampler', 'NearMiss', 'TomekLinks', 'EditedNearestNeighbours',
      'RepeatedEditedNearestNeighbours', 'AllKNN', 'CondensedNearestNeighbour',
      'OneSidedSelection', 'NeighbourhoodCleaningRule', 'InstanceHardnessThreshold',
      'ClusterCentroids'
    Oversampling:
      'RandomOverSampler', 'SMOTE', 'SMOTENC', 'SMOTEN', 'ADASYN',
      'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE'
    Hybrid:
      'SMOTEENN', 'SMOTETomek'
    Passthrough:
      'none' (or ``None``) → returns the string ``'passthrough'``

    Parameters
    ----------
    strategy_name : str or None
        Canonical sampler class name, case-sensitive (e.g., 'SMOTE').
        If None or 'none', returns 'passthrough'.
    **kwargs : dict
        Arguments forwarded to the sampler constructor.
        Examples:
          sampling_strategy=0.01, random_state=42, k_neighbors=5 (SMOTE/ADASYN/SVMSMOTE),
          version=1/2/3 (NearMiss), smote=SMOTE(...), enn=EditedNearestNeighbours(...) (SMOTEENN),
          tomek=TomekLinks(...) (SMOTETomek).

    Returns
    -------
    BaseSampler or 'passthrough'
        The configured sampler instance, or the literal string 'passthrough'.

    Raises
    ------
    ValueError
        If `strategy_name` is not supported.

    Examples
    --------
    >>> from imblearn.pipeline import Pipeline as ImbPipeline
    >>> sampler = get_resampling_pipeline("SMOTE", sampling_strategy='auto', k_neighbors=5, random_state=42)
    >>> pipe = ImbPipeline([("resample", sampler), ("clf", ...)])

    >>> sampler = get_resampling_pipeline("none")
    >>> pipe = ImbPipeline([("resample", sampler), ("clf", ...)])  # resample step is a passthrough
    """
    name = "none" if strategy_name is None else strategy_name

    registry: Dict[str, Type] = {
        # Undersampling
        "RandomUnderSampler": RandomUnderSampler,
        "NearMiss": NearMiss,
        "TomekLinks": TomekLinks,
        "EditedNearestNeighbours": EditedNearestNeighbours,
        "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
        "AllKNN": AllKNN,
        "CondensedNearestNeighbour": CondensedNearestNeighbour,
        "OneSidedSelection": OneSidedSelection,
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
        "InstanceHardnessThreshold": InstanceHardnessThreshold,
        "ClusterCentroids": ClusterCentroids,
        # Oversampling
        "RandomOverSampler": RandomOverSampler,
        "SMOTE": SMOTE,
        "SMOTENC": SMOTENC,
        "SMOTEN": SMOTEN,
        "ADASYN": ADASYN,
        "BorderlineSMOTE": BorderlineSMOTE,
        "KMeansSMOTE": KMeansSMOTE,
        "SVMSMOTE": SVMSMOTE,
        # Hybrid
        "SMOTEENN": SMOTEENN,
        "SMOTETomek": SMOTETomek,
    }

    if name in {"none", "passthrough"}:
        return "passthrough"

    if name not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown resampler '{strategy_name}'. Supported: {supported}")

    sampler_cls = registry[name]
    return sampler_cls(**kwargs)
