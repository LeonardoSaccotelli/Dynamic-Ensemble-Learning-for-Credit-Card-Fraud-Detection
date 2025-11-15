from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def transform_log1p(
    df: pd.DataFrame,
    cols: str | list[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Apply a row-wise ``log1p`` transform (``log(x + 1)``) to one or more columns.

    This helper uses a scikit-learn ``FunctionTransformer`` (``np.log1p``) and writes
    results to new columns named ``<col>_log1p``. Optionally drops the original columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. A copy is returned; the original is not modified.
    cols : str or list of str
        Column name(s) to transform.
    drop_original : bool, default True
        If True, drop the source columns after creating the ``*_log1p`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with transformed columns appended (and optionally source
        columns removed).

    Raises
    ------
    KeyError
        If any requested column is not present in ``df``.
    ValueError
        If any selected column contains values strictly less than ``-1``
        (invalid for ``log1p``).

    Notes
    -----
    - Purely row-wise transform → **no data leakage**.
    - If a column has non-numeric data, underlying casting/transform will error.
    - ``np.log1p`` is defined for ``x >= -1``.

    Examples
    --------
    >>> out = transform_log1p(df, "Amount")
    >>> out = transform_log1p(df, ["A", "B"], drop_original=False)
    """

    def _log1p_transformer() -> FunctionTransformer:
        return FunctionTransformer(np.log1p)

    df_transformed = df.copy()

    # Normalize cols and check presence
    cols_list: list[str] = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in cols_list if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # Domain check and transform
    transformer = _log1p_transformer()
    for col in cols_list:
        arr = df_transformed[col].astype(float).to_numpy()
        finite = np.isfinite(arr)
        if finite.any() and np.nanmin(arr[finite]) < -1.0:
            raise ValueError(f"Column '{col}' contains values < -1; invalid for log1p.")
        new_col = f"{col}_log1p"
        df_transformed[new_col] = transformer.fit_transform(df_transformed[[col]])

    if drop_original:
        df_transformed.drop(columns=cols_list, inplace=True)

    return df_transformed


def transform_sin_cos(
    df: pd.DataFrame,
    cols: str | list[str],
    period: int | float,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Encode cyclical features with sine and cosine components.

    For each column in ``cols``, compute:
    ``sin = sin(2π * x / period)`` and ``cos = cos(2π * x / period)``,
    creating new columns ``<col>_sin`` and ``<col>_cos``. By default, the original
    columns are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. A defensive copy is created; the original is not modified.
    cols : str or list of str
        Column name(s) to transform (numeric, cyclical values).
    period : int or float
        The periodicity of the variable (e.g., ``24`` for hours-of-day,
        ``86400`` for seconds-in-day). Units must match those of ``cols``.
    drop_original : bool, default True
        If True, drop the source columns after creating the sine/cosine columns.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``*_sin`` and ``*_cos`` columns appended (and
        optionally with the source columns removed).

    Raises
    ------
    KeyError
        If any requested column is not present in ``df``.
    ValueError
        If ``period`` is not strictly positive.
    TypeError
        Propagated if column data cannot be coerced to numeric for the trig functions.

    Notes
    -----
    - Purely row-wise transform (no learned statistics) → **no leakage**.
    - Ensure ``period`` matches the unit of the source feature (e.g., seconds vs hours).
    - Internally, a 2D view (``df[[col]]``) is passed to scikit-learn's
      ``FunctionTransformer`` and assigned back to a single new column.

    Examples
    --------
    Keep original ``Time`` (seconds in day):
    >>> out = transform_sin_cos(df, "Time", period=86400, drop_original=False)
    >>> {"Time", "Time_sin", "Time_cos"}.issubset(out.columns)
    True

    Multiple columns, drop originals:
    >>> out = transform_sin_cos(df, ["t1", "t2"], period=24)
    >>> {"t1_sin", "t1_cos", "t2_sin", "t2_cos"}.issubset(out.columns)
    True
    """
    if period <= 0:
        raise ValueError("'period' must be a strictly positive number.")

    def _sin_transformer(p: float | int) -> FunctionTransformer:
        return FunctionTransformer(lambda x: np.sin(x / p * 2 * np.pi))

    def _cos_transformer(p: float | int) -> FunctionTransformer:
        return FunctionTransformer(lambda x: np.cos(x / p * 2 * np.pi))

    df_transformed = df.copy()

    cols_list: list[str] = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in cols_list if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    sin_tf = _sin_transformer(period)
    cos_tf = _cos_transformer(period)

    for col in cols_list:
        sin_col_name = f"{col}_sin"
        cos_col_name = f"{col}_cos"
        # Pass as 2D for sklearn; assignment back to a 1D column is handled by pandas.
        df_transformed[sin_col_name] = sin_tf.fit_transform(df_transformed[[col]])
        df_transformed[cos_col_name] = cos_tf.fit_transform(df_transformed[[col]])

    if drop_original:
        df_transformed.drop(columns=cols_list, inplace=True)

    return df_transformed


def get_standard_scaler(
    columns: Union[Sequence[int], Sequence[str]],
    *,
    remainder: Literal["passthrough", "drop"] = "passthrough",
    with_mean: bool = True,
    with_std: bool = True,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that applies StandardScaler to selected columns.

    This is useful when you want to scale only a subset of features—especially
    **by positional index** after steps like FeatureUnion, where column names
    may be lost and the data is a NumPy array.

    Parameters
    ----------
    columns : sequence of int or str
        Columns to scale. Use **indices** when the input to the transformer is
        a NumPy array; use **names** when it is a pandas DataFrame.
    remainder : {"passthrough", "drop"}, default "passthrough"
        What to do with columns not listed in ``columns``:
        - "passthrough": keep them unchanged,
        - "drop": remove them.
    with_mean : bool, default True
        Center the data before scaling (StandardScaler parameter).
        Set to False if using sparse inputs.
    with_std : bool, default True
        Scale to unit variance (StandardScaler parameter).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A transformer that applies standardization to the specified columns and
        passes/drops the remainder according to ``remainder``.

    Notes
    -----
    - When your upstream pipeline returns a NumPy array (e.g., after feature
      selection or unions), prefer **index-based** selection.
    - If you later switch to chi-square tests, remember that ``chi2`` requires
      **non-negative** features, so StandardScaler would be inappropriate there.

    Examples
    --------
    Scale columns by **index** (NumPy input):
    >>> ct = get_standard_scaler(columns=[0, 1, 3], remainder="passthrough")

    Scale columns by **name** (DataFrame input):
    >>> ct = get_standard_scaler(columns=["Amount_log1p", "V1"], remainder="passthrough")

    Use inside a Pipeline:
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ("scale_selected", ct),
    ...     # ("select", get_feature_selection(...)),  # optional
    ...     # ("clf", SomeEstimator(...)),
    ... ])
    """
    if not columns:
        raise ValueError("`columns` must be a non-empty sequence of indices or names.")

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    preprocessor = ColumnTransformer(
        transformers=[("scaler", scaler, list(columns))],
        remainder=remainder,
    )
    return preprocessor
