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
    Apply a ``log1p`` transform to one or more DataFrame columns.

    This function computes ``log(x + 1)`` (via a scikit-learn
    :class:`sklearn.preprocessing.FunctionTransformer`) for each selected column
    and appends the result as a new column named ``<col>_log1p``. Optionally, the
    original source columns are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. The original is not modified; a copy is returned.
    cols : str or list of str
        Column name(s) to transform.
    drop_original : bool, default True
        If ``True``, drop the source columns after creating the ``*_log1p`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the transformed ``*_log1p`` columns appended and,
        if requested, the original columns removed.

    Raises
    ------
    KeyError
        If any requested column is not present in ``df``.
    ValueError
        If any selected column contains finite values strictly less than ``-1``
        (invalid domain for ``log1p``).

    Notes
    -----
    - This is a purely row-wise transform and does not introduce data leakage.
    - ``np.log1p`` is defined for inputs ``x >= -1``; values below ``-1`` are rejected.
    - Non-numeric column values must be castable to ``float``; otherwise the
      conversion will raise.

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
    Encode cyclical features using sine and cosine components.

    For each column in ``cols``, this function computes the cyclical embedding
    ``sin(2π * x / period)`` and ``cos(2π * x / period)``, writing results to new
    columns named ``<col>_sin`` and ``<col>_cos``. Optionally, the original source
    columns are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. The original is not modified; a copy is returned.
    cols : str or list of str
        Column name(s) to transform (numeric cyclical values).
    period : int or float
        Periodicity of the variable (e.g., ``24`` for hours-of-day, ``86400`` for
        seconds-in-day). Units must match those of ``cols``. Must be strictly positive.
    drop_original : bool, default True
        If ``True``, drop the source columns after creating the sine/cosine columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``*_sin`` and ``*_cos`` columns appended and, if
        requested, the original columns removed.

    Raises
    ------
    KeyError
        If any requested column is not present in ``df``.
    ValueError
        If ``period`` is not strictly positive.

    Notes
    -----
    - This is a purely row-wise transform and does not introduce data leakage.
    - Ensure ``period`` matches the unit of the source feature (e.g., seconds vs hours).
    - Internally, a 2D view (``df[[col]]``) is passed to scikit-learn
      :class:`sklearn.preprocessing.FunctionTransformer` and assigned back to a
      single new column.

    Examples
    --------
    >>> out = transform_sin_cos(df, "Time", period=86400, drop_original=False)
    >>> {"Time", "Time_sin", "Time_cos"}.issubset(out.columns)
    True

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
    Build a ColumnTransformer that standardizes a selected set of columns.

    This helper creates a :class:`sklearn.compose.ColumnTransformer` that applies
    :class:`sklearn.preprocessing.StandardScaler` to the specified columns while
    either passing through or dropping all remaining columns. It is particularly
    useful when selecting columns by positional index after pipeline steps that
    output NumPy arrays (e.g., unions or feature selection).

    Parameters
    ----------
    columns : sequence of int or sequence of str
        Columns to standardize. Use integer indices when the transformer input is a
        NumPy array; use string names when the input is a pandas DataFrame.
    remainder : {'passthrough', 'drop'}, default 'passthrough'
        Strategy for columns not listed in ``columns``. If ``'passthrough'``, keep
        them unchanged. If ``'drop'``, remove them.
    with_mean : bool, default True
        Whether to center the data before scaling (StandardScaler parameter).
        Set to ``False`` when working with sparse inputs.
    with_std : bool, default True
        Whether to scale features to unit variance (StandardScaler parameter).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        ColumnTransformer applying StandardScaler to ``columns`` and handling the
        remainder according to ``remainder``.

    Raises
    ------
    ValueError
        If ``columns`` is empty.

    Notes
    -----
    - When upstream steps return a NumPy array, prefer index-based selection.
    - Some feature selection methods (e.g., ``chi2``) require non-negative features;
      in those cases StandardScaler may be inappropriate.

    Examples
    --------
    >>> ct = get_standard_scaler(columns=[0, 1, 3], remainder="passthrough")
    >>> ct = get_standard_scaler(columns=["Amount_log1p", "V1"], remainder="passthrough")
    """

    if not columns:
        raise ValueError("`columns` must be a non-empty sequence of indices or names.")

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    preprocessor = ColumnTransformer(
        transformers=[("scaler", scaler, list(columns))],
        remainder=remainder,
    )
    return preprocessor
