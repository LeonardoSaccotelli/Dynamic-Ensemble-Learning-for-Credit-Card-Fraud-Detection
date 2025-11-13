from __future__ import annotations

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def transform_log1p(df, cols, drop_original=True):
    """
    Apply a row-wise ``log1p`` transform (``log(x + 1)``) to one or more columns.

    This helper wraps a scikit-learn ``FunctionTransformer`` (with ``np.log1p``) and
    writes results to new columns named ``<col>_log1p``. By default the original
    columns are dropped to keep the final schema compact.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. A defensive copy is created; the original is not modified.
    cols : str or list of str
        Column name(s) to transform.
    drop_original : bool, default True
        If True, drop the source columns after creating the ``*_log1p`` columns.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the transformed columns appended (and optionally with
        the source columns removed).

    Raises
    ------
    KeyError
        If any requested column is not present in ``df``.
    TypeError
        If the column data cannot be coerced to a numeric dtype acceptable by
        ``numpy.log1p``.
    ValueError
        If downstream assignment fails due to incompatible shapes/dtypes.

    Notes
    -----
    - This is a purely row-wise transform (no learned statistics) → **no leakage**.
    - ``np.log1p`` is defined for ``x >= -1``; values ``< -1`` will yield ``NaN`` and
      may trigger a runtime warning. Validate or clip upstream if necessary.
    - Internally, a 2D view of each column (``df[[col]]``) is passed to the
      transformer; the result is assigned back to a single new column.

    Examples
    --------
    Single column (drop original):
    >>> out = transform_log1p(df, "Amount")

    Multiple columns, keep originals:
    >>> out = transform_log1p(df, ["A", "B"], drop_original=False)
    >>> set(out.columns) >= {"A", "B", "A_log1p", "B_log1p"}
    True
    """

    def log1p_transformer():
        # np.log1p is applied element-wise, so it's directly compatible
        return FunctionTransformer(np.log1p)

    # Create a copy to avoid SettingWithCopy warnings
    df_transformed = df.copy()

    # Ensure cols is a list even if a single string is passed
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        # Define the new column name (e.g., "Amount_log")
        new_col_name = f"{col}_log1p"

        # Apply the transformation
        # We use double brackets [[col]] to ensure 2D input for fit_transform
        df_transformed[new_col_name] = log1p_transformer().fit_transform(df_transformed[[col]])

    # Drop original columns if requested
    if drop_original:
        df_transformed.drop(columns=cols, inplace=True)

    return df_transformed


def transform_sin_cos(df, cols, period, drop_original=True):
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
    TypeError
        If the column data cannot be coerced to a numeric dtype acceptable by
        ``numpy`` trigonometric functions.
    ValueError
        If ``period`` is not positive or assignment fails due to shape issues.

    Notes
    -----
    - Purely row-wise (no fit/learned stats) → **no leakage**.
    - Ensure ``period`` strictly matches the unit of the input feature (e.g.,
      seconds vs hours). If your feature has an offset (e.g., starts at 1), apply
      the offset upstream before calling this function.
    - Internally, each transformer receives a 2D view (``df[[col]]``) and returns
      a 2D array which is assigned to a single new column.

    Examples
    --------
    Seconds in day (keep original):
    >>> out = transform_sin_cos(df, "Time", period=86400, drop_original=False)
    >>> set(out.columns) >= {"Time", "Time_sin", "Time_cos"}
    True

    Multiple columns (drop originals):
    >>> out = transform_sin_cos(df, ["t1", "t2"], period=24)
    >>> set(out.columns) >= {"t1_sin", "t1_cos", "t2_sin", "t2_cos"}
    True
    """

    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    # Create a copy to avoid SettingWithCopy warnings
    df_transformed = df.copy()

    # Ensure cols is a list even if a single string is passed
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        # Define new column names automatically (e.g. "hour_sin", "hour_cos")
        sin_col_name = f"{col}_sin"
        cos_col_name = f"{col}_cos"

        # Apply transformations exactly as requested
        # Note: We use double brackets [[col]] to ensure 2D input for fit_transform
        df_transformed[sin_col_name] = sin_transformer(period).fit_transform(df_transformed[[col]])
        df_transformed[cos_col_name] = cos_transformer(period).fit_transform(df_transformed[[col]])

    # Drop original columns if requested
    if drop_original:
        df_transformed.drop(columns=cols, inplace=True)

    return df_transformed
