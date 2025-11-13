from __future__ import annotations

from typing import Hashable, Literal, Sequence

import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
    subset: Hashable | Sequence[Hashable] | None = None,
    keep: Literal["first", "last", False] = "first",
    *,
    inplace: bool = False,
    ignore_index: bool = False,
) -> pd.DataFrame | None:
    """
    Drop duplicate rows (thin wrapper around ``pandas.DataFrame.drop_duplicates``).

    This utility mirrors pandas' behavior while branching internally to pass
    literal ``True``/``False`` for the ``inplace`` argument. That avoids
    strict type-stub warnings in some tooling without changing pandas'
    runtime behavior.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    subset : hashable or sequence of hashable, optional
        Columns to consider when identifying duplicates. If ``None``, use all columns.
    keep : {'first', 'last', False}, default 'first'
        Which duplicates to keep:
        - ``'first'`` keeps the first occurrence and drops subsequent ones.
        - ``'last'`` keeps the last occurrence and drops previous ones.
        - ``False`` drops **all** duplicates (keeps none).
    inplace : bool, default False
        If ``True``, modify ``df`` in place and return ``None``.
        If ``False``, return a new DataFrame and leave ``df`` unchanged.
    ignore_index : bool, default False
        If ``True``, the result's index is reset to ``RangeIndex(0, …, n-1)``.
        When ``inplace=True``, this resets the index of ``df`` itself.

    Returns
    -------
    pandas.DataFrame or None
        A new DataFrame with duplicates removed if ``inplace=False``; otherwise ``None``.

    Raises
    ------
    KeyError
        If any label in ``subset`` is not a column of ``df``.

    Notes
    -----
    - This function delegates to ``DataFrame.drop_duplicates``; see pandas docs
      for additional details and performance considerations.
    - Branching with literal booleans (``inplace=True/False``) is intentional to
      satisfy strict pandas type stubs and silence static checker warnings.

    Examples
    --------
    Basic usage, return a new DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 1, 2], 'B': ['x', 'x', 'y']})
    >>> out = remove_duplicates(df)
    >>> len(out)
    2

    Consider only a subset of columns:

    >>> remove_duplicates(df, subset='A').shape[0]
    2

    Drop all duplicates (keep none):

    >>> remove_duplicates(df, keep=False).shape[0]
    2

    In place with index reset:

    >>> _ = remove_duplicates(df, inplace=True, ignore_index=True)
    >>> df.index[0] == 0
    True
    """
    if inplace:
        df.drop_duplicates(subset=subset, keep=keep, inplace=True, ignore_index=ignore_index)
        return None
    else:
        return df.drop_duplicates(
            subset=subset, keep=keep, inplace=False, ignore_index=ignore_index
        )
