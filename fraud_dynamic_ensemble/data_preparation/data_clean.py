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
    Remove duplicate rows from a DataFrame.

    This is a thin wrapper around :meth:`pandas.DataFrame.drop_duplicates` that
    branches on ``inplace`` to pass literal ``True``/``False``. This pattern is
    intended to avoid strict type-stub warnings in some tooling without changing
    pandas runtime behavior.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    subset : hashable or sequence of hashable, optional
        Column label(s) to consider when identifying duplicates. If ``None``,
        all columns are used.
    keep : {'first', 'last', False}, default 'first'
        Which duplicates to keep.
        If ``'first'``, keep the first occurrence.
        If ``'last'``, keep the last occurrence.
        If ``False``, drop all duplicates (keep none).
    inplace : bool, default False
        If ``True``, modify ``df`` in place and return ``None``.
        If ``False``, return a new DataFrame and leave ``df`` unchanged.
    ignore_index : bool, default False
        If ``True``, reset the index to ``RangeIndex(0, ..., n-1)`` in the
        returned object. When ``inplace=True``, this resets the index of ``df``
        itself.

    Returns
    -------
    pandas.DataFrame or None
        Deduplicated DataFrame if ``inplace=False``; otherwise ``None``.

    Raises
    ------
    KeyError
        If any label in ``subset`` is not a column of ``df``.

    Notes
    -----
    This function delegates to :meth:`pandas.DataFrame.drop_duplicates`. The
    explicit branching on ``inplace`` is intentional to satisfy strict pandas
    type stubs.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
    >>> remove_duplicates(df).shape[0]
    2

    >>> _ = remove_duplicates(df, inplace=True, ignore_index=True)
    >>> df.index[0]
    0
    """

    if inplace:
        df.drop_duplicates(subset=subset, keep=keep, inplace=True, ignore_index=ignore_index)
        return None
    else:
        return df.drop_duplicates(
            subset=subset, keep=keep, inplace=False, ignore_index=ignore_index
        )
