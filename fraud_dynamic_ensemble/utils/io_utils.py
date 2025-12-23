from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Union

PathLike = Union[str, Path]


def save_dict_json(
    data: Mapping[str, Any],
    path: PathLike,
    *,
    mode: Literal["w", "a"] = "w",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    Serialize a mapping to JSON on disk (overwrite or JSON Lines append).

    In write mode (``mode='w'``), this function writes a single pretty-printed JSON
    object. In append mode (``mode='a'``), it appends one compact JSON object per
    line (NDJSON/JSON Lines). Parent directories are not created.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary-like object to serialize.
    path : str or pathlib.Path
        Destination file path. The parent directory must already exist.
    mode : {'w', 'a'}, default 'w'
        Output mode. If ``'w'``, overwrite with a single pretty-printed JSON object.
        If ``'a'``, append one JSON object per line (JSON Lines / NDJSON).
    ensure_ascii : bool, default False
        If ``False``, write UTF-8 characters as-is. If ``True``, escape non-ASCII
        characters.
    sort_keys : bool, default False
        If ``True``, sort dictionary keys in the output.

    Returns
    -------
    None
        This function writes to disk and returns nothing.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{'w', 'a'}``.
    FileNotFoundError
        If the parent directory of ``path`` does not exist.

    Notes
    -----
    - Append mode writes JSON Lines (NDJSON), which is not a single valid JSON
      document. Use tools/readers that support line-delimited JSON.
    - This function does not create parent directories.

    Examples
    --------
    >>> save_dict_json({"run_id": 1, "score": 0.92}, "reports/metrics/run_1.json")

    >>> save_dict_json({"fold": 0, "ap": 0.88}, "reports/metrics/log.jsonl", mode="a")
    """

    if mode not in {"w", "a"}:
        raise ValueError('mode must be either "w" (overwrite) or "a" (append).')

    if mode == "w":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, sort_keys=sort_keys, indent=2)
            f.write("\n")
    else:  # "a" → JSON Lines
        with path.open("a", encoding="utf-8") as f:
            line = json.dumps(
                data,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                separators=(",", ":"),  # compact for one-line records
            )
            f.write(line + "\n")
