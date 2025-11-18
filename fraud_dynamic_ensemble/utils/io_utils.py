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
    Save a dictionary to a JSON file, either overwriting or appending.

    This function does **not** create parent directories. If the destination
    directory does not exist, a ``FileNotFoundError`` is raised.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary-like object to serialize.
    path : str or pathlib.Path
        Destination file path (parent directory must already exist).
    mode : {"w", "a"}, default "w"
        Write mode:
        - "w": overwrite with a single pretty-printed JSON object.
        - "a": append one compact JSON object per line (JSON Lines / NDJSON).
    ensure_ascii : bool, default False
        If False, write UTF-8 characters as-is; if True, escape non-ASCII.
    sort_keys : bool, default False
        Sort dictionary keys in the output.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``mode`` is not one of {"w", "a"}.
    FileNotFoundError
        If the parent directory of ``path`` does not exist.

    Notes
    -----
    - Appending uses **JSON Lines** format (one JSON object per line), which is
      not a single valid JSON document. Use tools that support NDJSON.

    Examples
    --------
    Overwrite with pretty JSON:
    >>> save_dict_json({"run_id": 1, "score": 0.92}, "reports/metrics/run_1.json")

    Append as JSON Lines:
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
