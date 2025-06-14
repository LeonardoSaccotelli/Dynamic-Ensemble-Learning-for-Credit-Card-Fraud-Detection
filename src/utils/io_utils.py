from pathlib import Path
import pandas as pd


def load_csv(
    path: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8",
    na_values: str | list[str] | None = None,
    dtype: dict | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    delimiter : str, optional
        Field delimiter for the CSV file. Default is ",".
    encoding : str, optional
        Character encoding. Default is "utf-8".
    na_values : str or list of str, optional
        Additional strings to recognize as NA/NaN.
    dtype : dict, optional
        Dictionary of column types to enforce.
    **kwargs : dict
        Additional keyword arguments passed to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    return pd.read_csv(
        path,
        delimiter=delimiter,
        encoding=encoding,
        na_values=na_values,
        dtype=dtype,
        **kwargs
    )


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8",
    index: bool = False,
    float_format: str | None = None,
    **kwargs
) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination file path.
    delimiter : str, optional
        Field delimiter. Default is ",".
    encoding : str, optional
        File encoding. Default is "utf-8".
    index : bool, optional
        Whether to write row indices. Default is False.
    float_format : str, optional
        Format string for floats, e.g. "%.3f".
    **kwargs : dict
        Additional keyword arguments passed to `pd.DataFrame.to_csv`.

    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        index=index,
        float_format=float_format,
        **kwargs
    )
