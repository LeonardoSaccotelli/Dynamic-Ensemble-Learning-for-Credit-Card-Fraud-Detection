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
    Load a CSV file into a pandas DataFrame with configurable parsing options.

    This is a light wrapper around :func:`pandas.read_csv` that first validates the
    file path and then forwards common options (delimiter, encoding, NA markers,
    and column dtypes) together with any additional keyword arguments.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file to be loaded.
    delimiter : str, default=","
        Field delimiter used in the file. This is passed to ``pd.read_csv`` as
        the ``delimiter``/``sep`` argument.
    encoding : str, default="utf-8"
        Text encoding used to decode the file.
    na_values : str | list[str] | None, optional
        Additional string(s) to recognize as NA/NaN in addition to the defaults
        used by pandas (e.g., ``"NA"``, ``"null"``, ``"?"``).
    dtype : dict | None, optional
        Mapping of column names to dtypes to enforce during parsing
        (e.g., ``{"id": "Int64", "date": "string"}``).
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`
        (e.g., ``usecols``, ``parse_dates``, ``nrows``, ``skiprows``, ``engine``).

    Returns
    -------
    pd.DataFrame
        The parsed tabular data.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    UnicodeDecodeError
        If the file cannot be decoded with the specified ``encoding``.
    pandas.errors.EmptyDataError
        If no columns to parse are found (empty file).
    pandas.errors.ParserError
        If a parsing error occurs (malformed CSV).
    ValueError
        If provided arguments (e.g., ``dtype`` mapping) are inconsistent.

    Examples
    --------
    Load a UTF-8 CSV with custom NA markers:

    >>> df = load_csv("data/measurements.csv", na_values=["NA", "null", "?"])
    >>> df.shape
    (1000, 12)

    Load a pipe-delimited file, enforcing dtypes and selecting columns:

    >>> df = load_csv(
    ...     "data/logs.txt",
    ...     delimiter="|",
    ...     dtype={"user_id": "Int64", "status": "category"},
    ...     usecols=["timestamp", "user_id", "status"],
    ...     parse_dates=["timestamp"]
    ... )
    >>> df.dtypes["status"].name
    'category'
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


def save_dataframe_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8",
    index: bool = False,
    float_format: str | None = None,
    **kwargs
) -> None:
    """
    Save a pandas DataFrame to a CSV file with configurable formatting.

    This thin wrapper around :meth:`pandas.DataFrame.to_csv` ensures that the
    destination directory exists (creating it if necessary) and then writes
    ``df`` to disk. Common options (delimiter, encoding, index handling, and
    float formatting) are exposed directly, while any additional keyword
    arguments are forwarded to :meth:`pandas.DataFrame.to_csv`.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    path : str | Path
        Destination file path for the CSV output. Parent directories are created
        automatically if they do not exist.
    delimiter : str, default=","
        Field delimiter used in the output file (passed as ``sep`` to ``to_csv``).
    encoding : str, default="utf-8"
        Text encoding used to write the file.
    index : bool, default=False
        Whether to write the row index.
    float_format : str | None, optional
        Format string for floating-point numbers (e.g., ``"%.3f"``).
    **kwargs
        Additional arguments forwarded to :meth:`pandas.DataFrame.to_csv`
        (e.g., ``columns``, ``na_rep``, ``date_format``, ``compression``, ``mode``).

    Returns
    -------
    None
        Writes the CSV file to disk and returns nothing.

    Raises
    ------
    PermissionError
        If the file cannot be written due to insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.
    ValueError
        If an unexpected error occurs; the original exception is chained and
        included for context.

    Examples
    --------
    Basic usage:

    >>> save_dataframe_to_csv(df, "outputs/results.csv")

    Use a semicolon as delimiter and include the index:

    >>> save_dataframe_to_csv(df, "outputs/results_sc.csv", delimiter=";", index=True)

    Control float precision:

    >>> save_dataframe_to_csv(df, "outputs/metrics.csv", float_format="%.4f")

    Pass extra arguments to ``to_csv`` (e.g., compression and date formatting):

    >>> save_dataframe_to_csv(
    ...     df,
    ...     "outputs/results.csv.gz",
    ...     compression="gzip",
    ...     date_format="%Y-%m-%d"
    ... )
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            index=index,
            float_format=float_format,
            **kwargs
        )

    except (PermissionError, OSError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to save CSV file to {path}: {e}") from e


def save_dataframe_to_excel(
    df: pd.DataFrame,
    path: str | Path,
    sheet_name: str = "Sheet1",
    **kwargs
) -> None:
    """
    Save a pandas DataFrame to an Excel workbook.

    This is a thin wrapper around :meth:`pandas.DataFrame.to_excel`. It ensures
    the parent directory of ``path`` exists (creating it if necessary) and then
    writes ``df`` to the specified Excel file, placing the data on the given
    worksheet name and omitting the index column by default.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be written to the Excel file.
    path : str | Path
        Destination file path for the Excel output (e.g., ``"reports/metrics/results.xlsx"``).
        Parent directories are created if they do not exist.
    sheet_name : str, default="Sheet1"
        Name of the worksheet to write.
    **kwargs
        Additional keyword arguments forwarded to :meth:`pandas.DataFrame.to_excel`
        (e.g., ``engine='openpyxl'``, ``na_rep='NA'``, ``float_format='%.4f'``,
        ``date_format='%Y-%m-%d'``, ``columns=[...]``, ``freeze_panes=(1, 0)``).

    Returns
    -------
    None
        Writes the Excel file to disk and returns nothing.

    Raises
    ------
    PermissionError
        If the file cannot be written due to insufficient permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.
    ValueError
        If writing fails for other reasons; the original exception is wrapped and
        chained for context.

    Examples
    --------
    Basic usage:

    >>> save_dataframe_to_excel(df, "reports/metrics.xlsx")

    Specify a custom sheet name and export only selected columns:

    >>> save_dataframe_to_excel(
    ...     df,
    ...     "reports/metrics.xlsx",
    ...     sheet_name="test_metrics",
    ...     columns=["model", "rmse", "mae"]
    ... )

    Control NA and float formatting, and freeze the header row:

    >>> save_dataframe_to_excel(
    ...     df,
    ...     "reports/metrics.xlsx",
    ...     na_rep="NA",
    ...     float_format="%.3f",
    ...     freeze_panes=(1, 0)
    ... )
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, sheet_name=sheet_name, index=False, **kwargs)
    except (PermissionError, OSError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to save EXCEL file to {path}: {e}") from e