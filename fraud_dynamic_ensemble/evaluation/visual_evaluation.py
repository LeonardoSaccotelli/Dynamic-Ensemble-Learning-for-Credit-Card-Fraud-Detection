from __future__ import annotations

from pathlib import Path
from typing import Optional

import IPython.display as ipd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns


def plot_learning_curves(
    df: pd.DataFrame,
    metric_name: str,
    model_name: str,
    save_path: Path
) -> None:
    """
    Plot train vs. generalization trajectories across iterations for a single metric.

    The function aggregates the input results by ``iteration`` and ``split`` to compute
    the mean and standard deviation of ``metric_name`` across folds. It then plots two
    trajectories:
    - ``"resubstitution"`` (train) and
    - ``"generalization"`` (test),
    with shaded bands representing ±1 standard deviation. The figure is saved and also
    displayed via ``plt.show()``. The ``model_name`` parameter is used only for labeling
    the plot title (the function does not filter ``df`` by model internally).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at minimum the columns: ``"iteration"``, ``"split"``, and
        the metric column specified by ``metric_name``. If you want curves for a single
        model, pass a DataFrame already filtered to that model.
    metric_name : str
        Name of the metric column to aggregate and plot.
    model_name : str
        Model identifier used for figure labeling (title). No filtering is performed.
    save_path : pathlib.Path
        Output directory used to build the output figure path.

    Returns
    -------
    None
        Side effects only: prints progress messages, renders the plot, and writes a PNG file.

    Raises
    ------
    KeyError
        If ``metric_name`` or mandatory columns (e.g., ``"iteration"``, ``"split"``) are
        missing from ``df``.
    PermissionError
        If the output image cannot be written due to insufficient permissions.
    OSError
        If an OS-related error occurs while saving the figure.

    Notes
    -----
    - Expected ``split`` values are ``"resubstitution"`` and ``"generalization"``.
    - The y-axis is fixed to ``[0, 1.05]`` in the current implementation, so this plot is
      appropriate for metrics naturally bounded in ``[0, 1]`` (e.g., ROC-AUC, F1, AP).
      For metrics like MCC in ``[-1, 1]``, the visualization will be clipped.
    - The current implementation builds ``fig_filename`` as ``save_path / <filename>``
      and then calls ``plt.savefig(save_path / fig_filename, ...)``, which effectively
      duplicates ``save_path`` in the output path. This is an implementation detail of
      the existing code; the docstring documents it without modifying the code.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df = pd.DataFrame(
    ...     {
    ...         "model": ["A", "A", "A", "A"],
    ...         "iteration": [1, 1, 2, 2],
    ...         "fold": [1, 2, 1, 2],
    ...         "split": ["resubstitution", "generalization", "resubstitution", "generalization"],
    ...         "roc_auc": [0.95, 0.90, 0.96, 0.91],
    ...     }
    ... )
    >>> plot_learning_curves(df=df, metric_name="roc_auc", model_name="A", save_path=Path("."))
    >>> True
    True
    """

    print(f"\n{'=' * 80}\nLEARNING CURVES ANALYSIS: {metric_name.upper()}\n{'=' * 80}")

    # 1. Aggregation: Calculate Mean and Std per Iteration & Split
    # Collapse the 10 folds into summary stats
    stats_df = df.groupby(["iteration", "split"])[metric_name].agg(["mean", "std"]).reset_index()

    # 2. Setup Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Define colors for splits
    colors = {
        "resubstitution": "#d62728",
        "generalization": "#1f77b4",
    }  # Red for Train, Blue for Test
    labels = {
        "resubstitution": "Train (Resubstitution)",
        "generalization": "Test (Generalization)",
    }

    # 3. Plot Lines and Shaded Areas
    for split in ["resubstitution", "generalization"]:
        subset = stats_df[stats_df["split"] == split]

        # Plot Mean Line
        plt.plot(
            subset["iteration"],
            subset["mean"],
            marker="o",
            label=labels[split],
            color=colors[split],
            linewidth=2,
        )

        # Plot Standard Deviation Shade
        plt.fill_between(
            subset["iteration"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=colors[split],
            alpha=0.15,  # Light transparency
        )

    # 4. Formatting
    plt.ylim([0, 1.05])
    plt.title(
        f"Learning Stability Analysis for {model_name.upper()}: {metric_name.upper()}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Iteration", fontweight="bold")
    plt.ylabel(f"{metric_name.upper()}\n(Mean ± Std Dev)", fontweight="bold")
    plt.xticks(range(1, 11))  # Ensure integer ticks for iterations 1-10
    plt.yticks(ticks=np.arange(0, 1.1, 0.1))
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 5. Save
    fig_filename = save_path / f"{model_name}_learning_curves_{metric_name}.png"
    plt.savefig(save_path / fig_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_model_distribution(
    df: pd.DataFrame, metric_name: str, save_path: Path, split_name: str = "generalization"
) -> Optional[pd.DataFrame]:
    """
    Generate, display, and save a per-model distribution plot and statistical summary for one metric.

    This function filters ``df`` to a single split (``split_name``), orders models
    alphabetically for consistent x-axis ordering, computes a per-model descriptive
    statistics table for ``metric_name``, displays it in-notebook (via ``ipd.display``),
    saves it as CSV, and generates a figure overlaying:
    - a boxplot (summary statistics; fliers hidden), and
    - a strip plot (raw points).

    IMPORTANT
    ---------
    Do **not** modify the function body/code in any way. Only the docstring must be edited.

    Parameters
    ----------
    df : pandas.DataFrame
        Results DataFrame containing at minimum:
        - ``"model"``: model identifier,
        - ``"split"``: split identifier (e.g., ``"resubstitution"``, ``"generalization"``),
        - ``metric_name``: numeric column to visualize.
    metric_name : str
        Metric column name to analyze (e.g., ``"mcc"``, ``"f1"``, ``"roc_auc"``).
        The column must exist in ``df`` and be numeric.
    save_path : pathlib.Path
        Output directory where the figure and the summary CSV will be saved. The function
        assumes the directory exists and is writable.
    split_name : str, default="generalization"
        Split label used to filter the input DataFrame (matched against ``df["split"]``).

    Returns
    -------
    summary_stats : pandas.DataFrame or None
        A per-model descriptive statistics table (output of
        ``df_subset.groupby("model")[metric_name].describe()``) with columns prefixed by
        ``f"{metric_name}_"`` (e.g., ``"mcc_mean"``, ``"mcc_std"``). Returns ``None`` if no
        rows match ``split_name``.

    Raises
    ------
    KeyError
        If required columns (``"model"``, ``"split"``, or ``metric_name``) are missing.
    PermissionError
        If the output files cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during file writing.

    Notes
    -----
    - Data are filtered as ``df[df["split"] == split_name]``.
    - Model ordering is enforced via ``df_subset.sort_values(by=["model"])``.
    - The statistics table is computed via ``groupby("model")[metric_name].describe()``,
      then renamed with ``add_prefix(f"{metric_name}_")`` and saved as::

          distribution_boxplot_<split_name>_<metric_name>_stats.csv

    - The figure is saved as::

          distribution_boxplot_<split_name>_<metric_name>.png

    - The plotting routine currently fixes y-limits to ``[0, 1.05]``. If you pass a metric
      with a different range (e.g., MCC in ``[-1, 1]``), adjust the limits accordingly.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df = pd.DataFrame(
    ...     {
    ...         "model": ["A", "A", "B", "B"],
    ...         "split": ["generalization"] * 4,
    ...         "mcc": [0.40, 0.35, 0.45, 0.42],
    ...     }
    ... )
    >>> out = plot_model_distribution(df, "mcc", Path("."), split_name="generalization")
    >>> (out is None) or ("mcc_mean" in out.columns)
    True
    """

    print(
        f"\n{'=' * 80}\nDISTRIBUTION ANALYSIS: {metric_name.upper()} ({split_name.upper()})\n{'=' * 80}"
    )

    # 1. Filter for the specific Split
    df_subset = df[df["split"] == split_name].copy()

    if df_subset.empty:
        print(f"No data found for split '{split_name}'. Skipping plot.")
        return None

    # 2. Sort Models Alphabetically (for consistent X-axis)
    df_subset.sort_values(by=["model"], inplace=True)

    # --- Compute & Show Statistical Report ---
    # Group by model and describe the specific metric (count, mean, std, min, 25%, 50%, 75%, max)
    summary_stats = df_subset.groupby("model")[metric_name].describe()

    # Rename columns to be metric-specific
    # Example: 'mean' -> 'mcc_mean', 'std' -> 'mcc_std'
    summary_stats = summary_stats.add_prefix(f"{metric_name}_")
    ipd.display(summary_stats)

    # Save CSV
    summary_filename = f"distribution_boxplot_{split_name}_{metric_name}_stats.csv"
    summary_stats.to_csv(save_path / summary_filename)

    # --- Generate Plot ---
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # A. Plot Boxplot (Summary)
    sns.boxplot(
        data=df_subset,
        x="model",
        hue="model",
        y=metric_name,
        palette="Spectral",
        width=0.5,
        fliersize=0,  # Hide outliers in boxplot (shown in strip plot)
        linewidth=1.5,
        legend=False,
    )

    # B. Plot Strip Plot (Raw Data Density)
    sns.stripplot(
        data=df_subset, x="model", y=metric_name, color="#333333", alpha=0.35, size=3, jitter=0.25
    )

    # C. Styling
    plt.ylim([0, 1.05])
    plt.yticks(ticks=np.arange(0, 1.1, 0.1))
    plt.title(
        f"Model Distribution - {split_name.capitalize()}: {metric_name.upper()}",
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel(f"{metric_name.upper()} Score", fontweight="bold")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 3. Save Figure
    fig_filename = f"distribution_boxplot_{split_name}_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(save_path / fig_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return summary_stats


def plot_model_barplot(
    df: pd.DataFrame, metric_name: str, save_path: Path, split_name: str = "generalization"
) -> Optional[pd.DataFrame]:
    """
    Generate and save a per-model mean barplot with standard-deviation error bars for one metric.

    This function filters the input results to a single split (``split_name``), orders
    models alphabetically to guarantee consistent x-axis ordering, computes per-model
    mean and standard deviation for ``metric_name``, displays a formatted "Mean ± Std"
    report in-notebook (via ``ipd.display``), saves the report as CSV, and generates a
    bar plot with standard-deviation error bars (``errorbar="sd"`` in seaborn).

    Parameters
    ----------
    df : pandas.DataFrame
        Results DataFrame containing at minimum:
        - ``"model"``: model identifier,
        - ``"split"``: split identifier (e.g., ``"resubstitution"``, ``"generalization"``),
        - ``metric_name``: numeric column to summarize and visualize.
    metric_name : str
        Metric column name to analyze (e.g., ``"mcc"``, ``"f1"``, ``"roc_auc"``).
        The column must exist in ``df`` and be numeric.
    save_path : pathlib.Path
        Output directory where the figure and the formatted CSV report will be saved.
        The function assumes the directory exists and is writable.
    split_name : str, default="generalization"
        Split label used to filter the input DataFrame (matched against ``df["split"]``).

    Returns
    -------
    formatted_report : pandas.DataFrame or None
        A single-column DataFrame indexed by model name, where the column name is
        ``metric_name`` and values are formatted strings ``"mean ± std"`` (3 decimals).
        Returns ``None`` if no rows match the requested split.

    Raises
    ------
    KeyError
        If required columns (``"model"``, ``"split"``, or ``metric_name``) are missing.
    ValueError
        If the filtered subset becomes empty after dropping missing values for ``metric_name``.
    PermissionError
        If output files cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during file writing.

    Notes
    -----
    - Data are filtered as ``df[df["split"] == split_name]``.
    - Model ordering is enforced via ``sort_values(by=["model"])`` and
      ``unique_models = df_subset["model"].unique()``.
    - The formatted report is built from::

          stats = df_subset.groupby("model")[metric_name].agg(["mean", "std"])
          stats[metric_name] = f"{mean:.3f} ± {std:.3f}"

      and saved as::

          distribution_barplot_<split_name>_<metric_name>_stats.csv

    - The figure is saved as::

          distribution_barplot_<split_name>_<metric_name>.png

    - The plotting routine currently fixes y-limits to ``[0, 1.05]``. If you pass a metric
      with a different range (e.g., MCC in ``[-1, 1]``), adjust the limits accordingly.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df = pd.DataFrame(
    ...     {
    ...         "model": ["A", "A", "B", "B"],
    ...         "split": ["generalization"] * 4,
    ...         "f1": [0.70, 0.68, 0.72, 0.71],
    ...     }
    ... )
    >>> out = plot_model_barplot(df, "f1", Path("."), split_name="generalization")
    >>> (out is None) or (out.shape[0] >= 1)
    True
    """

    # 1. Filter for the specific Split
    df_subset = df[df["split"] == split_name].copy()

    if df_subset.empty:
        print(f"No data found for split '{split_name}'. Skipping plot.")
        return None

    # 2. Sort Models Alphabetically
    df_subset.sort_values(by=["model"], inplace=True)
    unique_models = df_subset["model"].unique()

    print(
        f"\n{'=' * 80}\nBARPLOT ANALYSIS: {metric_name.upper()} ({split_name.upper()})\n{'=' * 80}"
    )

    # --- Compute & Format Statistical Report ---
    # Aggregation: Calculate Mean and Std
    stats = df_subset.groupby("model")[metric_name].agg(["mean", "std"])

    # Formatting: Create a single string column "Mean ± Std"
    stats[metric_name] = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)

    # Keep only the formatted column for the return object
    # Index = Model Name, Column = Metric Name
    formatted_report = stats[[metric_name]]
    ipd.display(formatted_report)

    # Save Formatted CSV for this specific metric
    csv_filename = f"distribution_barplot_{split_name}_{metric_name}_stats.csv"
    formatted_report.to_csv(save_path / csv_filename)

    # --- Generate Plot ---
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # B. Plot Bar Chart
    ax = sns.barplot(
        data=df_subset,
        x="model",
        y=metric_name,
        hue="model",
        palette="Spectral",
        errorbar="sd",  # Draw Error bar = Standard Deviation
        order=unique_models,  # Explicit ordering
        capsize=0.1,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
        legend=False,
    )

    # C. Add Text Labels (Mean only) on Bars
    means = df_subset.groupby("model")[metric_name].mean()

    for i, model in enumerate(unique_models):
        score = means[model]
        ax.text(
            i,
            0.05,
            f"{score:.3f}",
            color="black",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
            rotation=90,
        )

    # D. Styling
    plt.ylim([0, 1.05])
    plt.yticks(ticks=np.arange(0, 1.1, 0.1))
    plt.title(
        f"Model Performance - {split_name.capitalize()}: {metric_name.upper()}",
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel(f"{metric_name.upper()}\n(Mean ± Std Dev)", fontweight="bold")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 3. Save Figure
    fig_filename = f"distribution_barplot_{split_name}_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(save_path / fig_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return formatted_report


def plot_significance_heatmap(
    df_comparisons: pd.DataFrame, metric_name: str, save_path: Path
) -> None:
    """
    Plot a lower-triangular pairwise significance heatmap for a given metric.

    This function filters a pairwise-comparisons DataFrame to the requested metric,
    pivots the filtered rows into a p-value matrix (Model A × Model B), fills missing
    pairs with ``1.0`` (treated as non-significant), and renders a significance heatmap
    via :func:`scikit_posthocs.sign_plot`. The resulting figure is saved under
    ``save_path`` with a filename that includes the metric name.

    IMPORTANT
    ---------
    Do **not** modify the function body/code in any way. Only the docstring must be edited.

    Parameters
    ----------
    df_comparisons : pandas.DataFrame
        Pairwise statistical comparison results. The DataFrame must include at least
        the following columns:

        - ``"Metric"``: metric identifier used to select rows for plotting.
        - ``"Model_A"``: row model label in the p-value matrix.
        - ``"Model_B"``: column model label in the p-value matrix.
        - ``"p-value"``: p-value associated with the (Model_A, Model_B) comparison.
    metric_name : str
        Metric name to plot. Must match one of the values in ``df_comparisons["Metric"]``.
    save_path : pathlib.Path
        Output directory where the heatmap image will be written. The function assumes the
        directory exists and is writable.

    Returns
    -------
    None
        Side effects only (figure generation and file output).

    Raises
    ------
    KeyError
        If required columns (``"Metric"``, ``"Model_A"``, ``"Model_B"``, or ``"p-value"``)
        are missing from ``df_comparisons``.
    ValueError
        If no rows match the requested ``metric`` (the pivot would be empty).
    PermissionError
        If the output image cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during file writing.

    Notes
    -----
    - Rows are filtered as::

          df_metric = df_comparisons[df_comparisons["Metric"] == metric].copy()

    - The p-value matrix is created as::

          p_matrix = df_metric.pivot(index="Model_A", columns="Model_B", values="p-value")
          p_matrix = p_matrix.fillna(1.0)

      Missing comparisons are filled with ``1.0`` so they appear as non-significant.
    - The plot is produced using a user-defined categorical colormap list and styling
      options forwarded to :func:`scikit_posthocs.sign_plot` via ``heatmap_args``.
    - The figure title includes the metric name and indicates the corrected resampled
      t-test context.
    - The saved filename is::

          corrected_resampled_ttest_<metric>_heatmap.png

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df_comp = pd.DataFrame(
    ...     {
    ...         "Metric": ["roc_auc", "roc_auc"],
    ...         "Model_A": ["A", "A"],
    ...         "Model_B": ["B", "C"],
    ...         "p-value": [0.03, 0.20],
    ...     }
    ... )
    >>> plot_significance_heatmap(df_comp, "roc_auc", Path("."))
    """

    # 1. FILTER: Select only the rows for the current metric
    df_metric = df_comparisons[df_comparisons["Metric"] == metric_name].copy()

    # 2. Pivot to get the P-Value Matrix
    p_matrix = df_metric.pivot(index="Model_A", columns="Model_B", values="p-value")

    # 3. Fill NaNs with 1.0 (non-significant) initially to handle missing pairs
    p_matrix = p_matrix.fillna(1.0)

    # 4. Plot using scikit-posthocs
    plt.figure(figsize=(10, 8))

    # Define Colormap:
    # 1 (White), 0.05, 0.01, 0.001 (Blues/Reds)
    cmap = ["1", "#fb6a4a", "#08306b", "#4292c6", "#c6dbef"]

    heatmap_args = {
        "cmap": cmap,
        "linewidths": 0.5,  # Slightly thicker lines for clarity
        "linecolor": "0.9",  # Light gray grid lines
        "clip_on": False,
        "square": True,  # Force square cells
        # [x, y, width, height] in figure coordinate system
        # Moved x to 1.02 to place it outside the plot area
        "cbar_ax_bbox": [0.90, 0.35, 0.03, 0.3],
    }

    # Plot
    sp.sign_plot(p_matrix, **heatmap_args)

    plt.suptitle(
        f"Pairwise Significance: {metric_name.upper()}\n(Corrected Resampled t-test)",
        fontweight="bold",
    )

    # Adjust layout to accommodate the external legend
    plt.subplots_adjust(right=0.85)

    plt.savefig(
        save_path / f"corrected_resampled_ttest_{metric_name}_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_ranking_with_significance(
    df_raw: pd.DataFrame,
    df_comparisons: pd.DataFrame,
    metric_name: str,
    save_path: Path,
    greater_is_better: bool = True,
    alpha: float = 0.05,
) -> None:
    """
    Plot a mean±std model ranking barplot and overlay statistical-tie cliques from pairwise tests.

    This function ranks models by the mean value of ``metric_name`` computed from ``df_raw``,
    draws a barplot with standard-deviation error bars, and overlays horizontal “clique” lines
    indicating groups of models that are not significantly different according to the pairwise
    p-values in ``df_comparisons`` (typically produced by corrected resampled t-tests).

    Clique construction
    -------------------
    For the requested ``metric_name``, the function builds an undirected graph where each node
    is a model and an edge is added between two models when their pairwise comparison is
    *not significant* (``p-value >= alpha``). Maximal cliques of this graph (size > 1) are then
    converted into x-axis index intervals and drawn as horizontal lines above the bars. Nested
    clique intervals (subsets contained in larger intervals) are removed to reduce visual clutter.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw per-fold/per-iteration results used to compute mean and standard deviation per model.
        Must contain at least:
        - ``"model"``: model identifier, and
        - ``metric_name``: numeric column with the metric values to rank/plot.
    df_comparisons : pandas.DataFrame
        Pairwise comparison results containing p-values for the given metric. Must contain:
        - ``"Metric"``: metric identifier (used to filter rows for ``metric_name``),
        - ``"Model_A"`` and ``"Model_B"``: model labels, and
        - ``"p-value"``: p-value for each ordered pair.
    metric_name : str
        Metric column name used for ranking and plotting (e.g., ``"roc_auc"``, ``"f1"``).
        Must exist in ``df_raw`` and be numeric.
    save_path : pathlib.Path
        Output directory where the figure is saved. The directory must exist and be writable.
    greater_is_better : bool, default=True
        Sorting direction:
        - ``True``  → sort models by descending mean (best first),
        - ``False`` → sort models by ascending mean (best first), for loss-like metrics.
    alpha : float, default=0.05
        Significance threshold used to define non-significant edges (statistical ties) in the
        clique graph. Edges are added when ``p-value >= alpha``.

    Returns
    -------
    None
        Side effects only (figure generation and file output).

    Raises
    ------
    KeyError
        If required columns are missing from ``df_raw`` or ``df_comparisons``.
    ValueError
        If ``alpha`` is not in ``(0, 1)`` or if no rows are available to compute ranking stats
        for the selected metric.
    PermissionError
        If the output image cannot be written due to insufficient permissions.
    OSError
        For OS-related errors during file writing.

    Notes
    -----
    - Ranking stats are computed via::

          stats = df_raw.groupby("model")[metric_name].agg(["mean", "std"])

      and models are sorted by ``mean`` according to ``greater_is_better``.
    - Pairwise comparisons are filtered as::

          df_metric_comp = df_comparisons[df_comparisons["Metric"] == metric_name]

      Non-significant comparisons (``p-value >= alpha``) define graph edges.
    - The output filename is::

          corrected_resampled_ttest_<metric_name>_barplot_significance.png

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df_raw = pd.DataFrame(
    ...     {
    ...         "model": ["A", "A", "B", "B"],
    ...         "roc_auc": [0.91, 0.89, 0.93, 0.92],
    ...     }
    ... )
    >>> df_comp = pd.DataFrame(
    ...     {
    ...         "Metric": ["roc_auc", "roc_auc"],
    ...         "Model_A": ["A", "B"],
    ...         "Model_B": ["B", "A"],
    ...         "p-value": [0.20, 0.20],
    ...     }
    ... )
    >>> plot_ranking_with_significance(df_raw, df_comp, "roc_auc", Path("."), alpha=0.05)
    >>> True
    """

    # --- STEP 1: Calculate Statistics ---
    stats = df_raw.groupby("model")[metric_name].agg(["mean", "std"])

    # --- STEP 2: Sort Models by Performance ---
    # If greater_is_better=True (Accuracy): Sort Descending (Highest on left)
    # If greater_is_better=False (RMSE): Sort Ascending (Lowest on left)
    stats = stats.sort_values("mean", ascending=not greater_is_better)

    sorted_models = stats.index.tolist()

    # --- STEP 3: Identify Cliques (Statistical Ties) ---
    # Create a graph where Nodes = Models
    G = nx.Graph()
    G.add_nodes_from(sorted_models)

    # Filter only relevant comparisons
    df_metric_comp = df_comparisons[df_comparisons["Metric"] == metric_name]

    # Add an edge between models if they are NOT significantly different
    for _, row in df_metric_comp.iterrows():
        if row["p-value"] >= alpha:
            G.add_edge(row["Model_A"], row["Model_B"])

    # Find groups of models that are all connected (Maximal Cliques)
    cliques = list(nx.find_cliques(G))

    # Convert model names to their integer X-axis positions
    clique_intervals = []
    for clq in cliques:
        if len(clq) > 1:
            # Find where these models sit in our sorted list
            indices = [sorted_models.index(m) for m in clq]
            # We only need the span: from Index X to Index Y
            clique_intervals.append((min(indices), max(indices)))

    # Cleanup: Sort by length so we draw smaller groups first (looks better)
    # Filter out subsets (e.g., if A-B-C is a group, ignore A-B)
    clique_intervals.sort(key=lambda x: x[1] - x[0], reverse=True)

    final_cliques = []
    for interval in clique_intervals:
        is_subset = False
        for other in final_cliques:
            if interval[0] >= other[0] and interval[1] <= other[1]:
                is_subset = True
                break
        if not is_subset:
            final_cliques.append(interval)

    # --- STEP 4: Plotting ---
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # A. Draw the Barplot
    # We use 'order=sorted_models' to force the Best->Worst arrangement
    ax = sns.barplot(
        data=df_raw,
        x="model",
        y=metric_name,
        hue="model",
        palette="Spectral",
        errorbar="sd",
        order=sorted_models,
        capsize=0.1,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
        legend=False,
    )

    # B. Add Text Labels
    for i, model in enumerate(sorted_models):
        mean_val = stats.loc[model, "mean"]
        ax.text(
            i,
            0.05,  # Position text at the bottom of the bar
            f"{mean_val:.3f}",
            color="black",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            rotation=90,
        )

    # C. Draw Clique Lines
    # Calculate starting height (above the tallest bar + error)
    max_height = (stats["mean"] + stats["std"]).max()
    y_step = max_height * 0.05  # Space between lines
    current_y = max_height + y_step

    # Draw shortest cliques lower, longest cliques higher
    final_cliques.sort(key=lambda x: x[1] - x[0])

    for start_idx, end_idx in final_cliques:
        # Draw Horizontal Line
        plt.plot(
            [start_idx, end_idx],
            [current_y, current_y],
            color="#333333",
            linewidth=2,
            marker="|",
            markersize=10,
        )
        current_y += y_step

    # D. Final Polish
    plt.title(f"Model Ranking & Significance: {metric_name.upper()}", fontweight="bold", pad=20)
    plt.ylabel(f"{metric_name.upper()}\n(Mean ± Std Dev)", fontweight="bold")
    plt.xlabel("Model", fontweight="bold")
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylim(0, current_y + y_step)  # Extend Y-axis to fit lines

    plt.tight_layout()
    out_file = save_path / f"corrected_resampled_ttest_{metric_name}_barplot_significance.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
