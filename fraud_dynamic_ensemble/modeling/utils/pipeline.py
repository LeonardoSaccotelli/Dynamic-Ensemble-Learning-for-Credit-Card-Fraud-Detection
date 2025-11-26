from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from fraud_dynamic_ensemble.data_preparation.data_construct import get_standard_scaler
from fraud_dynamic_ensemble.data_preparation.feature_selection import get_feature_selection
from fraud_dynamic_ensemble.data_preparation.sampling import get_resampling_pipeline


def build_model_pipeline(
    estimator: BaseEstimator,
    numerical_features_to_standardize: Sequence[Union[int, str]],
    fs_k_best_to_keep: Union[int, str],
    resampling_method: Optional[str],
    resampling_params: Optional[Mapping[str, Any]],
) -> ImbPipeline:
    """
    Assemble a leakage-safe training pipeline: scaling → two-stage feature selection
    → optional resampling → classifier.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier implementing ``fit``/``predict`` (e.g., LogisticRegression, XGBClassifier).
    numerical_features_to_standardize : Sequence[int] | Sequence[str]
        Indices or column names to scale with StandardScaler in a ColumnTransformer.
    fs_k_best_to_keep : int or {"all"}
        Number of features kept by SelectKBest. Use ``"all"`` to pass all features.
    resampling_method : str or None
        Canonical name of the sampler, e.g. ``"SMOTE"``, ``"RandomUnderSampler"``,
        ``"ADASYN"``, ``"SMOTEENN"``, or ``None``/``"none"`` for passthrough.
    resampling_params : Mapping[str, Any] or None
        Extra kwargs for the sampler constructor (e.g., ``sampling_strategy``, ``random_state``,
        ``k_neighbors``). If ``None``, no extra kwargs are passed.

    Returns
    -------
    imblearn.pipeline.Pipeline
        An ``ImbPipeline`` with steps:
        ``('preprocessor', ColumnTransformer(StandardScaler))``,
        ``('feature_selection_filter', SelectKBest)``,
        ``('feature_selection_embedded', SelectFromModel(LogisticRegression(L1)))``,
        ``('resampling', <sampler | 'passthrough'>)``,
        ``('classifier', estimator)``.
    """

    # Step 1: Preprocessing (scaling selected columns with StandardScaler)
    preprocessor = get_standard_scaler(columns=numerical_features_to_standardize)

    # Step 2: Feature selection
    fs_filter, fs_embedded = get_feature_selection(k=fs_k_best_to_keep)

    # Step 3: Resampling
    resampling = get_resampling_pipeline(strategy_name=resampling_method, **resampling_params)

    # Step 4: Preprocessing + Estimator
    final_pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection_filter", fs_filter),
            ("feature_selection_embedded", fs_embedded),
            ("resampling", resampling),
            ("classifier", estimator),
        ]
    )

    return final_pipeline


def get_final_selected_features(
    pipeline: Union[ImbPipeline, Pipeline],
    feature_names: Sequence[str],
) -> Tuple[List[int], List[str]]:
    """
    Extract final selected feature indices and names from a fitted two-step
    sequential selection pipeline.

    The pipeline is expected to include, in order:
      1) ``"feature_selection_filter"``  (e.g., ``SelectKBest``), then
      2) ``"feature_selection_embedded"`` (e.g., ``SelectFromModel`` with L1 LR).

    The function reads the boolean mask from the filter step, applies the embedded
    step on that reduced space, and maps the embedded mask back to **global**
    indices (relative to the selector input), finally returning both indices
    and names.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline
        A **fitted** pipeline exposing steps named
        ``"feature_selection_filter"`` and ``"feature_selection_embedded"``,
        each implementing ``get_support()``.
    feature_names : Sequence[str]
        Feature names **aligned with the input of the first selector**. If your
        pipeline contains a preprocessor (e.g., ``ColumnTransformer``), pass the
        names **after** preprocessing (e.g.,
        ``pipeline.named_steps['preprocessor'].get_feature_names_out()``) so the
        dimensionality matches the selector input.

    Returns
    -------
    fs_final_features_indices : list of int
        Global indices kept **after** the second (embedded) selection step.
    fs_final_features_names : list of str
        Names corresponding to ``fs_final_features_indices``, taken from
        ``feature_names``.

    Raises
    ------
    KeyError
        If the expected steps (``'feature_selection_filter'``,
        ``'feature_selection_embedded'``) are not present.
    IndexError
        If any returned index exceeds the bounds of ``feature_names`` (typically
        due to misalignment between the preprocessor output and provided names).

    Notes
    -----
    - This utility assumes **sequential** selectors (filter → embedded). For
      parallel unions (``FeatureUnion``), use an inspection routine tailored to
      unions.
    - Ensure the pipeline is **fitted**; otherwise ``get_support()`` will fail.

    Examples
    --------
    >>> pre = pipeline.named_steps["preprocessor"]
    >>> names = pre.get_feature_names_out()
    >>> fs_idx, fs_names = get_final_selected_features(pipeline, names)
    >>> fs_idx[:5], fs_names[:5]
    ([0, 5, 12, 19, 27], ['V1', 'V4', 'V10', 'Amount_log1p', 'Time_sin'])
    """

    # Access the specific steps by the names defined in your ImbPipeline
    try:
        fs_filter = pipeline.named_steps["feature_selection_filter"]
        fs_embedded = pipeline.named_steps["feature_selection_embedded"]
    except KeyError as e:
        raise KeyError(f"Could not find the expected step name in pipeline: {e}")

    # Get indices from Step 1 (Filter)
    # fs_filter_mask is size N (all features)
    fs_filter_mask = fs_filter.get_support()

    # fs_filter_indices contains the indices (e.g., [0, 5, 12]) kept by the filter
    fs_filter_indices = [i for i, keep in enumerate(fs_filter_mask) if keep]

    # Get indices from Step 2 (Embedded)
    # fs_embedded_mask is size M (where M < N, only features kept by Step 1)
    fs_embedded_mask = fs_embedded.get_support()

    # Map Step 2 mask back to Global Indices
    # We iterate through the mask of step 2. If it says "Keep" (True),
    # we take the corresponding integer from our list of Step 1 indices.
    fs_final_features_indices = [
        fs_filter_indices[i] for i, keep in enumerate(fs_embedded_mask) if keep
    ]

    # Retrieve Names
    # We map the final global indices to the provided feature name list
    fs_final_features_name = [feature_names[i] for i in fs_final_features_indices]

    return fs_final_features_indices, fs_final_features_name
