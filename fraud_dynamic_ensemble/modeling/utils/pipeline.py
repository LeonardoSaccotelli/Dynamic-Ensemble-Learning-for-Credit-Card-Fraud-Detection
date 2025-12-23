from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
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
    Build a leakage-safe imbalanced-learn pipeline for binary classification.

    The resulting :class:`imblearn.pipeline.Pipeline` enforces a consistent
    train-time ordering for each CV split:
    1) standardize selected numeric features,
    2) apply filter-based feature selection (SelectKBest),
    3) optionally resample the training fold to address imbalance, and
    4) fit the final estimator.

    This structure is intended for cross-validation and hyperparameter tuning so
    that scaling, feature selection, and resampling are fit/applied only on the
    training portion of each split, reducing leakage risk.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier used as the last pipeline step (named ``"classifier"``).
        Must implement ``fit`` and ``predict`` (and optionally ``predict_proba`` if
        required by downstream evaluation).
    numerical_features_to_standardize : Sequence[int or str]
        Feature indices or column names to standardize in the preprocessing step.
        Indices are appropriate for NumPy-array inputs; names are appropriate for
        pandas DataFrame inputs.
    fs_k_best_to_keep : int or {'all'}
        Default number of top features to keep in SelectKBest. Use ``'all'`` to
        keep all features. This default can be overridden in tuning via
        ``feature_selection_filter__k``.
    resampling_method : str or None
        Canonical resampling strategy name (e.g., ``'SMOTE'``, ``'RandomUnderSampler'``,
        ``'SMOTEENN'``). If ``None`` or ``'none'``, resampling is disabled and the
        resampling step is configured as passthrough.
    resampling_params : Mapping[str, Any] or None
        Optional keyword arguments forwarded to the sampler constructor via
        :func:`get_resampling_pipeline`. If ``None``, no additional parameters are
        provided.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline with the following steps:
        - ``('preprocessor', ColumnTransformer)``
        - ``('feature_selection_filter', SelectKBest)``
        - ``('resampling', BaseSampler or 'passthrough')``
        - ``('classifier', estimator)``

    Notes
    -----
    - Resampling is applied only during ``fit`` (train-time) and not during
      ``predict``.
    - Step names are chosen to support scikit-learn's parameter routing for tuning
      (e.g., ``classifier__C``, ``feature_selection_filter__k``).
    - The pipeline can operate on either pandas DataFrames or NumPy arrays, provided
      the selected column representation matches the input type.

    Examples
    --------
    >>> pipe = build_model_pipeline(
    ...     estimator=svc,
    ...     numerical_features_to_standardize=idx_num_features,
    ...     fs_k_best_to_keep=20,
    ...     resampling_method="SMOTE",
    ...     resampling_params={"sampling_strategy": 0.2, "random_state": 42},
    ... )
    >>> search_space = {
    ...     "feature_selection_filter__k": [10, 20, 30, "all"],
    ...     "classifier__C": [0.1, 1.0, 10.0],
    ... }
    """

    # Step 1: Preprocessing (scaling selected columns with StandardScaler)
    preprocessor = get_standard_scaler(columns=numerical_features_to_standardize)

    # Step 2: Feature selection
    fs_filter = get_feature_selection(k=fs_k_best_to_keep)

    # Step 3: Resampling
    safe_resampling_params: dict[str, Any] = dict(resampling_params or {})

    # If you consider None/"none" as "no resampling", avoid passing kwargs
    if resampling_method is None or resampling_method == "none":
        resampling = get_resampling_pipeline(strategy_name=None)
    else:
        resampling = get_resampling_pipeline(
            strategy_name=resampling_method, **safe_resampling_params
        )

    # Step 4: Preprocessing (Step 1 - Step 3) + Estimator
    final_pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection_filter", fs_filter),
            ("resampling", resampling),
            ("classifier", estimator),
        ]
    )

    return final_pipeline


def build_standardization_and_feature_order(
    X: pd.DataFrame,
    numerical_features_to_standardize: Sequence[str],
) -> Tuple[List[int], List[str], List[str]]:
    """
    Derive standardization indices and post-preprocessing feature order.

    This helper computes (1) the integer indices of columns to be standardized and
    (2) the expected feature-name order after a preprocessing step implemented as a
    :class:`sklearn.compose.ColumnTransformer` that applies a scaler to the selected
    columns and uses ``remainder='passthrough'`` for all others.

    Under these assumptions, downstream steps see features ordered as:
    ``[scaled_features (in requested order), remaining_features (in original order)]``.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature DataFrame prior to preprocessing. Column order is treated as
        the canonical "original" feature order.
    numerical_features_to_standardize : Sequence[str]
        Names of features to be standardized. Each name must exist in ``X.columns``.
        The order of this sequence determines the order of the scaled block in the
        transformed feature list.

    Returns
    -------
    idx_num_features_to_standardize : list of int
        Integer indices of standardized features with respect to ``X.columns``. These
        indices are suitable for index-based selection in a ColumnTransformer.
    original_feature_names : list of str
        Feature names in the original order (``X.columns``).
    transformed_feature_names : list of str
        Feature names in the expected post-preprocessing order:
        standardized features first (in the order given by
        ``numerical_features_to_standardize``), followed by the remaining features
        in their original order.

    Raises
    ------
    ValueError
        If any feature listed in ``numerical_features_to_standardize`` is not present
        in ``X.columns``.

    Notes
    -----
    - This function assumes the scaler does not change feature dimensionality (e.g.,
      StandardScaler).
    - If preprocessing becomes more complex (multiple transformers, one-hot encoding,
      dropped columns), prefer ``preprocessor.get_feature_names_out()`` over manual
      reconstruction.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"V1": [0.1, 0.2], "V2": [1.0, 2.0], "Amount": [10.0, 20.0]})
    >>> idx_num, orig_names, trans_names = build_standardization_and_feature_order(
    ...     X=X,
    ...     numerical_features_to_standardize=["Amount", "V2"],
    ... )
    >>> idx_num
    [2, 1]
    >>> orig_names
    ['V1', 'V2', 'Amount']
    >>> trans_names
    ['Amount', 'V2', 'V1']
    """

    # Original feature names and name -> index mapping
    original_feature_names = X.columns.tolist()
    features_index = {name: idx for idx, name in enumerate(original_feature_names)}

    # Check that all requested features to standardize are present
    missing_features = set(numerical_features_to_standardize) - features_index.keys()
    if missing_features:
        raise ValueError(
            f"The following features requested for standardization were not "
            f"found in the dataset columns: {missing_features}.\n"
            f"Available features: {original_feature_names}"
        )

    # Map feature names to integer indices (for ColumnTransformer by index)
    idx_num_features_to_standardize = [
        features_index[name] for name in numerical_features_to_standardize
    ]

    # Build post-preprocessing feature order:
    # scaled features first, then all remaining features in original order.
    n_features = len(original_feature_names)
    scaled_indices = list(idx_num_features_to_standardize)
    passthrough_indices = [i for i in range(n_features) if i not in scaled_indices]

    transformed_feature_names = [original_feature_names[i] for i in scaled_indices] + [
        original_feature_names[i] for i in passthrough_indices
    ]

    return idx_num_features_to_standardize, original_feature_names, transformed_feature_names


def get_final_selected_features(
    pipeline: Union[ImbPipeline, Pipeline],
    feature_names: Sequence[str],
) -> Tuple[List[int], List[str]]:
    """
    Extract selected feature indices and names from a fitted SelectKBest step.

    The pipeline is expected to contain a fitted step named
    ``"feature_selection_filter"`` implementing ``get_support()`` (typically a
    :class:`sklearn.feature_selection.SelectKBest`). This helper reads the boolean
    support mask, converts it into integer indices, and maps those indices to the
    provided feature names.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline
        Fitted pipeline exposing a step named ``"feature_selection_filter"`` that
        implements ``get_support()``.
    feature_names : Sequence[str]
        Feature names aligned with the input to ``"feature_selection_filter"``.
        If the pipeline includes preprocessing that changes feature order, pass
        names after preprocessing (e.g., from
        ``pipeline.named_steps["preprocessor"].get_feature_names_out()``) so the
        dimensionality matches the selector input.

    Returns
    -------
    selected_features_indices : list of int
        Indices of selected features relative to the selector input.
    selected_features_names : list of str
        Names corresponding to ``selected_features_indices``.

    Raises
    ------
    KeyError
        If the pipeline does not contain a step named ``"feature_selection_filter"``.
    IndexError
        If selected indices are out of bounds for ``feature_names`` (typically due to
        misalignment between the selector input and the provided feature name list).

    Notes
    -----
    - The pipeline must be fitted; otherwise ``get_support()`` will fail.
    - This helper assumes a single filter selector step. If you introduce multiple
      selection stages, you will need a different inspection routine.

    Examples
    --------
    >>> pre = pipeline.named_steps["preprocessor"]
    >>> names = pre.get_feature_names_out()
    >>> idx, sel_names = get_final_selected_features(pipeline, names)
    >>> (idx[:3], sel_names[:3])
    ([0, 5, 12], ['V1', 'V4', 'V10'])
    """

    # Access the specific steps by the names defined in your ImbPipeline
    try:
        fs_filter = pipeline.named_steps["feature_selection_filter"]
    except KeyError as e:
        raise KeyError(f"Could not find the expected step name in pipeline: {e}")

    # Get indices from Step 1 (Filter)
    # fs_filter_mask is size N (all features)
    fs_filter_mask = fs_filter.get_support()

    # fs_filter_indices contains the indices (e.g., [0, 5, 12]) kept by the filter
    fs_filter_indices = [i for i, keep in enumerate(fs_filter_mask) if keep]

    # Retrieve Names
    # We map the final global indices to the provided feature name list
    fs_final_features_name = [feature_names[i] for i in fs_filter_indices]

    return fs_filter_indices, fs_final_features_name
