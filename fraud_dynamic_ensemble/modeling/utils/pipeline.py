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
    Build a leakage-safe imbalanced-learn pipeline for supervised classification.

    The returned pipeline enforces a consistent order of operations for each CV split:

    1. **Preprocessing**: scale selected numerical columns via a
       :class:`sklearn.compose.ColumnTransformer` wrapping
       :class:`sklearn.preprocessing.StandardScaler`.
    2. **Feature selection**: apply :class:`sklearn.feature_selection.SelectKBest` to retain
       ``fs_k_best_to_keep`` features (or all features when ``k="all"``).
    3. **Resampling (optional, train-time only)**: apply an imbalanced-learn sampler during
       ``fit`` to address class imbalance. If resampling is disabled, the step is configured
       as passthrough.
    4. **Classifier**: fit/predict using the provided final estimator.

    This structure is intended for cross-validation and hyperparameter tuning so that
    preprocessing, feature selection, and resampling are fit **only** on the training portion
    of each split, preventing data leakage.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier appended as the last step of the pipeline (named ``"classifier"``).
        Must implement ``fit`` and ``predict`` (and optionally ``predict_proba`` if required
        by downstream evaluation).
    numerical_features_to_standardize : Sequence[Union[int, str]]
        Feature indices or column names to scale with :class:`~sklearn.preprocessing.StandardScaler`
        inside the preprocessing step. The underlying preprocessing factory
        (:func:`get_standard_scaler`) must support the provided type (indices for NumPy arrays,
        column names for pandas DataFrames).
    fs_k_best_to_keep : Union[int, str]
        Default value for ``k`` in :class:`~sklearn.feature_selection.SelectKBest`.
        Use ``"all"`` to keep all features. This is the pipeline construction default; it can
        be overridden during hyperparameter tuning by searching over
        ``"feature_selection_filter__k"``.
    resampling_method : Optional[str]
        Canonical name of the resampling strategy (e.g., ``"SMOTE"``, ``"RandomUnderSampler"``,
        ``"ADASYN"``, ``"SMOTEENN"``). If ``None`` or case-insensitive ``"none"``, resampling is
        disabled and the resampling step is configured as passthrough.
    resampling_params : Optional[Mapping[str, Any]]
        Optional keyword arguments forwarded to the sampler constructor via
        :func:`get_resampling_pipeline` (e.g., ``sampling_strategy``, ``random_state``,
        ``k_neighbors``). If ``None``, no extra keyword arguments are provided.

    Returns
    -------
    imblearn.pipeline.Pipeline
        An :class:`imblearn.pipeline.Pipeline` (``ImbPipeline``) with the following step names:

        - ``("preprocessor", preprocessor)``
        - ``("feature_selection_filter", fs_filter)``
        - ``("resampling", resampling)``
        - ``("classifier", estimator)``

    Notes
    -----
    - **Leakage safety:** when used within cross-validation, all data-dependent steps
      (scaling, feature selection, and resampling) are fitted only on the training split.
    - **Train-time only resampling:** imbalanced-learn resampling occurs during ``fit`` and is
      not applied during ``predict`` / inference.
    - **Tuning hooks:** the step names are chosen to support hyperparameter tuning using
      scikit-learn’s double-underscore convention (e.g., ``"classifier__C"``,
      ``"feature_selection_filter__k"``, sampler-specific parameters if exposed).
    - **Input types:** the pipeline can be used with NumPy arrays or pandas DataFrames, as
      long as the preprocessing utilities support the chosen
      ``numerical_features_to_standardize`` representation.

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
    Prepare indices and post-preprocessing feature order for standardization.

    This helper derives:
    1) the integer indices of the columns to standardize, and
    2) the feature-name order *after* a preprocessing step implemented as a
       :class:`sklearn.compose.ColumnTransformer` that applies a scaler to a subset of
       columns and uses ``remainder="passthrough"`` for all others.

    Under these assumptions, the post-preprocessing order seen by downstream steps is:

    ``[scaled_features (in the requested order),
      all_other_features (in original column order)]``.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature dataframe **before** preprocessing. Column order is assumed to be the
        original feature order.
    numerical_features_to_standardize : Sequence[str]
        Names of the features to be standardized by the scaler inside the
        :class:`~sklearn.compose.ColumnTransformer`. Each name must be present in
        ``X.columns``. The order of this sequence determines the order of the scaled block
        in the transformed feature list.

    Returns
    -------
    idx_num_features_to_standardize : List[int]
        Integer indices of the features to standardize, aligned with the original column
        order of ``X``. These indices are suitable to be passed to a ColumnTransformer that
        selects columns by index.
    original_feature_names : List[str]
        Original feature names in the same order as ``X.columns``.
    transformed_feature_names : List[str]
        Feature names in the post-preprocessing order, as seen by downstream steps (e.g.,
        feature selectors). With a single scaler transformer applied to the selected indices
        and ``remainder="passthrough"``, this order is:
        ``scaled_features`` first (in the order provided by ``numerical_features_to_standardize``),
        followed by all remaining columns in their original order.

    Raises
    ------
    ValueError
        If any feature requested for standardization is not present in ``X.columns``.

    Notes
    -----
    - This function assumes the standardization transformer does **not** change the number of
      features (e.g., :class:`~sklearn.preprocessing.StandardScaler`).
    - If preprocessing becomes more complex (multiple transformers, one-hot encoding, dropped
      columns, etc.), prefer using ``preprocessor.get_feature_names_out()`` rather than
      reconstructing the order manually.

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
    Extract selected feature indices and names from a fitted ``SelectKBest`` step.

    The pipeline is expected to include a fitted selector step named
    ``"feature_selection_filter"`` (typically a ``SelectKBest`` instance).
    This utility reads the selector boolean mask via ``get_support()``,
    converts it into global indices (relative to the selector input),
    and returns both indices and corresponding feature names.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline
        A **fitted** pipeline exposing a step named ``"feature_selection_filter"``
        implementing ``get_support()`` (e.g., ``SelectKBest``).
    feature_names : Sequence[str]
        Feature names aligned with the input of ``"feature_selection_filter"``.
        If your pipeline contains a preprocessor (e.g., ``ColumnTransformer``),
        pass the names after preprocessing (e.g.,
        ``pipeline.named_steps['preprocessor'].get_feature_names_out()``) so the
        dimensionality matches the selector input.

    Returns
    -------
    selected_features_indices : list of int
        Global indices kept by the ``"feature_selection_filter"`` step.
    selected_features_names : list of str
        Names corresponding to ``selected_features_indices``, taken from
        ``feature_names``.

    Raises
    ------
    KeyError
        If the expected step (``"feature_selection_filter"``) is not present.
    IndexError
        If any returned index exceeds the bounds of ``feature_names`` (typically
        due to misalignment between the selector input and the provided names).

    Notes
    -----
    - The pipeline must be **fitted**; otherwise ``get_support()`` will fail.
    - This utility assumes a *single* filter selector. If you later reintroduce
      multiple selectors (sequential or unions), you will need a different
      inspection routine.

    Examples
    --------
    >>> pre = pipeline.named_steps["preprocessor"]
    >>> names = pre.get_feature_names_out()
    >>> idx, sel_names = get_final_selected_features(pipeline, names)
    >>> idx[:5], sel_names[:5]
    ([0, 5, 12, 19, 27], ['V1', 'V4', 'V10', 'Amount_log1p', 'Time_sin'])
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
