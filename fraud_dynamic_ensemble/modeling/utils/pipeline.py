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
    Assemble a leakage-safe training pipeline: scaling → SelectKBest → optional resampling → classifier.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier implementing ``fit``/``predict`` (e.g., SVC, XGBClassifier).
    numerical_features_to_standardize : Sequence[int] | Sequence[str]
        Indices or column names to scale with StandardScaler in a ColumnTransformer.
    fs_k_best_to_keep : int or {"all"}
        Number of features kept by SelectKBest. Use ``"all"`` to pass all features.
        This value is the *default* ``k`` used to build the pipeline; it can be
        overridden during hyperparameter tuning by including
        ``feature_selection_filter__k`` in the search space.
    resampling_method : str or None
        Canonical name of the sampler, e.g. ``"SMOTE"``, ``"RandomUnderSampler"``,
        ``"ADASYN"``, ``"SMOTEENN"``, or ``None``/``"none"`` for passthrough.
    resampling_params : Mapping[str, Any] or None
        Extra kwargs for the sampler constructor (e.g., ``sampling_strategy``,
        ``random_state``, ``k_neighbors``). If ``None``, no extra kwargs are passed.

    Returns
    -------
    imblearn.pipeline.Pipeline
        An ``ImbPipeline`` with steps:
        ``('preprocessor', ColumnTransformer(StandardScaler))``,
        ``('feature_selection_filter', SelectKBest)``,
        ``('resampling', <sampler | 'passthrough'>)``,
        ``('classifier', estimator)``.

    Notes
    -----
    - The sampler step is applied during ``fit`` (training) as provided by imbalanced-learn
      pipelines; at inference time, resampling is not performed.
    - If ``resampling_method`` disables resampling (``None``/``"none"``), the resampling
      step should behave as ``'passthrough'`` (depending on your implementation of
      ``get_resampling_pipeline``).
    """

    # Step 1: Preprocessing (scaling selected columns with StandardScaler)
    preprocessor = get_standard_scaler(columns=numerical_features_to_standardize)

    # Step 2: Feature selection
    fs_filter = get_feature_selection(k=fs_k_best_to_keep)

    # Step 3: Resampling
    resampling = get_resampling_pipeline(strategy_name=resampling_method, **resampling_params)

    # Step 4: Preprocessing + Estimator
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

    This helper performs the following steps:

    1. Maps original column names to integer indices.
    2. Validates that all requested features to standardize are present.
    3. Builds the list of integer indices to be passed to the scaler
       (``idx_num_features_to_standardize``).
    4. Reconstructs the feature order **after** the preprocessing step
       implemented by a ``ColumnTransformer`` that:
       - applies a transformer (e.g., ``StandardScaler``) to the selected
         columns (by index), and
       - uses ``remainder='passthrough'`` for all remaining columns.

       Under these assumptions, the transformed feature order is:

       ``[scaled_features (in the order of `idx_num_features_to_standardize`),
          all_other_features (in original column order)]``.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features dataframe **before** preprocessing. Columns are assumed
        to be in the original feature order.
    numerical_features_to_standardize : sequence of str
        Names of the features to be standardized by the scaler inside the
        ``ColumnTransformer`` (e.g., column names from ``X.columns``).

    Returns
    -------
    idx_num_features_to_standardize : list of int
        Integer indices of the features to be standardized, aligned with
        the original column order of ``X``. These indices are suitable to be
        passed directly to the scaler in the ``ColumnTransformer``.
    original_feature_names : list of str
        Original feature names in the same order as ``X.columns``.
    transformed_feature_names : list of str
        Feature names in the **post-preprocessing** order, i.e. as seen by
        downstream selectors (``"feature_selection_filter"``,
        ``"feature_selection_embedded"``) when the preprocessor is a
        ``ColumnTransformer`` with a scaler on ``idx_num_features_to_standardize``
        and ``remainder='passthrough'``.

    Raises
    ------
    ValueError
        If any feature requested for standardization is not found among
        ``X.columns``.

    Notes
    -----
    - This function assumes that the scaler used for standardization does
      **not** change the dimensionality (e.g., ``StandardScaler``).
    - If your preprocessing pipeline becomes more complex (e.g., multiple
      transformers, one-hot encoding), consider switching to
      ``preprocessor.get_feature_names_out()`` instead of reconstructing
      the order manually.
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
