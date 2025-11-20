from typing import Any, Mapping, Optional, Sequence, Union

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator

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
