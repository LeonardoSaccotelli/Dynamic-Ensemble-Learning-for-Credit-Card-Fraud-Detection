from imblearn.pipeline import Pipeline as ImbPipeline

from fraud_dynamic_ensemble.data_preparation.data_construct import get_standard_scaler
from fraud_dynamic_ensemble.data_preparation.feature_selection import get_feature_selection
from fraud_dynamic_ensemble.data_preparation.sampling import get_resampling_pipeline


def build_model_pipeline(
    estimator,
    numerical_features_to_standardize,
    fs_k_best_to_keep,
    resampling_method,
    resampling_params,
):
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
