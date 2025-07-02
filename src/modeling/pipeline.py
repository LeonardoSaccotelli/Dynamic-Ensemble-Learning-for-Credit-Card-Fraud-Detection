from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression


def build_base_model_pipeline(
    estimator: BaseEstimator,
    numerical_columns_to_scale: list[int],
    k_best: int = 20,
    remainder: str = "passthrough",
    random_state: int = 42,
) -> Pipeline:
    """
    Build a scikit-learn pipeline with scaling, feature selection pipeline,
    and final estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The machine learning model (e.g., RandomForestClassifier).

    numerical_columns_to_scale : list of int
        List of numerical index of column names to scale using StandardScaler.

    k_best : int, optional
        Number of top features to keep with SelectKBest. Default is 20.

    remainder : str, optional
        How to handle the remaining columns (default is 'passthrough').

    random_state : int, optional
        Random state for reproducibility (used in SelectFromModel).

    Returns
    -------
    Pipeline
        A scikit-learn pipeline that applies scaling, two-step feature
        selection, and then fits the provided estimator.
    """
    # Step 1: Preprocessing (scaling selected columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), numerical_columns_to_scale)
        ],
        remainder=remainder
    )

    # Step 2: Feature selection pipeline
    feature_selection = Pipeline([
        ("select_kbest", SelectKBest(score_func=f_classif, k=k_best)),
        ("select_from_model", SelectFromModel(
            LogisticRegression(
                tol=1e-2,
                C=0.1,
                penalty="l1",
                solver="saga",
                max_iter=25000,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        ))
    ])

    # Step 3: Full pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("feature_selection", feature_selection),
        ("classifier", estimator)
    ])

    return pipeline


def get_final_selected_features(pipeline: Pipeline, feature_names: list[str]) -> tuple[list[int], list[str]]:
    """
    Extract final selected feature indices and names from a pipeline
    with nested SelectKBest + SelectFromModel.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted sklearn pipeline with steps including feature_selection.

    feature_names : list of str
        List of original column names (before ColumnTransformer).

    Returns
    -------
    tuple[list[int], list[str]]
        - Indices of selected features
        - Names of selected features
    """
    feature_selection_pipeline = pipeline.named_steps["feature_selection"]
    kbest = feature_selection_pipeline.named_steps["select_kbest"]
    sfm = feature_selection_pipeline.named_steps["select_from_model"]

    # Get indices from k-best
    mask_kbest = kbest.get_support()
    kbest_indices = [i for i, keep in enumerate(mask_kbest) if keep]

    # Get indices from SelectFromModel (applied on k-best)
    mask_sfm = sfm.get_support()
    final_indices = [kbest_indices[i] for i, keep in enumerate(mask_sfm) if keep]

    # Get final feature names
    final_names = [feature_names[i] for i in final_indices]

    return final_indices, final_names