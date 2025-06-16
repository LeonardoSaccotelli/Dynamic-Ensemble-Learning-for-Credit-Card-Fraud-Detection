from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator


def get_model_and_search_space(model_name: str, random_state: int | None = None) -> tuple[BaseEstimator, dict]:
    """
    Return an estimator and its hyperparameter distribution for RandomizedSearchCV.

    Parameters
    ----------
    model_name : str
        The name of the model ('rf', 'svc', 'mlp').
    random_state : int or None, optional
        Random state for reproducibility. Default is None.

    Returns
    -------
    tuple
        A tuple (estimator, param_distributions) suitable for RandomizedSearchCV.
    """
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced")
        param_dist = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }

    elif model_name == "SVC":
        model = SVC(probability=True, random_state=random_state)
        param_dist = {
            "C": uniform(0.1, 10.0),
            "gamma": ["scale", "auto"],
            "kernel": ["linear", "rbf"]
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model, param_dist