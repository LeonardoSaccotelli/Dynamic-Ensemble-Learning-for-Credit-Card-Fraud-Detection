from typing import List, Tuple
from scipy.stats import randint, uniform

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


def get_base_model_and_search_space(model_name: str, random_state: int | None = None) -> tuple[BaseEstimator, dict]:
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
    model_configurations = {
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "model_args": {
                "random_state": random_state,
                "n_jobs": -1,
                "class_weight": "balanced"
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),
                "classifier__max_depth": randint(5, 30),
                "classifier__min_samples_split": randint(2, 10),
                "classifier__min_samples_leaf": randint(1, 10),
                "classifier__max_features": ["sqrt", "log2", None],
            }
        },
        "SVC": {
            "model_class": SVC,
            "model_args": {
                "probability": True,
                "random_state": random_state
            },
            "param_dist": {
                "classifier__C": uniform(0.1, 10.0),
                "classifier__gamma": ["scale", "auto"],
                "classifier__kernel": ["linear", "rbf"]
            }
        }
    }

    base_model_config = model_configurations.get(model_name)

    if base_model_config is None:
        raise ValueError(f"Unknown model name: {model_name}")

    model = base_model_config["model_class"](**base_model_config["model_args"])
    param_dist = base_model_config["param_dist"]

    return model, param_dist


def get_static_ensemble_model(
    model_name: str,
    estimators: List[Tuple[str, BaseEstimator]]
) -> BaseEstimator:
    """
    Create a static ensemble model from pre-fitted base classifiers.

    Parameters
    ----------
    model_name : str
        Name of the ensemble model to create (e.g., "voting").

    estimators : list of (str, BaseEstimator)
        List of tuples containing (name, fitted estimator) to include in the ensemble.

    Returns
    -------
    BaseEstimator
        An ensemble model composed of the provided estimators.

    Raises
    ------
    ValueError
        If an unsupported model name is provided.
    """
    ensemble_model_configurations = {
        "VotingClassifier": {
            "model_class": VotingClassifier,
            "model_args": {
                "voting": "soft"
            }
        }
    }

    static_ens_config = ensemble_model_configurations.get(model_name)

    if static_ens_config is None:
        raise ValueError(f"Unknown static ensemble model name: {model_name}")

    model_args = static_ens_config["model_args"]
    ensemble_model = static_ens_config["model_class"](estimators=estimators, **model_args)

    return ensemble_model
