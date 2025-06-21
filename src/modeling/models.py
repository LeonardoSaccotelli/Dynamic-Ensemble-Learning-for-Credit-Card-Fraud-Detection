from typing import List, Tuple, Optional
from scipy.stats import randint, uniform, loguniform

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from deslib.dcs import APriori, APosteriori, LCA, MLA, OLA
from deslib.des import KNORAE, KNORAU, KNOP, DESP, DESKNN, DESClustering, METADES
from deslib.des.probabilistic import RRC, DESKL, Exponential, Logarithmic
from deslib.static import StackedClassifier


def get_base_model_and_search_space(model_name: str, random_state: int | None = None) -> tuple:
    """
    Return an estimator and its hyperparameter distribution for RandomizedSearchCV.
    """
    model_configurations = {
        "SVC": {
            "model_class": SVC,
            "model_args": {
                "probability": True,
                "random_state": random_state
            },
            "param_dist": {
                "classifier__C": uniform(0.1, 10.0),
                "classifier__gamma": ["scale", "auto"],
                "classifier__kernel": ["rbf", "poly"]
            }
        },
        "KNeighborsClassifier": {
            "model_class": KNeighborsClassifier,
            "model_args": {},
            "param_dist": {
                "classifier__n_neighbors": randint(3, 20),
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        },
        "DecisionTreeClassifier": {
            "model_class": DecisionTreeClassifier,
            "model_args": {
                "random_state": random_state,
                "class_weight": "balanced",
            },
            "param_dist": {
                "classifier__max_depth": randint(3, 20),
                "classifier__min_samples_split": randint(2, 10),
                "classifier__min_samples_leaf": randint(1, 10),
            }
        },
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
        "ExtraTreesClassifier": {
            "model_class": ExtraTreesClassifier,
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
        "BalancedRandomForestClassifier": {
            "model_class": BalancedRandomForestClassifier,
            "model_args": {
                "random_state": random_state,
                "n_jobs": -1,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),
                "classifier__max_depth": randint(5, 30),
                "classifier__min_samples_split": randint(2, 10),
                "classifier__min_samples_leaf": randint(1, 10),
                "classifier__max_features": ["sqrt", "log2", None],
            }
        },
        "RUSBoostClassifier": {
            "model_class": RUSBoostClassifier,
            "model_args": {
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),
                "classifier__learning_rate": uniform(0.01, 1.0)
            }
        },
        "AdaBoostClassifier": {
            "model_class": AdaBoostClassifier,
            "model_args": {
                "random_state": random_state
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),
                "classifier__learning_rate": uniform(0.01, 1.0)
            }
        },
        "LogitBoostClassifier": {
            "model_class": GradientBoostingClassifier,
            "model_args": {
                "random_state": random_state,
                "loss": "log_loss"
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),
                "classifier__learning_rate": uniform(0.01, 1.0),
                "classifier__max_depth": randint(3, 10)
            }
        },
        "XGBClassifier": {
            "model_class": XGBClassifier,
            "model_args": {
                "random_state": random_state,
                "eval_metric": "logloss"
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),
                "classifier__max_depth": randint(3, 10),
                "classifier__learning_rate": uniform(0.01, 0.3),
                "classifier__subsample": uniform(0.5, 0.5),
                "classifier__colsample_bytree": uniform(0.5, 0.5),
            }
        },
        "LGBMClassifier": {
            "model_class": LGBMClassifier,
            "model_args": {
                "random_state": random_state
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),
                "classifier__max_depth": randint(3, 10),
                "classifier__learning_rate": uniform(0.01, 0.3),
                "classifier__num_leaves": randint(20, 150)
            }
        },
        "MLPClassifier": {
            "model_class": MLPClassifier,
            "model_args": {
                "random_state": random_state,
                "max_iter": 300
            },
            "param_dist": {
                "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "classifier__alpha": uniform(1e-5, 1e-2),
                "classifier__learning_rate_init": uniform(1e-4, 1e-2)
            }
        }
    }

    if model_name not in model_configurations:
        raise ValueError(f"Unknown model name: {model_name}")

    config = model_configurations[model_name]
    model = config["model_class"](**config["model_args"])
    return model, config["param_dist"]


def get_static_ensemble_model(
    model_name: str,
    estimators: List[Tuple[str, BaseEstimator]],
    weights: Optional[List[float]] = None
) -> BaseEstimator:
    """
    Create a static ensemble model from pre-fitted base classifiers.

    Parameters
    ----------
    model_name : str
        Name of the ensemble model to create (e.g., "VotingClassifier", "VotingClassifier_weighted").

    estimators : list of (str, BaseEstimator)
        List of tuples containing (name, fitted estimator) to include in the ensemble.

    weights : list of float, optional
        Weights to assign to each classifier in soft voting. Used only if model_name is "VotingClassifier_weighted".

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
        },
        "VotingClassifier_weighted": {
            "model_class": VotingClassifier,
            "model_args": {
                "voting": "soft",
                "weights": weights
            }
        }
    }

    static_ens_config = ensemble_model_configurations.get(model_name)

    if static_ens_config is None:
        raise ValueError(f"Unknown static ensemble model name: {model_name}")

    model_args = static_ens_config["model_args"]
    ensemble_model = static_ens_config["model_class"](estimators=estimators, **model_args)

    return ensemble_model


def get_des_model(
    model_name: str,
    pool_classifiers: List[BaseEstimator]
) -> BaseEstimator:
    """
    Create a DES (Dynamic Ensemble Selection) model from pre-fitted base classifiers.

    Parameters
    ----------
    model_name : str
        Name of the DES model to create.

    pool_classifiers : list of BaseEstimator
        List of fitted base classifiers to use as the pool.

    Returns
    -------
    BaseEstimator
        A DES model instance initialized with the provided pool.

    Raises
    ------
    ValueError
        If an unsupported DES model name is provided.
    """
    des_model_configurations = {
        "APriori": {
            "model_class": APriori,
            "model_args": {}
        },
        "APosteriori": {
            "model_class": APosteriori,
            "model_args": {}
        },
        "LCA": {
            "model_class": LCA,
            "model_args": {}
        },
        "MLA": {
            "model_class": MLA,
            "model_args": {}
        },
        "OLA": {
            "model_class": OLA,
            "model_args": {}
        },
        "KNORAE": {
            "model_class": KNORAE,
            "model_args": {}
        },
        "KNORAU": {
            "model_class": KNORAU,
            "model_args": {}
        },
        "METADES": {
            "model_class": METADES,
            "model_args": {}
        },
        "DESP": {
            "model_class": DESP,
            "model_args": {}
        },
        "DESKNN": {
            "model_class": DESKNN,
            "model_args": {}
        },
        "DESClustering": {
            "model_class": DESClustering,
            "model_args": {}
        },
        "KNOP": {
            "model_class": KNOP,
            "model_args": {}
        },
        "DESKL": {
            "model_class": DESKL,
            "model_args": {}
        },
        "Exponential": {
            "model_class": Exponential,
            "model_args": {}
        },
        "Logarithmic": {
            "model_class": Logarithmic,
            "model_args": {}
        },
        "RRC": {
            "model_class": RRC,
            "model_args": {}
        },
        "StackedClassifier": {
            "model_class": StackedClassifier,
            "model_args": {}
        }
    }

    des_config = des_model_configurations.get(model_name)

    if des_config is None:
        raise ValueError(f"Unknown DES model name: {model_name}")

    model_class = des_config["model_class"]
    model_args = des_config["model_args"]

    return model_class(pool_classifiers=pool_classifiers, **model_args)
