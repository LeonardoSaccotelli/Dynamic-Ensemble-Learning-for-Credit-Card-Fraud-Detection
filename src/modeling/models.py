from typing import List, Tuple, Optional
from scipy.stats import randint, uniform, loguniform

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

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
                "coef0": 0.0,
                "shrinking": True,
                "probability": True,
                "tol": 1e-3,
                "cache_size": 200,

                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "verbose": False,
                "max_iter": -1,
                "random_state": random_state,
            },
            "param_dist": {
                # Regularization parameter. Smaller values specify stronger regularization.
                "classifier__C": loguniform(1e-3, 1e3),
                "classifier__gamma": loguniform(1e-4, 1e0), # Kernel coefficient for 'rbf' and 'poly' kernels.
                "classifier__kernel": ["rbf", "poly", "linear"], # Kernel type
                # Degree of the polynomial kernel function (only relevant if kernel='poly').
                "classifier__degree": randint(2, 5),
            }
        },
        "KNeighborsClassifier": {
            "model_class": KNeighborsClassifier,
            "model_args": {
                "n_jobs": -1,
                "algorithm": "auto",
                "leaf_size": 30,
            },
            "param_dist": {
                "classifier__n_neighbors": randint(3, 20), # Number of neighbors to use.

                # Weight function used in prediction.
                # 'uniform': all neighbors have equal weight.
                # 'distance': closer neighbors have greater influence.
                "classifier__weights": ["uniform", "distance"],

                # Power parameter for the Minkowski metric:
                # p=1 is equivalent to Manhattan distance, p=2 to Euclidean.
                "classifier__p": [1, 2]
            }
        },
        "DecisionTreeClassifier": {
            "model_class": DecisionTreeClassifier,
            "model_args": {
                "criterion": "gini", # The function to measure the quality of a split.

                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "splitter": "best",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__max_depth": randint(3, 20), # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(2, 10), # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(1, 10), # Minimum number of samples required at a leaf node.
                "classifier__max_features": ["sqrt", "log2"], # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(2, 20), # Maximum number of terminal nodes. Limits model complexity.

                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1),

                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            }
        },
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "model_args": {
                "criterion": "gini", # The function to measure the quality of a split.
                "bootstrap": True,  # Bootstrapping (sampling with replacement) enabled.
                "oob_score": False,

                "n_jobs": -1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300), # Number of trees in the forest.
                "classifier__max_depth": randint(3, 20), # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(2, 10), # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(1, 10), # Minimum number of samples required at a leaf node.
                "classifier__max_features": ["sqrt", "log2"], # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(2, 20), # Maximum number of terminal nodes. Limits model complexity.
                "classifier__max_samples": uniform(0.5, 0.5),

                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1), # Subsampling of rows per tree (when bootstrap=True).

                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            }
        },
        "ExtraTreesClassifier": {
            "model_class": ExtraTreesClassifier,
            "model_args": {
                "criterion": "gini", # The function to measure the quality of a split.
                "bootstrap": False,  # Each tree is trained using the whole learning sample (bootstrap = False)
                "max_samples": None,
                "oob_score": False,

                "n_jobs": -1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300), # Number of trees in the forest.
                "classifier__max_depth": randint(3, 20), # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(2, 10), # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(1, 10), # Minimum number of samples required at a leaf node.
                "classifier__max_features": ["sqrt", "log2"], # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(2, 20), # Maximum number of terminal nodes. Limits model complexity.

                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1), # Subsampling of rows per tree (when bootstrap=True).

                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            }
        },
        "BalancedRandomForestClassifier": {
            "model_class": BalancedRandomForestClassifier,
            "model_args": {
                "criterion": "gini",  # The function to measure the quality of a split.
                "bootstrap": False,  # Each tree is trained using the whole learning sample (bootstrap = False)
                "max_samples": None,
                "oob_score": False,
                "sampling_strategy": "all", # Sampling information to sample the data set: "all"=resample all classes
                "replacement": True, # Whether to sample randomly with replacement or not.
                "n_jobs": -1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300), # Number of trees in the forest.
                "classifier__max_depth": randint(3, 20), # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(2, 10), # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(1, 10), # Minimum number of samples required at a leaf node.
                "classifier__max_features": ["sqrt", "log2"], # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(2, 20), # Maximum number of terminal nodes. Limits model complexity.

                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1), # Subsampling of rows per tree (when bootstrap=True).

                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            }
        },
        "AdaBoostClassifier": {
            "model_class": AdaBoostClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(max_depth=1),
                "random_state": random_state
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),  # Number of weak learners

                # Weight applied to each classifier at each boosting iteration.
                # A higher learning rate increases the contribution of each classifier.
                "classifier__learning_rate": loguniform(1e-3, 1.0)
            }
        },
        "LogitBoostClassifier": {
            "model_class": GradientBoostingClassifier,
            "model_args": {
                "loss": "log_loss", # ‘log_loss’ refers to binomial and multinomial deviance, the same as used in logistic regression.
                "criterion": "friedman_mse",
                "subsample": 1.0,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300), # Number of boosting stages to perform.
                "classifier__max_depth": randint(3, 20),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(2, 10), # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(1, 10),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": ["sqrt", "log2"], # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(2, 20), # Maximum number of terminal nodes. Limits model complexity.

                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1), # Subsampling of rows per tree (when bootstrap=True).

                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),

                # Weight applied to each classifier at each boosting iteration.
                # A higher learning rate increases the contribution of each classifier.
                "classifier__learning_rate": loguniform(1e-3, 1.0)
            }
        },
        "XGBClassifier": {
            "model_class": XGBClassifier,
            "model_args": {
                "objective": "binary:logistic",  # Binary classification with logistic loss.
                "eval_metric": "logloss",  # Consistent with binary:logistic.
                "n_jobs": -1,  # Parallel training.
                "random_state": random_state
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of boosting rounds (trees).
                "classifier__max_depth": randint(3, 10),  # Maximum tree depth — lower = less overfitting.
                "classifier__learning_rate": loguniform(0.01, 0.3),  # Shrinks the contribution of each tree.
                "classifier__subsample": uniform(0.6, 0.4),  # Fraction of samples per tree. Helps generalization.
                "classifier__colsample_bytree": uniform(0.6, 0.4),
                # Fraction of features per tree. Avoids co-adaptation.
                "classifier__gamma": uniform(0.0, 5.0),  # Minimum loss reduction for a split. Acts as regularization.
                "classifier__reg_alpha": loguniform(1e-4, 10.0),  # L1 regularization on weights.
                "classifier__reg_lambda": loguniform(1e-4, 10.0),  # L2 regularization on weights.
                "classifier__scale_pos_weight": uniform(1.0, 10.0),  # Used to balance positive and negative weights.
                "classifier__min_child_weight": randint(1, 10),  # Minimum sum of instance weight (hessian) in child.
                "classifier__max_delta_step": randint(0, 10)  # Helps with logistic regression in imbalanced data.
            }
        },
        "LGBMClassifier": {
            "model_args": {
                "random_state": random_state,
                "n_jobs": -1,
                "is_unbalance": True  # Handles imbalance by adjusting weights
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of boosting rounds
                "classifier__learning_rate": loguniform(1e-3, 1e-1),  # Controls contribution of each tree
                "classifier__max_depth": randint(3, 20),  # Limits the depth of each tree
                "classifier__num_leaves": randint(20, 150),  # Number of leaves per tree
                "classifier__min_child_samples": randint(10, 100),  # Minimum data in one leaf
                "classifier__subsample": uniform(0.6, 0.4),  # Fraction of data used per tree
                "classifier__colsample_bytree": uniform(0.6, 0.4),  # Fraction of features per tree
                "classifier__reg_alpha": loguniform(1e-5, 1e-1),  # L1 regularization
                "classifier__reg_lambda": loguniform(1e-5, 1e-1),  # L2 regularization
            }
        },
        "RUSBoostClassifier": {
            "model_class": RUSBoostClassifier,
            "model_args": {
                "sampling_strategy": "auto",  # Sampling information to sample the data set: "auto"='not minority'.
                "replacement": False,  # Whether to sample randomly with replacement or not.
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),  # Number of weak learners

                # Weight applied to each classifier at each boosting iteration.
                # A higher learning rate increases the contribution of each classifier.
                "classifier__learning_rate": loguniform(1e-3, 1.0)
            }
        },
        "MLPClassifier": {
            "model_class": MLPClassifier,
            "model_args": {
                "random_state": random_state,
                "max_iter": 10000,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "activation": "relu",
                "solver": "adam",
            },
            "param_dist": {
                "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "classifier__alpha": loguniform(1e-5, 1e-2),  # Regularization strength
                "classifier__learning_rate_init": loguniform(1e-4, 1e-2)  # Initial learning rate
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
            "model_args": {
                "meta_classifier": LogisticRegression(
                                        solver='saga',
                                        random_state=0,
                                        class_weight='balanced',
                                        max_iter=10000)

            }
        }
    }

    des_config = des_model_configurations.get(model_name)

    if des_config is None:
        raise ValueError(f"Unknown DES model name: {model_name}")

    model_class = des_config["model_class"]
    model_args = des_config["model_args"]

    return model_class(pool_classifiers=pool_classifiers, **model_args)


def get_resampling_pipeline(
    resampler_name: str | None,
    random_state: int = 42
) -> ImbPipeline:
    """
    Return a resampling pipeline using one of the five selected strategies.

    Parameters
    ----------
    resampler_name : str | None
        One of the predefined resampling strategies:
        ['Under_0.005','Under_0.005_SMOTEENN']

    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    ImbPipeline
        A resampling pipeline ready to be inserted before the model in a full pipeline.

    Raises
    ------
    ValueError
        If the strategy name is not one of the predefined options.
    """
    strategy_dict = {
        "Under_0.005": ImbPipeline([
            ("undersample", RandomUnderSampler(sampling_strategy=0.005, random_state=random_state))
        ]),
        "Under_0.005_SMOTEENN": ImbPipeline([
            ("undersample", RandomUnderSampler(sampling_strategy=0.005, random_state=random_state)),
            ("resample", SMOTEENN(random_state=random_state))
        ])
    }

    if resampler_name not in strategy_dict:
        raise ValueError(f"Unknown resampler '{resampler_name}'. Available: {list(strategy_dict.keys())}")

    return strategy_dict[resampler_name]
