from __future__ import annotations

from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_base_model_and_search_space(model_name: str, random_state: int | None = None) -> tuple:
    """
    Return a classifier instance and its hyperparameter search space.

    This factory builds a scikit-learn/imbalanced-learn/XGBoost/LightGBM estimator
    (configured with sensible defaults for class imbalance where applicable) and a
    parameter distribution dictionary intended for `RandomizedSearchCV`.

    The parameter names in the returned `param_dist` assume your estimator lives in a
    scikit-learn `Pipeline` step called **"classifier"** (e.g., `"classifier__C"`).

    Parameters
    ----------
    model_name : str
        Canonical model key. Supported:
        {'SVC', 'MLPClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier',
         'RandomForestClassifier', 'ExtraTreesClassifier', 'BalancedRandomForestClassifier',
         'AdaBoostClassifier', 'LogitBoostClassifier', 'XGBClassifier',
         'RUSBoostClassifier'}.
    random_state : int or None, optional
        Random seed forwarded to models that accept it (e.g., trees, MLP, XGB/LGBM,
        imblearn ensembles). If `None`, models use their library default.

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        The configured classifier instance (not fitted).
    param_dist : dict
        Mapping of hyperparameter names → SciPy distributions / lists, suitable for
        `RandomizedSearchCV(param_distributions=...)`. Names are prefixed with
        `"classifier__"` to match a Pipeline step named `"classifier"`.

    Raises
    ------
    ValueError
        If `model_name` is not one of the supported keys.
    ImportError
        If the requested model requires an optional dependency that is not installed
        (e.g., `xgboost`, `imbalanced-learn`).

    Notes
    -----
    - **Imbalance handling**:
      - Many tree models are initialized with `class_weight='balanced'`.
      - `BalancedRandomForestClassifier` and `RUSBoostClassifier` (imblearn) perform
        internal resampling; `sampling_strategy='all'` for BRF is set via defaults.
      - `XGBClassifier` includes `scale_pos_weight` in the search space.
    - **Distributions**:
      - Spaces use `scipy.stats` `randint`, `uniform`, and `loguniform` (for positive,
        multiplicative ranges such as `C`, `gamma`, learning rates, regularization).
    - **SVC**:
      - `probability=True` enables calibrated probabilities (slower than decision
        function). Kernels searched: `{'rbf','poly','linear'}` with degree for `poly`.
    - **MLPClassifier**:
      - Uses `early_stopping=True` and a large `max_iter` to converge reliably.
    - **Tree ensembles**:
      - Random/Extra Trees search typical depth/split/leaf/max_features ranges; some
        include `max_leaf_nodes`, `min_impurity_decrease`, and `ccp_alpha`.
    - **Gradient/Ada/LogitBoost**:
      - `LogitBoostClassifier` here is implemented with `GradientBoostingClassifier`
        and `loss='log_loss'`; learning rate and structure are part of the search.
    - **Pipeline naming**:
      - The `"classifier__"` prefix in `param_dist` requires:
        `Pipeline([('classifier', <estimator>)])`.

    Examples
    --------
    Build a pipeline and randomized search:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> clf, space = get_base_model_and_search_space('RandomForestClassifier', random_state=42)
    >>> pipe = Pipeline([('classifier', clf)])
    >>> search = RandomizedSearchCV(
    ...     estimator=pipe,
    ...     param_distributions=space,
    ...     n_iter=30,
    ...     cv=5,
    ...     scoring='average_precision',
    ...     random_state=42,
    ...     n_jobs=-1
    ... )

    Swap to XGBoost:

    >>> clf, space = get_base_model_and_search_space('XGBClassifier', random_state=42)
    >>> pipe = Pipeline([('classifier', clf)])
    >>> search = RandomizedSearchCV(pipe, space, n_iter=50, cv=5, n_jobs=-1, random_state=42)
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
                "classifier__gamma": loguniform(
                    1e-4, 1e0
                ),  # Kernel coefficient for 'rbf' and 'poly' kernels.
                "classifier__kernel": ["rbf", "poly", "linear"],  # Kernel type
                # Degree of the polynomial kernel function (only relevant if kernel='poly').
                "classifier__degree": randint(2, 5),
            },
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
                "classifier__hidden_layer_sizes": [
                    (32,),
                    (64,),
                    (128,),
                    (64, 32),
                    (128, 32),
                    (128, 64),
                    (128, 64, 32),
                ],
                "classifier__alpha": loguniform(1e-5, 1e-2),  # Regularization strength
                "classifier__learning_rate_init": loguniform(1e-4, 1e-2),  # Initial learning rate
            },
        },
        "KNeighborsClassifier": {
            "model_class": KNeighborsClassifier,
            "model_args": {
                "n_jobs": 1,
                "algorithm": "auto",
                "leaf_size": 30,
            },
            "param_dist": {
                "classifier__n_neighbors": randint(3, 20),  # Number of neighbors to use.
                # Weight function used in prediction.
                # 'uniform': all neighbors have equal weight.
                # 'distance': closer neighbors have greater influence.
                "classifier__weights": ["uniform", "distance"],
                # Power parameter for the Minkowski metric:
                # p=1 is equivalent to Manhattan distance, p=2 to Euclidean.
                "classifier__p": [1, 2],
            },
        },
        "DecisionTreeClassifier": {
            "model_class": DecisionTreeClassifier,
            "model_args": {
                "criterion": "gini",  # The function to measure the quality of a split.
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "splitter": "best",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__max_depth": randint(
                    3, 20
                ),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(
                    2, 10
                ),  # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(
                    1, 10
                ),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": [
                    "sqrt",
                    "log2",
                ],  # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(
                    2, 20
                ),  # Maximum number of terminal nodes. Limits model complexity.
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1),
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            },
        },
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "model_args": {
                "criterion": "gini",  # The function to measure the quality of a split.
                "bootstrap": True,  # Bootstrapping (sampling with replacement) enabled.
                "oob_score": False,
                "n_jobs": 1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of trees in the forest.
                "classifier__max_depth": randint(
                    3, 20
                ),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(
                    2, 10
                ),  # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(
                    1, 10
                ),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": [
                    "sqrt",
                    "log2",
                ],  # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(
                    2, 20
                ),  # Maximum number of terminal nodes. Limits model complexity.
                "classifier__max_samples": uniform(0.5, 0.5),
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(
                    0.0, 0.1
                ),  # Subsampling of rows per tree (when bootstrap=True).
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            },
        },
        "ExtraTreesClassifier": {
            "model_class": ExtraTreesClassifier,
            "model_args": {
                "criterion": "gini",  # The function to measure the quality of a split.
                "bootstrap": False,  # Each tree is trained using the whole learning sample (bootstrap = False)
                "max_samples": None,
                "oob_score": False,
                "n_jobs": 1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of trees in the forest.
                "classifier__max_depth": randint(
                    3, 20
                ),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(
                    2, 10
                ),  # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(
                    1, 10
                ),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": [
                    "sqrt",
                    "log2",
                ],  # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(
                    2, 20
                ),  # Maximum number of terminal nodes. Limits model complexity.
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(
                    0.0, 0.1
                ),  # Subsampling of rows per tree (when bootstrap=True).
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            },
        },
        "BalancedRandomForestClassifier": {
            "model_class": BalancedRandomForestClassifier,
            "model_args": {
                "criterion": "gini",  # The function to measure the quality of a split.
                "bootstrap": False,  # Each tree is trained using the whole learning sample (bootstrap = False)
                "max_samples": None,
                "oob_score": False,
                "sampling_strategy": "all",  # Sampling information to sample the data set: "all"=resample all classes
                "replacement": True,  # Whether to sample randomly with replacement or not.
                "n_jobs": 1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of trees in the forest.
                "classifier__max_depth": randint(
                    3, 20
                ),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(
                    2, 10
                ),  # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(
                    1, 10
                ),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": [
                    "sqrt",
                    "log2",
                ],  # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(
                    2, 20
                ),  # Maximum number of terminal nodes. Limits model complexity.
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(
                    0.0, 0.1
                ),  # Subsampling of rows per tree (when bootstrap=True).
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            },
        },
        "AdaBoostClassifier": {
            "model_class": AdaBoostClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(max_depth=1),
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 200),  # Number of weak learners
                # Weight applied to each classifier at each boosting iteration.
                # A higher learning rate increases the contribution of each classifier.
                "classifier__learning_rate": loguniform(1e-3, 1.0),
            },
        },
        "LogitBoostClassifier": {
            "model_class": GradientBoostingClassifier,
            "model_args": {
                "loss": "log_loss",  # ‘log_loss’ refers to binomial and multinomial deviance, the same as used in logistic regression.
                "criterion": "friedman_mse",
                "subsample": 1.0,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(
                    50, 300
                ),  # Number of boosting stages to perform.
                "classifier__max_depth": randint(
                    3, 20
                ),  # Maximum depth of the tree. Controls overfitting.
                "classifier__min_samples_split": randint(
                    2, 10
                ),  # Minimum number of samples required to split an internal node.
                "classifier__min_samples_leaf": randint(
                    1, 10
                ),  # Minimum number of samples required at a leaf node.
                "classifier__max_features": [
                    "sqrt",
                    "log2",
                ],  # Number of features to consider when looking for the best split.
                "classifier__max_leaf_nodes": randint(
                    2, 20
                ),  # Maximum number of terminal nodes. Limits model complexity.
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(
                    0.0, 0.1
                ),  # Subsampling of rows per tree (when bootstrap=True).
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
                # Weight applied to each classifier at each boosting iteration.
                # A higher learning rate increases the contribution of each classifier.
                "classifier__learning_rate": loguniform(1e-3, 1.0),
            },
        },
        "XGBClassifier": {
            "model_class": XGBClassifier,
            "model_args": {
                "objective": "binary:logistic",  # Binary classification with logistic loss.
                "eval_metric": "logloss",  # Consistent with binary:logistic.
                "n_jobs": 1,  # Parallel training.py.
                "random_state": random_state,
            },
            "param_dist": {
                "classifier__n_estimators": randint(50, 300),  # Number of boosting rounds (trees).
                "classifier__max_depth": randint(
                    3, 10
                ),  # Maximum tree depth — lower = less overfitting.
                "classifier__learning_rate": loguniform(
                    0.01, 0.3
                ),  # Shrinks the contribution of each tree.
                "classifier__subsample": uniform(
                    0.6, 0.4
                ),  # Fraction of samples per tree. Helps generalization.
                "classifier__colsample_bytree": uniform(0.6, 0.4),
                # Fraction of features per tree. Avoids co-adaptation.
                "classifier__gamma": uniform(
                    0.0, 5.0
                ),  # Minimum loss reduction for a split. Acts as regularization.
                "classifier__reg_alpha": loguniform(1e-4, 10.0),  # L1 regularization on weights.
                "classifier__reg_lambda": loguniform(1e-4, 10.0),  # L2 regularization on weights.
                "classifier__scale_pos_weight": uniform(
                    1.0, 10.0
                ),  # Used to balance positive and negative weights.
                "classifier__min_child_weight": randint(
                    1, 10
                ),  # Minimum sum of instance weight (hessian) in child.
                "classifier__max_delta_step": randint(
                    0, 10
                ),  # Helps with logistic regression in imbalanced data.
            },
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
                "classifier__learning_rate": loguniform(1e-3, 1.0),
            },
        },
    }

    if model_name not in model_configurations:
        raise ValueError(f"Unknown model name: {model_name}")

    config = model_configurations[model_name]
    model = config["model_class"](**config["model_args"])
    return model, config["param_dist"]
