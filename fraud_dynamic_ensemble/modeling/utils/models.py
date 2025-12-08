from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from deslib.dcs import LCA, MLA, OLA, APosteriori, APriori
from deslib.des import DESKNN, DESP, KNOP, KNORAE, KNORAU, METADES, DESClustering
from deslib.des.probabilistic import DESKL, RRC, Exponential, Logarithmic
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _tree_common_param_space(
    prefix: str = "classifier__",
    max_depth_min: int = 3,
    max_depth_max: int = 20,
    min_samples_split_min: int = 2,
    min_samples_split_max: int = 10,
    min_samples_leaf_min: int = 1,
    min_samples_leaf_max: int = 10,
    max_leaf_nodes_min: int = 2,
    max_leaf_nodes_max: int = 20,
    min_impurity_decrease_min: float = 0.0,
    min_impurity_decrease_max: float = 0.1,
    ccp_alpha_min: float = 0.0,
    ccp_alpha_max: float = 0.01,
    max_features_choices: Iterable[str] = ("sqrt", "log2"),
) -> Dict[str, Any]:
    """
    Build a common hyperparameter search space for tree-based models.

    This helper returns a dictionary of SciPy distributions / lists for typical
    decision-tree structure parameters. It is intended to be merged into a
    `param_distributions` dictionary for `RandomizedSearchCV`.

    All integer ranges are interpreted using ``scipy.stats.randint(a, b)``,
    which samples integers in ``[a, b)`` (``b`` is exclusive). Floating ranges
    are interpreted as ``a + uniform(0, b - a)`` (i.e. continuous in
    ``[a, b)``).

    Parameters
    ----------
    prefix : str, optional
        Prefix for parameter names, usually the Pipeline step name plus
        ``"__"``. Defaults to ``"classifier__"``.
    max_depth_min : int, optional
        Minimum value for ``max_depth``. Default is 3.
    max_depth_max : int, optional
        Maximum value (exclusive) for ``max_depth``. Default is 20.
    min_samples_split_min : int, optional
        Minimum value for ``min_samples_split``. Default is 2.
    min_samples_split_max : int, optional
        Maximum value (exclusive) for ``min_samples_split``. Default is 10.
    min_samples_leaf_min : int, optional
        Minimum value for ``min_samples_leaf``. Default is 1.
    min_samples_leaf_max : int, optional
        Maximum value (exclusive) for ``min_samples_leaf``. Default is 10.
    max_leaf_nodes_min : int, optional
        Minimum value for ``max_leaf_nodes``. Default is 2.
    max_leaf_nodes_max : int, optional
        Maximum value (exclusive) for ``max_leaf_nodes``. Default is 20.
    min_impurity_decrease_min : float, optional
        Minimum value for ``min_impurity_decrease``. Default is 0.0.
    min_impurity_decrease_max : float, optional
        Maximum value (exclusive) for ``min_impurity_decrease``. Default is 0.1.
    ccp_alpha_min : float, optional
        Minimum value for ``ccp_alpha``. Default is 0.0.
    ccp_alpha_max : float, optional
        Maximum value (exclusive) for ``ccp_alpha``. Default is 0.01.
    max_features_choices : iterable of str, optional
        Categorical choices for ``max_features``. Default is
        ``("sqrt", "log2")``.

    Returns
    -------
    dict
        Dictionary mapping parameter names (with prefix) to SciPy distributions
        or lists, suitable for use in ``RandomizedSearchCV``:

        - ``<prefix>max_depth``
        - ``<prefix>min_samples_split``
        - ``<prefix>min_samples_leaf``
        - ``<prefix>max_features``
        - ``<prefix>max_leaf_nodes``
        - ``<prefix>min_impurity_decrease``
        - ``<prefix>ccp_alpha``

    Notes
    -----
    - ``max_depth``: Maximum depth of the tree. Controls overfitting.
    - ``min_samples_split``: Minimum number of samples to split an internal node.
    - ``min_samples_leaf``: Minimum number of samples required at a leaf node.
    - ``max_features``: Number of features considered at each split.
    - ``max_leaf_nodes``: Maximum number of terminal nodes. Limits complexity.
    - ``min_impurity_decrease``: A node is split if the impurity decrease
      is at least this value.
    - ``ccp_alpha``: Complexity parameter for Minimal Cost-Complexity Pruning.
    """
    return {
        f"{prefix}max_depth": randint(max_depth_min, max_depth_max),
        f"{prefix}min_samples_split": randint(min_samples_split_min, min_samples_split_max),
        f"{prefix}min_samples_leaf": randint(min_samples_leaf_min, min_samples_leaf_max),
        f"{prefix}max_features": list(max_features_choices),
        f"{prefix}max_leaf_nodes": randint(max_leaf_nodes_min, max_leaf_nodes_max),
        f"{prefix}min_impurity_decrease": uniform(
            min_impurity_decrease_min,
            min_impurity_decrease_max - min_impurity_decrease_min,
        ),
        f"{prefix}ccp_alpha": uniform(
            ccp_alpha_min,
            ccp_alpha_max - ccp_alpha_min,
        ),
    }


def _boosting_core_param_space(
    prefix: str = "classifier__",
    n_estimators_min: int = 100,
    n_estimators_max: int = 1000,
    learning_rate_min: float = 1e-3,
    learning_rate_max: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a common hyperparameter search space for boosting core parameters.

    This helper returns a dictionary for the two central hyperparameters of most
    boosting algorithms:

    - ``n_estimators``: number of boosting stages (additive steps).
    - ``learning_rate``: shrinkage factor applied at each boosting step.

    Both parameters are returned with names prefixed by ``prefix``, so that they
    can be used directly in a scikit-learn ``Pipeline`` where the boosting
    estimator is in a step named, for example, ``"classifier"``.

    Parameters
    ----------
    prefix : str, optional
        Prefix for parameter names, usually the Pipeline step name plus
        ``"__"``. Defaults to ``"classifier__"``.
    n_estimators_min : int, optional
        Minimum value for ``n_estimators``. Default is 100.
    n_estimators_max : int, optional
        Maximum value (exclusive) for ``n_estimators``. Default is 1000.
        The effective sampled range is ``[n_estimators_min, n_estimators_max)``.
    learning_rate_min : float, optional
        Minimum value for ``learning_rate`` (log-uniform lower bound).
        Must be strictly positive. Default is ``1e-3``.
    learning_rate_max : float, optional
        Maximum value for ``learning_rate`` (log-uniform upper bound).
        Must be strictly greater than ``learning_rate_min``.
        Default is ``1.0``.

    Returns
    -------
    dict
        Dictionary mapping:

        - ``<prefix>n_estimators`` → ``scipy.stats.randint``
        - ``<prefix>learning_rate`` → ``scipy.stats.loguniform``

        suitable for use in ``RandomizedSearchCV``.

    Notes
    -----
    - ``n_estimators`` controls the number of boosting stages.
      Larger values reduce bias but may increase variance and training time.
    - ``learning_rate`` scales the contribution of each stage (shrinkage).
      Smaller values typically require more estimators but improve regularization.
    """
    return {
        f"{prefix}n_estimators": randint(n_estimators_min, n_estimators_max),
        f"{prefix}learning_rate": loguniform(learning_rate_min, learning_rate_max),
    }


def get_static_model_and_search_space(
        model_name: str,
        random_state: int | None = None,
        use_cost_sensitive_learning: bool = True
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Build a classifier instance and its hyperparameter search space.

    This factory constructs a scikit-learn / imbalanced-learn / XGBoost
    estimator (optionally configured for cost-sensitive learning on
    imbalanced data) together with a parameter distribution dictionary
    intended for use with CV-based hyperparameter search, such as
    ``RandomizedSearchCV`` or ``HalvingRandomSearchCV``.

    The parameter names in the returned ``param_dist`` assume that your
    estimator is wrapped in a scikit-learn ``Pipeline`` step called
    ``"classifier"``, so all hyperparameters are prefixed with
    ``"classifier__"`` (e.g. ``"classifier__C"``, ``"classifier__n_estimators"``).

    Parameters
    ----------
    model_name : str
        Canonical model key. Supported values are::

            {
                'SVC',
                'MLPClassifier',
                'KNeighborsClassifier',
                'DecisionTreeClassifier',
                'RandomForestClassifier',
                'ExtraTreesClassifier',
                'BalancedRandomForestClassifier',
                'BaggingDecisionTreeClassifier',
                'AdaBoostClassifier',
                'LogitBoostClassifier',
                'XGBClassifier',
                'RUSBoostClassifier',
            }

    random_state : int or None, optional
        Random seed forwarded to models that accept it (e.g., trees, MLP,
        gradient boosting, XGBoost, imbalanced-learn ensembles). If ``None``,
        models use their library default.

    use_cost_sensitive_learning : bool, default=True
        If ``True``, models are configured (where applicable) to handle
        class imbalance via:

        - ``class_weight='balanced'`` or ``'balanced_subsample'`` for many
          tree-based classifiers.
        - Internal resampling in imbalanced-learn models such as
          ``BalancedRandomForestClassifier`` and ``RUSBoostClassifier`` via
          ``sampling_strategy``.
        - Tuning of ``scale_pos_weight`` in ``XGBClassifier``.

        If ``False``, these cost-sensitive mechanisms are disabled:

        - Any ``class_weight`` present in the model arguments is set to
          ``None`` (including the internal tree in the
          ``BaggingDecisionTreeClassifier``).
        - For ``XGBClassifier``, ``classifier__scale_pos_weight`` is removed
          from the search space and ``scale_pos_weight`` is fixed to ``1.0``.
        - For imbalanced-learn models that define ``sampling_strategy``
          (e.g., ``BalancedRandomForestClassifier``, ``RUSBoostClassifier``),
          it is set to ``None``.
        - For ``BalancedRandomForestClassifier``, both internal resampling
          and ``class_weight`` are disabled, making it behave closer to a
          standard ``RandomForestClassifier`` (with some overhead from the
          wrapper class).

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        The configured classifier instance (not fitted).
    param_dist : dict
        Mapping of hyperparameter names to SciPy distributions or lists,
        suitable for ``RandomizedSearchCV`` or ``HalvingRandomSearchCV``
        (via the ``param_distributions=...`` argument). All names are
        prefixed with ``"classifier__"`` to match a Pipeline step named
        ``"classifier"``.

    Raises
    ------
    ValueError
        If ``model_name`` is not one of the supported keys.
    ImportError
        If the requested model requires an optional dependency that is not
        installed (e.g., ``xgboost``, ``imbalanced-learn``).

    Notes
    -----
    Imbalance handling (when ``use_cost_sensitive_learning=True``)
        - Many tree-based models are initialized with
          ``class_weight='balanced'`` or ``'balanced_subsample'``.
        - ``BalancedRandomForestClassifier`` and ``RUSBoostClassifier``
          (from imbalanced-learn) perform internal resampling via
          ``sampling_strategy``.
        - ``XGBClassifier`` includes ``scale_pos_weight`` in the search
          space to handle strong class imbalance.

    Distributions and shared helpers
        - Integer ranges use ``scipy.stats.randint(a, b)`` which samples
          integers in ``[a, b)``.
        - Continuous ranges use ``scipy.stats.uniform(a, b - a)`` to sample
          values in ``[a, b)``.
        - Positive, multiplicative ranges (e.g. ``C``, ``gamma``, learning
          rates, regularization strengths) use
          ``scipy.stats.loguniform(low, high)``.
        - Tree-based models share a common structural search space via
          ``_tree_common_param_space()``, which controls:
          ``max_depth``, ``min_samples_split``, ``min_samples_leaf``,
          ``max_features``, ``max_leaf_nodes``, ``min_impurity_decrease``,
          and ``ccp_alpha``.
        - Boosting algorithms (AdaBoost, LogitBoost, XGBoost, RUSBoost)
          share a core search space via ``_boosting_core_param_space()``,
          which tunes ``n_estimators`` and ``learning_rate``; each model
          then adds its own structural and regularization parameters.

    Model-specific behaviour
        - SVC:
          ``probability=True`` is enabled to provide calibrated probabilities
          (slower than using the decision function). The search space covers
          ``C``, ``gamma``, and the kernel family
          ``{'rbf', 'poly', 'linear'}``, including the polynomial degree.
        - MLPClassifier:
          Uses ``early_stopping=True`` with a large ``max_iter`` and searches
          over hidden layer sizes, L2 regularization (``alpha``), initial
          learning rate, and ``batch_size``.
        - Tree ensembles (DecisionTree, RandomForest, ExtraTrees,
          BalancedRandomForest):
          Share the tree-structure search space (depth, splits, leaves,
          features, leaf nodes, impurity decrease, pruning strength), with
          additional model-specific parameters such as the number of trees
          and ``max_samples`` for bagging.
        - BaggingDecisionTreeClassifier:
          Wraps a class-balanced ``DecisionTreeClassifier`` inside a
          ``BaggingClassifier``. The search space includes both
          bagging-level parameters (``n_estimators``, ``max_samples``,
          ``max_features``) and the internal tree structure, using the
          ``"classifier__estimator__"`` prefix (e.g.
          ``classifier__estimator__max_depth``).
        - Gradient/Ada/LogitBoost:
          ``LogitBoostClassifier`` is implemented with
          ``GradientBoostingClassifier(loss='log_loss')`` and tunes the
          boosting core parameters, shallow tree structure (depth, leaf size,
          max features), and subsampling rate.
          ``AdaBoostClassifier`` and ``RUSBoostClassifier`` tune only the
          boosting core (number of stages and learning rate), using decision
          stumps as base learners by default.
        - XGBClassifier:
          Tunes boosting core parameters, tree capacity
          (``max_depth``, ``min_child_weight``), stochasticity
          (``subsample``, ``colsample_bytree``), regularization
          (``reg_alpha``, ``reg_lambda``, ``gamma``), and
          ``scale_pos_weight`` for class imbalance (unless explicitly
          disabled via ``use_cost_sensitive_learning=False``).

    Pipeline naming
        The returned ``param_dist`` assumes that the estimator is wrapped in
        a Pipeline step named ``"classifier"``. For example:

        ``Pipeline([('classifier', <estimator>)])``

        so that all hyperparameters can be referenced as
        ``"classifier__<param_name>"`` in ``RandomizedSearchCV`` or
        ``HalvingRandomSearchCV``.

    Examples
    --------
    Build a pipeline and randomized search for a Random Forest:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> clf, space = get_static_model_and_search_space(
    ...     'RandomForestClassifier',
    ...     random_state=42,
    ... )
    >>> pipe = Pipeline([('classifier', clf)])
    >>> search = RandomizedSearchCV(
    ...     estimator=pipe,
    ...     param_distributions=space,
    ...     n_iter=30,
    ...     cv=5,
    ...     scoring='average_precision',
    ...     random_state=42,
    ...     n_jobs=-1,
    ... )

    Switch to XGBoost with the same interface, disabling cost-sensitive tweaks:

    >>> clf, space = get_static_model_and_search_space(
    ...     'XGBClassifier',
    ...     random_state=42,
    ...     use_cost_sensitive_learning=False,
    ... )
    >>> pipe = Pipeline([('classifier', clf)])
    >>> search = RandomizedSearchCV(
    ...     estimator=pipe,
    ...     param_distributions=space,
    ...     n_iter=50,
    ...     cv=5,
    ...     n_jobs=-1,
    ...     random_state=42,
    ... )
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
                # Kernel coefficient for 'rbf' and 'poly' kernels.
                "classifier__gamma": loguniform(1e-4, 1e0),
                # Kernel type
                "classifier__kernel": ["rbf", "poly", "linear"],
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
                # Regularization strength
                "classifier__alpha": loguniform(1e-5, 1e-2),
                # Initial learning rate
                "classifier__learning_rate_init": loguniform(1e-4, 1e-2),
                "classifier__batch_size": [16, 32, 64, 128],
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
                # Number of neighbors to use.
                "classifier__n_neighbors": randint(3, 20),
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
                # The function to measure the quality of a split.
                "criterion": "gini",
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "splitter": "best",
                "random_state": random_state,
            },
            "param_dist": _tree_common_param_space(),
        },
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "model_args": {
                # The function to measure the quality of a split.
                "criterion": "gini",
                # Bootstrapping (sampling with replacement) enabled.
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": 1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                # Number of trees in the forest.
                "classifier__n_estimators": randint(100, 1000),
                # Controls the size of the bootstrap sample (the subset of data)
                # used to train each individual decision tree in the forest: [0.5, 0.5 + 0.4] = [0.5, 0.9]
                "classifier__max_samples": uniform(0.5, 0.4),
                **_tree_common_param_space(),
            },
        },
        "ExtraTreesClassifier": {
            "model_class": ExtraTreesClassifier,
            "model_args": {
                # The function to measure the quality of a split.
                "criterion": "gini",
                # Each tree is trained using the whole learning sample (bootstrap = False, max_samples = None)
                "bootstrap": False,
                "max_samples": None,
                "oob_score": False,
                "n_jobs": 1,
                # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                # adjust weights inversely proportional to class frequencies in the input data.
                "class_weight": "balanced",
                "random_state": random_state,
            },
            "param_dist": {
                # Number of trees in the forest.
                "classifier__n_estimators": randint(100, 1000),
                **_tree_common_param_space(),
            },
        },
        "BalancedRandomForestClassifier": {
            "model_class": BalancedRandomForestClassifier,
            "model_args": {
                # The function to measure the quality of a split.
                "criterion": "gini",
                # Each tree is trained using the whole learning sample (bootstrap = False, max_samples = None)
                # Bootstrapping is already taken care by the internal sampler using replacement=True
                "bootstrap": False,
                "max_samples": None,
                "oob_score": False,
                # Sampling information to sample the data set: "all"=resample all classes
                "sampling_strategy": "all",
                # Whether to sample randomly with replacement or not.
                "replacement": True,
                "n_jobs": 1,
                # The “balanced_subsample” mode is the same as “balanced” except
                # that weights are computed based on the bootstrap sample for every tree grown.
                "class_weight": "balanced_subsample",
                "random_state": random_state,
            },
            "param_dist": {
                # Number of trees in the forest.
                "classifier__n_estimators": randint(100, 1000),
                **_tree_common_param_space(),
            },
        },
        "BaggingDecisionTreeClassifier": {
            "model_class": BaggingClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(
                    # The function to measure the quality of a split.
                    criterion="gini",
                    # Weights associated with classes. The “balanced” mode uses the values of y to automatically
                    # adjust weights inversely proportional to class frequencies in the input data.
                    class_weight="balanced",
                    splitter="best",
                    random_state=random_state,
                ),
            },
            "param_dist": {
                # --- Bagging-level hyperparameters ---
                # Number of trees in the ensemble
                "classifier__n_estimators": randint(100, 1000),
                # Fraction of samples used per base estimator: [0.5, 1.0)
                "classifier__max_samples": uniform(0.5, 0.5),
                # Fraction of features used per base estimator: [0.5, 1.0)
                "classifier__max_features": uniform(0.5, 0.5),
                # --- Internal DecisionTree hyperparameters ---
                # Reuse the common tree space but for the *internal* estimator
                # i.e. "classifier__estimator__max_depth", etc.
                **_tree_common_param_space(prefix="classifier__estimator__"),
            },
        },
        "AdaBoostClassifier": {
            "model_class": AdaBoostClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(max_depth=1),
                "random_state": random_state,
            },
            "param_dist": _boosting_core_param_space(
                n_estimators_min=50,
                n_estimators_max=800,
                learning_rate_min=1e-3,
                learning_rate_max=1.0,
            ),
        },
        "LogitBoostClassifier": {
            "model_class": GradientBoostingClassifier,
            "model_args": {
                # log_loss refers to binomial and multinomial deviance,
                # the same as used in logistic regression.
                "loss": "log_loss",
                # Function to measure the quality of a split.
                "criterion": "friedman_mse",
                # Fraction of samples to be used for fitting the individual base learners.
                "subsample": 1.0,
                # Early stopping criteria
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "random_state": random_state,
            },
            "param_dist": {
                **_boosting_core_param_space(
                    n_estimators_min=50,
                    n_estimators_max=400,
                    learning_rate_min=1e-2,
                    learning_rate_max=0.3,
                ),
                # Maximum depth of the tree. Controls overfitting.
                "classifier__max_depth": randint(1, 4),
                # Minimum number of samples required to split an internal node.
                "classifier__min_samples_split": randint(2, 21),
                # Minimum number of samples required at a leaf node.
                "classifier__min_samples_leaf": randint(1, 10),
                # Number of features to consider when looking for the best split.
                "classifier__max_features": ["sqrt", "log2"],
                # Subsampling of rows per tree (when bootstrap=True).
                "classifier__max_leaf_nodes": randint(2, 20),
                # A node will be split if this split induces a decrease of the
                # impurity greater than or equal to this value.
                "classifier__min_impurity_decrease": uniform(0.0, 0.1),
                # Complexity parameter used for Minimal Cost-Complexity Pruning.
                # Values typically very small (0.0 to ~0.05).
                "classifier__ccp_alpha": uniform(0.0, 0.01),
            },
        },
        "XGBClassifier": {
            "model_class": XGBClassifier,
            "model_args": {
                # Binary classification with logistic loss.
                "objective": "binary:logistic",
                # Consistent with binary:logistic.
                "eval_metric": "logloss",
                "n_jobs": 1,
                "random_state": random_state,
            },
            "param_dist": {
                **_boosting_core_param_space(
                    n_estimators_min=200,
                    n_estimators_max=800,
                    learning_rate_min=1e-2,
                    learning_rate_max=0.2,
                ),
                # Maximum tree depth — lower = less overfitting.
                "classifier__max_depth": randint(3, 10),
                # Fraction of samples per tree. Helps generalization: [0.6, 0.6 + 0.4] = [0.6, 1.0]
                "classifier__subsample": uniform(0.6, 0.4),
                # Fraction of features per tree. Avoids co-adaptation.
                "classifier__colsample_bytree": uniform(0.6, 0.4),
                # Minimum loss reduction for a split. Acts as regularization.
                "classifier__gamma": uniform(0.0, 5.0),
                # L1 regularization on weights.
                "classifier__reg_alpha": loguniform(1e-4, 10.0),
                # L2 regularization on weights.
                "classifier__reg_lambda": loguniform(1e-4, 10.0),
                # Used to balance positive and negative weights.
                "classifier__scale_pos_weight": loguniform(0.5, 50.0),
                # Minimum sum of instance weight (hessian) in child.
                "classifier__min_child_weight": randint(1, 10),
                # Helps with logistic regression in imbalanced data.
                "classifier__max_delta_step": randint(0, 10),
            },
        },
        "RUSBoostClassifier": {
            "model_class": RUSBoostClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(max_depth=1),
                # Sampling information to sample the data set: "auto"='not minority'.
                "sampling_strategy": "auto",
                # Whether to sample randomly with replacement or not.
                "replacement": False,
                "random_state": random_state,
            },
            "param_dist": _boosting_core_param_space(
                n_estimators_min=50,
                n_estimators_max=800,
                learning_rate_min=1e-3,
                learning_rate_max=1.0,
            ),
        },
    }

    if model_name not in model_configurations:
        raise ValueError(f"Unknown model name: {model_name}")

    config = model_configurations[model_name]

    if not use_cost_sensitive_learning:
        # Standard Sklearn 'class_weight' (SVC, RF, DT, ExtraTrees)
        if "class_weight" in config["model_args"]:
            config["model_args"]["class_weight"] = None

        # Nested Estimator (BaggingClassifier)
        if model_name == "BaggingDecisionTreeClassifier":
            # Set the internal tree's weights to None
            config["model_args"]["estimator"].class_weight = None

        # XGBClassifier (scale_pos_weight)
        if model_name == "XGBClassifier":
            # Remove from search space to prevent tuning it
            if "classifier__scale_pos_weight" in config["param_dist"]:
                del config["param_dist"]["classifier__scale_pos_weight"]
            # Force default behavior (1.0 = equal weight)
            config["model_args"]["scale_pos_weight"] = 1.0

        # Imbalanced-Learn Models (BalancedRF, RUSBoost)
        # If cost sensitivity is off, we disable the internal resampling strategy.
        if "sampling_strategy" in config["model_args"]:
            config["model_args"]["sampling_strategy"] = None

        # BalancedRF specific: if we turn off sampling, it behaves like a standard RF
        # but with the overhead of the BalancedRF class structure.
        if model_name == "BalancedRandomForestClassifier":
            if "class_weight" in config["model_args"]:
                config["model_args"]["class_weight"] = None

    model = config["model_class"](**config["model_args"])

    return model, config["param_dist"]


def get_des_model(
        model_name: str,
        random_state: int | None = None,
        use_cost_sensitive_learning: bool = True,
) -> Tuple[BaseEstimator, Dict[str, Any], BaseEstimator, Dict[str, Any]]:
    """
    Build the **pool (bagging) configuration** and an **unfitted DESlib model**
    for a Dynamic Ensemble Selection (DES) workflow.

    This factory returns:
      1) a **bagging pool estimator** (intended for the ``"classifier"`` step of your
         training pipeline) **plus** its hyperparameter search space; and
      2) an **unfitted DES model instance** **plus** a dictionary of default kwargs
         you can apply later (e.g., via ``set_params``) together with the tuned pool.

    The typical two-stage process is:

    1. **Pool tuning (TRAIN)**:
       plug ``pool_estimator`` into your pipeline, tune it with
       ``pool_param_dist`` on the training data, and extract the tuned bagger
       (e.g., ``best_pipe.named_steps["classifier"]``).

    2. **DES fitting (DSEL)**:
       inject the tuned pool into the returned ``des_model`` via
       ``des_model.set_params(pool_classifiers=fitted_pool)``, then fit it on
       preprocessed DSEL data.

    Parameters
    ----------
    model_name : {"APriori", "APosteriori", "LCA", "MLA", "OLA",
                  "KNORAE", "KNORAU", "DESP", "DESKNN", "DESClustering",
                  "KNOP", "DESKL", "Exponential", "Logarithmic", "RRC", "METADES"}
        Name of the DES method to instantiate.
    random_state : int or None, optional
        Forwarded to the **pool** via
        :func:`get_static_model_and_search_space("BaggingDecisionTreeClassifier")`.
        The DES instance itself is created with library defaults; if you need to
        control its randomness (when supported), apply ``des_kwargs`` via
        ``des_model.set_params(**des_kwargs)`` or pass overrides manually.
    use_cost_sensitive_learning : bool, default=True
        Whether to configure the **pool** for cost-sensitive learning on
        imbalanced data. This flag is passed directly to
        :func:`get_static_model_and_search_space("BaggingDecisionTreeClassifier")`:

        - If ``True``, the internal ``DecisionTreeClassifier`` and the bagging
          ensemble are configured with imbalance-aware defaults (e.g.,
          ``class_weight='balanced'``).
        - If ``False``, the bagging pool is built without cost-sensitive tweaks
          (no class weights, no internal imbalance heuristics).

    Returns
    -------
    pool_estimator : sklearn.ensemble.BaggingClassifier
        Bagging ensemble configured (optionally) for imbalanced data. The base
        estimator is a ``DecisionTreeClassifier``; use this in your pipeline's
        ``"classifier"`` step during the pool tuning stage.
    pool_param_dist : dict
        Hyperparameter search space for the pool, including bagging-level params
        (e.g., ``classifier__n_estimators``, ``classifier__max_samples``,
        ``classifier__max_features``) and internal tree params
        (e.g., ``classifier__estimator__max_depth``, ``...__min_samples_leaf``,
        ``...__ccp_alpha``). Suitable for use with ``RandomizedSearchCV`` or
        ``HalvingRandomSearchCV`` via the ``param_distributions=...`` argument.
    des_model : sklearn.base.BaseEstimator
        **Unfitted** DESlib estimator instance (e.g., ``KNORAE``, ``METADES``).
        Created with library defaults (no kwargs applied yet). You are expected
        to inject the tuned pool later with:

        ``des_model.set_params(pool_classifiers=fitted_pool)``.
    des_kwargs : dict
        Suggested default keyword arguments for the chosen DES (e.g., ``k``,
        ``DFP``, ``IH_rate``, ``voting``, ``n_jobs``). This dict **does not**
        include ``pool_classifiers``; you must add it before fitting.

    Raises
    ------
    ValueError
        If an unsupported ``model_name`` is provided.

    Notes
    -----
    - **Pool format:**
      Many DES methods accept either a list of fitted base estimators or the
      fitted bagging object itself as ``pool_classifiers``. Choose according
      to the DES method’s API and your design choice.
    - **Preprocessing alignment:**
      The DSEL set must be transformed with the **same fitted preprocessing**
      used for pool training before calling ``des_model.fit``. Typically this
      is handled by building a final pipeline:

      ``Pipeline(preprocessing_steps + [('classifier', des_model)])``

      and calling ``final_pipeline.fit(X_dsel_trans, y_dsel)``.

    Examples
    --------
    >>> pool_est, pool_space, des, des_kwargs = get_des_model(
    ...     "KNORAE",
    ...     random_state=42,
    ...     use_cost_sensitive_learning=True,
    ... )
    """
    # Pool: BaggingDecisionTreeClassifier + its search space
    pool_estimator, pool_param_dist = get_static_model_and_search_space(
        model_name="BaggingDecisionTreeClassifier",
        random_state=random_state,
        use_cost_sensitive_learning=use_cost_sensitive_learning,
    )

    # DES model configuration (class + default kwargs)
    des_model_configurations = {
        "APriori": {
            "model_class": APriori,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "APosteriori": {
            "model_class": APosteriori,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "LCA": {
            "model_class": LCA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "MLA": {
            "model_class": MLA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "OLA": {
            "model_class": OLA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "KNORAE": {
            "model_class": KNORAE,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "KNORAU": {
            "model_class": KNORAU,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESP": {
            "model_class": DESP,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESKNN": {
            "model_class": DESKNN,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "pct_accuracy": 0.5,
                "pct_diversity": 0.3,
                "more_diverse": True,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESClustering": {
            "model_class": DESClustering,
            "model_args": {
                "pct_accuracy": 0.5,
                "pct_diversity": 0.3,
                "more_diverse": True,
                "metric_performance": "accuracy_score",
                "n_clusters": 5,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "KNOP": {
            "model_class": KNOP,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESKL": {
            "model_class": DESKL,
            "model_args": {
                "k": 8,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "Exponential": {
            "model_class": Exponential,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "Logarithmic": {
            "model_class": Logarithmic,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "RRC": {
            "model_class": RRC,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "METADES": {
            "model_class": METADES,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
    }

    if model_name not in des_model_configurations:
        raise ValueError(f"Unknown DES model name: {model_name}")

    des_config = des_model_configurations[model_name]
    model = des_config["model_class"]()
    model_args = des_config["model_args"]

    return pool_estimator, pool_param_dist, model, model_args
