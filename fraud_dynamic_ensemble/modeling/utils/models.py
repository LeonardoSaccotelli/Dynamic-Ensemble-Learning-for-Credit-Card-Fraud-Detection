from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

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
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
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
    Build a shared hyperparameter search space for tree-based classifiers.

    This helper returns a ``param_distributions`` dictionary suitable for
    ``RandomizedSearchCV`` (or compatible search utilities). Integer-valued
    parameters are sampled using ``scipy.stats.randint(a, b)`` (support ``[a, b)``),
    while continuous parameters are sampled using ``scipy.stats.uniform(loc, scale)``
    (support ``[loc, loc + scale)``). Parameter keys are prefixed (typically with a
    pipeline step name such as ``'classifier__'``).

    Parameters
    ----------
    prefix : str, default 'classifier__'
        Prefix prepended to every parameter name (e.g., pipeline step name plus ``'__'``).
    max_depth_min : int, default 3
        Minimum value for ``max_depth`` (inclusive).
    max_depth_max : int, default 20
        Maximum value for ``max_depth`` (exclusive).
    min_samples_split_min : int, default 2
        Minimum value for ``min_samples_split`` (inclusive).
    min_samples_split_max : int, default 10
        Maximum value for ``min_samples_split`` (exclusive).
    min_samples_leaf_min : int, default 1
        Minimum value for ``min_samples_leaf`` (inclusive).
    min_samples_leaf_max : int, default 10
        Maximum value for ``min_samples_leaf`` (exclusive).
    max_leaf_nodes_min : int, default 2
        Minimum value for ``max_leaf_nodes`` (inclusive).
    max_leaf_nodes_max : int, default 20
        Maximum value for ``max_leaf_nodes`` (exclusive).
    min_impurity_decrease_min : float, default 0.0
        Lower bound for ``min_impurity_decrease`` (inclusive).
    min_impurity_decrease_max : float, default 0.1
        Upper bound for ``min_impurity_decrease`` (exclusive).
    ccp_alpha_min : float, default 0.0
        Lower bound for ``ccp_alpha`` (inclusive).
    ccp_alpha_max : float, default 0.01
        Upper bound for ``ccp_alpha`` (exclusive).
    max_features_choices : Iterable[str], default ('sqrt', 'log2')
        Categorical choices for ``max_features``.

    Returns
    -------
    param_distributions : dict[str, Any]
        Mapping from prefixed parameter names to SciPy distributions / categorical lists:
        - ``<prefix>max_depth`` : ``randint``
        - ``<prefix>min_samples_split`` : ``randint``
        - ``<prefix>min_samples_leaf`` : ``randint``
        - ``<prefix>max_features`` : ``list[str]``
        - ``<prefix>max_leaf_nodes`` : ``randint``
        - ``<prefix>min_impurity_decrease`` : ``uniform``
        - ``<prefix>ccp_alpha`` : ``uniform``

    Notes
    -----
    - For ``randint(a, b)``, SciPy samples integers in ``[a, b)`` (upper bound excluded).
    - For ``uniform(loc, scale)``, SciPy samples continuous values in
      ``[loc, loc + scale)``.
    - The returned dictionary is designed to be merged into a larger search space
      for composite estimators (e.g., pipelines).

    Examples
    --------
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> space = _tree_common_param_space(prefix="classifier__")
    >>> # search = RandomizedSearchCV(pipe, param_distributions=space, n_iter=30, cv=5)
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
    Build a shared hyperparameter search space for boosting core parameters.

    This helper returns a ``param_distributions`` dictionary for the two
    fundamental hyperparameters used by most boosting estimators: the number of
    boosting stages (``n_estimators``) and the shrinkage factor
    (``learning_rate``). Keys are prefixed (typically with a pipeline step name
    such as ``'classifier__'``) so the result can be used directly in
    scikit-learn model-selection utilities.

    Parameters
    ----------
    prefix : str, default 'classifier__'
        Prefix prepended to every parameter name (e.g., pipeline step name plus ``'__'``).
    n_estimators_min : int, default 100
        Minimum value for ``n_estimators`` (inclusive).
    n_estimators_max : int, default 1000
        Maximum value for ``n_estimators`` (exclusive).
    learning_rate_min : float, default 1e-3
        Lower bound for ``learning_rate`` (strictly positive).
    learning_rate_max : float, default 1.0
        Upper bound for ``learning_rate`` (must be greater than ``learning_rate_min``).

    Returns
    -------
    param_distributions : dict[str, Any]
        Mapping from prefixed parameter names to SciPy distributions:
        - ``<prefix>n_estimators`` : ``scipy.stats.randint``
        - ``<prefix>learning_rate`` : ``scipy.stats.loguniform``

    Notes
    -----
    - ``randint(a, b)`` samples integers in ``[a, b)`` (upper bound excluded).
    - ``loguniform(a, b)`` samples positive values in ``[a, b)`` on a log scale.
    - Smaller ``learning_rate`` typically requires larger ``n_estimators`` for
      comparable training error, but can improve regularization.

    Examples
    --------
    >>> space = _boosting_core_param_space(prefix="classifier__")
    >>> sorted(space.keys())
    ['classifier__learning_rate', 'classifier__n_estimators']
    """

    return {
        f"{prefix}n_estimators": randint(n_estimators_min, n_estimators_max),
        f"{prefix}learning_rate": loguniform(learning_rate_min, learning_rate_max),
    }


def get_static_model_and_search_space(
    model_name: str,
    random_state: int | None = None,
    use_cost_sensitive_learning: bool = True,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Instantiate a static classifier and its estimator-level hyperparameter search space.

    This factory returns (1) an unfitted estimator and (2) an estimator-only
    ``param_distributions`` dictionary suitable for CV-based hyperparameter search
    (e.g., ``RandomizedSearchCV`` / ``HalvingRandomSearchCV``). Returned parameter
    names are prefixed for a pipeline step named ``"classifier"`` (e.g.,
    ``classifier__C``).

    The returned search space intentionally excludes pipeline-level parameters
    (e.g., ``feature_selection_filter__k``); those must be added by the orchestration
    layer that builds the full pipeline.

    Parameters
    ----------
    model_name : str
        Canonical model key identifying which estimator to build. Supported keys are:
        ``'SVC'``, ``'MLPClassifier'``, ``'KNeighborsClassifier'``,
        ``'DecisionTreeClassifier'``, ``'RandomForestClassifier'``,
        ``'ExtraTreesClassifier'``, ``'BalancedRandomForestClassifier'``,
        ``'BaggingDecisionTreeClassifier'``, ``'AdaBoostClassifier'``,
        ``'LogitBoostClassifier'``, ``'XGBClassifier'``, ``'RUSBoostClassifier'``.
    random_state : int or None, default None
        Random seed forwarded to estimators that support it. If ``None``, library
        defaults are used.
    use_cost_sensitive_learning : bool, default True
        If ``True``, configure applicable estimators for imbalanced learning using
        built-in cost sensitivity (e.g., ``class_weight='balanced'`` / internal
        sampling strategies, and optionally tuning ``scale_pos_weight`` for XGBoost).
        If ``False``, disable these mechanisms where possible (e.g., set
        ``class_weight=None``, remove ``classifier__scale_pos_weight`` from the search
        space, set imbalanced-learn ``sampling_strategy=None``).

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        Unfitted classifier instance configured according to ``model_name`` and
        ``use_cost_sensitive_learning``.
    param_dist : dict[str, Any]
        Estimator-level hyperparameter search space (SciPy distributions / categorical
        lists). Keys are prefixed with ``'classifier__'`` to match a pipeline step
        named ``"classifier"``.

    Raises
    ------
    ValueError
        If ``model_name`` is not a supported key.

    Notes
    -----
    - The returned ``param_dist`` is estimator-only by design; add pipeline-level
      parameters (e.g., feature selection ``k``) outside this function.
    - Integer-valued ranges typically use ``scipy.stats.randint(a, b)`` (support
      ``[a, b)``), while positive multiplicative ranges typically use
      ``scipy.stats.loguniform(low, high)``.
    - When ``use_cost_sensitive_learning=False``, this function mutates the selected
      configuration for the requested model (e.g., removing a key from its search
      space). If you reuse the returned configuration objects elsewhere, ensure you
      treat them as per-call outputs.

    Examples
    --------
    >>> clf, space = get_static_model_and_search_space(
    ...     "RandomForestClassifier",
    ...     random_state=42,
    ...     use_cost_sensitive_learning=True,
    ... )
    >>> sorted(space)[:3]
    ['classifier__ccp_alpha', 'classifier__max_depth', 'classifier__max_features']
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
    param_dist: Dict[str, Any] = dict(config["param_dist"])

    return model, param_dist


def get_static_ensemble_model_and_search_space(
    ensemble_type: str,
    model_pool: Sequence[str],
    random_state: int | None = None,
    use_cost_sensitive_learning: bool = True,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Instantiate a static ensemble (voting/stacking) and a merged nested search space.

    This factory builds an unfitted ensemble estimator from a list of base-model
    identifiers and returns a single merged ``param_distributions`` dictionary that
    targets each nested sub-estimator using scikit-learn's parameter routing
    (e.g., ``classifier__svc_0__C``).

    The function:
    1) calls :func:`get_static_model_and_search_space` for each entry in ``model_pool``,
    2) assigns a unique name to each estimator instance (duplicates allowed), and
    3) rewrites base search-space keys from ``classifier__<param>`` into
       ``classifier__<est_name>__<param>`` so the result can be used to tune the
       ensemble when it is placed under a pipeline step named ``"classifier"``.

    Parameters
    ----------
    ensemble_type : str
        Ensemble type to instantiate. Supported values are:
        - ``"VotingClassifier"`` (soft voting over probabilities),
        - ``"StackingClassifier"`` (stacked generalization with a logistic-regression
          meta-learner).
    model_pool : Sequence[str]
        Base-model identifiers to include (e.g., ``["SVC", "XGBClassifier"]``).
        Duplicates are allowed; each occurrence becomes a distinct estimator instance.
    random_state : int or None, default None
        Random seed forwarded to base estimators (via the base factory) and to the
        stacking meta-learner when applicable.
    use_cost_sensitive_learning : bool, default True
        Forwarded to base estimators (via the base factory). For stacking, when ``True``
        the meta-learner is configured with ``class_weight="balanced"``.

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        Unfitted ensemble estimator instance:
        - ``sklearn.ensemble.VotingClassifier`` when ``ensemble_type="VotingClassifier"``,
        - ``sklearn.ensemble.StackingClassifier`` when ``ensemble_type="StackingClassifier"``.
    param_dist : dict[str, Any]
        Merged nested hyperparameter search space for all base estimators. Keys target
        the sub-estimators inside the ensemble assuming the ensemble is mounted under a
        pipeline step named ``"classifier"``. Example rewrite:
        - input key: ``"classifier__C"``
        - output key: ``"classifier__svc_0__C"``

    Raises
    ------
    ValueError
        If ``model_pool`` is empty or if ``ensemble_type`` is not supported.

    Notes
    -----
    - Soft voting requires each base estimator to implement ``predict_proba``; ensure
      your base factory configures probabilistic outputs where needed (e.g.,
      ``SVC(probability=True)``).
    - This function sets ensemble ``n_jobs=1`` to avoid nested parallelism; prefer
      outer-level parallelism (e.g., CV search ``n_jobs`` or outer-fold parallelism).
    - Key rewriting assumes the base factory emits keys prefixed with ``"classifier__"``.
      If you change that prefix, update the rewriting logic accordingly.

    Examples
    --------
    >>> ens, space = get_static_ensemble_model_and_search_space(
    ...     ensemble_type="VotingClassifier",
    ...     model_pool=["SVC", "XGBClassifier"],
    ...     random_state=42,
    ... )
    >>> any(k.startswith("classifier__svc_0__") for k in space)
    True
    """

    if not model_pool:
        raise ValueError("model_pool list cannot be empty.")

    estimators = []
    ensemble_param_dist = {}

    # 1. Build Base Estimators and Merge Spaces
    for idx, model_name in enumerate(model_pool):
        # Retrieve the base model and its specific search space
        base_model, base_space = get_static_model_and_search_space(
            model_name,
            random_state=random_state,
            use_cost_sensitive_learning=use_cost_sensitive_learning,
        )

        # Create a unique name for this estimator instance (e.g., 'xgbclassifier_0')
        # This name is crucial for the scikit-learn parameter routing.
        est_name = f"{model_name.lower()}_{idx}"
        estimators.append((est_name, base_model))

        # Rewrite search space keys.
        # Original: "classifier__max_depth"
        # Target (inside Pipeline > Ensemble): "classifier__<est_name>__max_depth"
        for key, distribution in base_space.items():
            # Remove the standard prefix provided by the factory function
            # We assume the factory returns keys starting with "classifier__"
            clean_param = key.replace("classifier__", "")

            # Construct the new nested key
            new_key = f"classifier__{est_name}__{clean_param}"
            ensemble_param_dist[new_key] = distribution

    # 2. Construct the Ensemble
    if ensemble_type == "VotingClassifier":
        # Soft voting returns the class label as argmax of the sum of predicted probabilities.
        # This requires 'probability=True' in SVC (handled in base factory).
        model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=len(model_pool))

    elif ensemble_type == "StackingClassifier":
        # Define the meta-learner
        final_layer_args = {"random_state": random_state, "solver": "lbfgs", "max_iter": 1000}

        if use_cost_sensitive_learning:
            final_layer_args["class_weight"] = "balanced"

        final_estimator = LogisticRegression(**final_layer_args)

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=len(model_pool),
            # 'passthrough': False -> Train meta-model only on predictions of base models
            passthrough=False,
            cv=5,  # Internal CV for training the meta-model
        )
    else:
        raise ValueError(
            f"Unknown ensemble_type: {ensemble_type}. Use 'VotingClassifier' or 'StackingClassifier'."
        )

    return model, ensemble_param_dist


def get_des_model(
    model_name: str,
    random_state: int | None = None,
    use_cost_sensitive_learning: bool = True,
) -> Tuple[BaseEstimator, Dict[str, Any], BaseEstimator, Dict[str, Any]]:
    """
    Instantiate the pool (bagging) model and a DESlib estimator configuration for DES.

    This factory returns two coupled components required by your Dynamic Ensemble
    Selection (DES) workflow:

    1) **Pool model** (bagging) + its hyperparameter search space:
       The pool is intended to be tuned as the ``"classifier"`` step of your
       standard training pipeline (preprocessing → feature selection → resampling → pool).

    2) **DESlib model** (unfitted) + a dict of default kwargs:
       The DES model is returned unfitted and the kwargs are provided separately so
       callers can inject the tuned pool via ``pool_classifiers=...`` and apply
       remaining DES parameters via ``set_params(**des_kwargs)`` before fitting on DSEL.

    Parameters
    ----------
    model_name : str
        DES method identifier. Supported values are:
        ``{"APriori", "APosteriori", "LCA", "MLA", "OLA", "KNORAE", "KNORAU", "DESP",
        "DESKNN", "DESClustering", "KNOP", "DESKL", "Exponential", "Logarithmic",
        "RRC", "METADES"}``.
    random_state : int or None, default None
        Random seed forwarded to the pool factory
        (:func:`get_static_model_and_search_space("BaggingDecisionTreeClassifier")`).
        The DES estimator itself is instantiated without constructor kwargs in this
        function; if a DES method exposes randomness control, apply it later through
        ``des_kwargs`` (and/or explicit overrides).
    use_cost_sensitive_learning : bool, default True
        Whether to configure the pool for cost-sensitive learning on imbalanced data.
        This flag is forwarded to the pool factory.

    Returns
    -------
    pool_estimator : sklearn.base.BaseEstimator
        Unfitted bagging ensemble used as the pool of classifiers.
        Intended to be placed under a pipeline step named ``"classifier"`` during the
        pool-tuning stage.
    pool_param_dist : dict[str, Any]
        Hyperparameter search space for the pool, compatible with CV search over a
        pipeline where the pool sits in the ``"classifier"`` step (e.g.,
        ``classifier__n_estimators``, ``classifier__max_samples``,
        ``classifier__estimator__max_depth``).
    des_model : sklearn.base.BaseEstimator
        Unfitted DESlib estimator instance corresponding to ``model_name``.
    des_kwargs : dict[str, Any]
        Default keyword arguments for the DES model (e.g., ``k``, ``DFP``, ``IH_rate``,
        ``voting``, ``n_jobs``). This dictionary does **not** include ``pool_classifiers``;
        callers typically add it before fitting. If you mutate this dictionary, copy it
        first to avoid unintended cross-call side effects.

    Raises
    ------
    ValueError
        If ``model_name`` is not a supported DES identifier.

    Notes
    -----
    - This function returns an **estimator-only** pool search space. Pipeline-level
      search keys (e.g., ``feature_selection_filter__k``) must be added by the orchestration
      layer that constructs the full pipeline.
    - DESlib methods may require either the fitted bagging object or a list/array of base
      estimators as ``pool_classifiers``. If needed, pass ``fitted_pool.estimators_`` instead
      of the bagger instance.
    - The typical two-stage workflow is:
      (i) tune/fit the pool on TRAIN,
      (ii) transform DSEL using the fitted preprocessing and fit the DES model on DSEL
      with ``pool_classifiers`` injected.

    Examples
    --------
    >>> pool_est, pool_space, des, des_kwargs = get_des_model(
    ...     "KNORAE",
    ...     random_state=42,
    ...     use_cost_sensitive_learning=True,
    ... )
    >>> "classifier__n_estimators" in pool_space
    True
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
                "Kp": 8,
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
