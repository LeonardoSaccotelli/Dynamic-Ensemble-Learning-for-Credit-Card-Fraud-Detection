from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

#################################################################
# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILENAME = "credit_card_fraud_sampling.csv"
INTERIM_FILENAME = "credit_card_fraud_cleaned.csv"
PROCESSED_FILENAME = "credit_card_fraud_features.csv"
EXTERNAL_FILENAME = "creditcardfraud.csv"

#################################################################
# seed for reproducibility
RANDOM_STATE = 42

#################################################################
# dataset_sampling.py parameters
POLICY = "keep_all_minority"
N_ROWS = None
FRAC = None
RATIO = 50

#################################################################
# features.py parameters
DAY_SECONDS = 86_400.0  # 24 * 60 * 60

#################################################################
# train.py parameters

# feature transformation
NUMERICAL_FEATURES_TO_STANDARDIZE = ["Amount_log1p"]

# feature selection
FS_K_BEST_TO_KEEP: int | str = 20

# Candidate values used by RandomizedSearchCV to tune SelectKBest.
# NOTE: ensure each int <= n_features AFTER preprocessing. "all" is allowed.
FS_K_BEST_CANDIDATES: list[int | str] = [20, 22, 24, 26, 28, "all"]

# resampling method
USE_COST_SENSITIVE_LEARNING = True
RESAMPLING_METHOD = None
RESAMPLING_PARAMS = {}

# outer evaluation setting: RepeatedStratifiedKFold (10 x 10)
CV_OUTER_N_SPLITS = 10
CV_OUTER_N_REPEATS = 10
CV_OUTER_PARALLEL_N_JOBS = 1

# inner evaluation setting for dynamical ensemble models
DSEL_SIZE = 0.15

# inner evaluation setting for hyperparameters tuning: RandomizedSearchCV
TUNING_N_ITER = 30
TUNING_CV_INNER_N_SPLITS = 5
TUNING_SCORING = "f1"
TUNING_N_JOBS = -1

# classic ML model + static ensemble models
STATIC_MODELS = [
    "SVC",
    "MLPClassifier",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
]

DES_MODELS = [
    "LCA",
    "DESClustering",
    "DESP",
    "KNORAU",
    "METADES"
]

# experiment tracking
EXPERIMENT_NAME = f"CostSensitiveLearning___RandomizedSearchCV__niter_{TUNING_N_ITER}__cv_{TUNING_CV_INNER_N_SPLITS}"
EXPERIMENT_DESCRIPTION = ""

#################################################################
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
