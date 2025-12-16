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
# experiment tracking
EXPERIMENT_NAME = "CostSensitiveLearning_KBest25_RandomizedSearchCV30Iter_InnerCV5Folds"
EXPERIMENT_DESCRIPTION = ""

# feature transformation
NUMERICAL_FEATURES_TO_STANDARDIZE = ["Amount_log1p"]

# feature selection
FS_K_BEST_TO_KEEP = 25

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

DES_MODELS = ["LCA", "DESClustering", "DESP", "KNORAU", "METADES"]

#################################################################
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
