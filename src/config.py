from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from datetime import datetime

# Load environment variables from .env file if it exists
load_dotenv()

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

# PIPELINE AND FEATURES SELECTION
NUMERICAL_FEATURES_TO_NORMALIZE = ["Log_Amount"]
K_BEST_TO_KEEP = 20

# EXPERIMENT AND EVALUATION SETTINGS
CV_N_SPLITS = 10
CV_N_REPEATS = 10
DSEL_SIZE = 0.15

# HYPERPARAMETER TUNING SETTINGS FOR RANDOMIZED_SEARCH_CV
N_ITER_TUNING = 30
CV_TUNING = 5
SCORING_TUNING = "f1"
N_JOBS_TUNING = -1

# MODELS TO TRAIN
BASE_MODELS = ["DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
               "BalancedRandomForestClassifier", "RUSBoostClassifier",
               "XGBClassifier", "AdaBoostClassifier", "LogitBoostClassifier",
               "MLPClassifier", "SVC"]

STATIC_ENS_MODELS = ["VotingClassifier", "VotingClassifier_weighted"]

DES_MODELS = ["APosteriori", "APriori", "LCA", "MLA", "OLA",
              "DESClustering", "DESP", "DESKNN",]
              #"KNOP", "KNORAE", "KNORAU", "METADES",
              #"RRC","DESKL", "Exponential", "Logarithmic",
              #"StackedClassifier"]

POOL_MODELS = ["RandomForestClassifier", "SVC", "BalancedRandomForestClassifier",
               "RUSBoostClassifier", "XGBClassifier", "AdaBoostClassifier"]

RESAMPLING_METHOD = None

# GENERAL SETTINGS
RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RESAMPLING_METHOD}"
RANDOM_STATE = 42

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
