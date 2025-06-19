from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

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
NUMERICAL_FEATURES_TO_NORMALIZE = ["LogAmount"]
K_BEST_TO_KEEP = 20

# EXPERIMENT AND EVALUATION SETTINGS
RANDOM_STATE = 42
CV_N_SPLITS = 10
CV_N_REPEATS = 10
DSEL_SIZE = 0.2

# MODELS TO TRAIN
BASE_MODELS = ["RandomForestClassifier", "SVC"]
DES_MODELS = ["OLA", "KNORA"]


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
