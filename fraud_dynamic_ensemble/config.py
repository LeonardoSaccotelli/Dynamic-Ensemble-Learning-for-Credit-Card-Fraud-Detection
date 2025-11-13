from datetime import datetime
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

RAW_FILENAME = "credit_card_fraud_sampling.csv"
INTERIM_FILENAME = "credit_card_fraud_cleaned.csv"
PROCESSED_FILENAME = "credit_card_fraud_features.csv"
EXTERNAL_FILENAME = "creditcardfraud.csv"

# SEED FOR REPRODUCIBILITY
RANDOM_STATE = 42

# EXPERIMENT TRACKING
RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# DATASET_SAMPLING.PY PARAMETERS
POLICY = "keep_all_minority"
N_ROWS = None
FRAC = None
RATIO = 50

# FEATURES.PY PARAMETERS
DAY_SECONDS = 86_400.0  # 24 * 60 * 60

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
