from pathlib import Path
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_STORE = Path(os.getenv("FEATURE_STORE", DATA_DIR / "feature_store"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))

# Create directories if they don't exist
for directory in [FEATURE_STORE, RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE = LOGS_DIR / "quant.log"

# Model parameters
DEFAULT_LR = float(os.getenv("DEFAULT_LR", "0.05"))
DEFAULT_MAX_DEPTH = int(os.getenv("DEFAULT_MAX_DEPTH", "5"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "10"))

# Feature columns used in strategies
FACTOR_COLS = [
    'price_deviation_20',
    'price_position_20',
    'RSI_low_dist_14',
    'bb_position',
    'macd_hist',
    'vol_20d',
    'momentum_3',
    'momentum_5',
    'momentum_10',
    'reversal_1',
    'reversal_3',
    'reversal_weighted',
    'reversal_rsi',
    'vol_ratio',
    'RSI_14',
    'RSI_7',
    'RSI_3',
    'BB_upper',
    'BB_lower',
    'OBV',
    'MFI',
    'VWAP_diff_10',
    'vol_std_5',
    'vol_std_10',
    'MACD_default_line',
    'MACD_default_signal',
    'MACD_default_hist',
    'MACD_default_cross',
    'MACD_short_line',
    'MACD_short_signal',
    'MACD_short_hist',
    'MACD_short_cross',
]

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
) 