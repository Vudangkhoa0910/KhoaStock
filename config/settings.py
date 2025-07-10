# Author: Vu Dang Khoa

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

DEFAULT_SYMBOLS = ['VCB', 'VNM', 'FPT']

MARKET_HOURS = {
    'morning_start': (9, 15),
    'morning_end': (11, 30),
    'afternoon_start': (13, 0),
    'afternoon_end': (14, 45)
}

MODEL_CONFIG = {
    'prediction_horizons': [1, 2, 3],
    'train_test_split': 0.8,
    'validation_split': 0.2,
    'random_seed': 42
}

API_CONFIG = {
    'rate_limit_per_minute': 15,
    'retry_attempts': 3,
    'timeout_seconds': 30
}
