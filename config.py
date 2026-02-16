"""
Configuration file for demand forecasting project
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / 'train.csv'
TEST_FILE = DATA_DIR / 'test.csv'
STORE_FILE = DATA_DIR / 'store.csv'

# Forecast parameters
FORECAST_WEEKS = 6  # Predict 6 weeks ahead
TRAIN_TEST_SPLIT_DATE = None  # Will be determined from data

# Model hyperparameters
RANDOM_STATE = 42

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# Prophet parameters
PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'seasonality_mode': 'multiplicative'
}

# Feature engineering
LAG_FEATURES = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]  # Days to lag
ROLLING_WINDOWS = [7, 14, 28]  # Rolling window sizes

# Visualization settings
FIGSIZE = (15, 6)
STYLE = 'seaborn-v0_8-darkgrid'
