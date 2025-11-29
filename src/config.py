"""
config.py

Central configuration for paths and common constants used across the project.
"""

from pathlib import Path

# Base project directory (assumes this file is in src/)
# __file__ = this file, .resolve() = absolute path, .parent = src/, .parent.parent = repo root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw CSV files (input datasets)
TRANSACTIONS_RAW_PATH = RAW_DATA_DIR / "_counterfeit_transactions.csv"
PRODUCTS_RAW_PATH = RAW_DATA_DIR / "counterfeit_products.csv"


# Optional: processed versions if you ever decide to save cleaned data
TRANSACTIONS_PROCESSED_PATH = PROCESSED_DATA_DIR / "transactions_processed.csv"
PRODUCTS_PROCESSED_PATH = PROCESSED_DATA_DIR / "products_processed.csv"

# Results paths
RESULTS_DIR = BASE_DIR / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
ROC_DIR = RESULTS_DIR / "roc_curves"

# Common ML constants
RANDOM_STATE = 42 # fixed seed for reproducibility (splits + models)
TEST_SIZE = 0.2 # 20% of data used as test set
