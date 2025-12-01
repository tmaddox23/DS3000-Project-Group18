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

# Raw CSV files (input datasets)
TRANSACTIONS_RAW_PATH = DATA_DIR / "counterfeit_transactions.csv"
PRODUCTS_RAW_PATH = DATA_DIR / "counterfeit_products.csv"


# Common ML constants
RANDOM_STATE = 42 # fixed seed for reproducibility (splits + models)
TEST_SIZE = 0.2 # 20% of data used as test set
