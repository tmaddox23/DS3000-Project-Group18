"""
data_loading.py

Helper functions to load raw datasets.
All model scripts should use these instead of reading CSV paths manually.
"""

import pandas as pd
from .config import TRANSACTIONS_RAW_PATH, PRODUCTS_RAW_PATH


def load_transactions():
    """Load the raw transactions dataset as a pandas DataFrame."""
    df = pd.read_csv(TRANSACTIONS_RAW_PATH)
    # Optional: basic sanity checks here
    return df


def load_products():
    """Load the raw products/sellers dataset as a pandas DataFrame."""
    df = pd.read_csv(PRODUCTS_RAW_PATH)
    # Optional: basic sanity checks here
    return df
