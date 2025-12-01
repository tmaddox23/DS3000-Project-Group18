# Helper functions to load raw datasets.

import pandas as pd
from .config import TRANSACTIONS_RAW_PATH, PRODUCTS_RAW_PATH


def load_transactions():
    df = pd.read_csv(TRANSACTIONS_RAW_PATH)
    return df


def load_products():
    df = pd.read_csv(PRODUCTS_RAW_PATH)
    return df
