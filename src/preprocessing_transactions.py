"""
Preprocessing pipeline for the transactions dataset.
- Selects features + target (involves_counterfeit)
- Encodes categoricals/booleans
- Scales numeric features
- Returns train/test splits and the preprocessing pipeline
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TEST_SIZE, RANDOM_STATE
from .data_loading import load_transactions


def get_transactions_features_and_target(df: pd.DataFrame):
    """
    Split transactions dataframe into features X and target y.

    df : raw transactions DataFrame

    Returns
    -------
    X : DataFrame of features
    y : Series of target labels (0/1)
    """
    target_col = "involves_counterfeit"  # binary target

    # ID columns (not used as features)
    id_cols = ["transaction_id", "customer_id"]

    # Feature columns (all non-ID, non-target)
    feature_cols = [
        "transaction_date",
        "customer_age",
        "customer_location",
        "quantity",
        "unit_price",
        "total_amount",
        "payment_method",
        "shipping_speed",
        "customer_history_orders",
        "discount_applied",
        "discount_percentage",
        "shipping_cost",
        "delivery_time_days",
        "refund_requested",
        "velocity_flag",
        "geolocation_mismatch",
        "device_fingerprint_new",
    ]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def build_transactions_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - One-hot encodes categorical/boolean features
    - Scales numeric features
    """

    # Numeric columns to be standardized
    numeric_cols = [
        "customer_age",
        "quantity",
        "unit_price",
        "total_amount",
        "customer_history_orders",
        "discount_percentage",
        "shipping_cost",
        "delivery_time_days",
    ]

    # Categorical/boolean columns to be one-hot encoded
    categorical_cols = [
        "transaction_date",
        "customer_location",
        "payment_method",
        "shipping_speed",
        "discount_applied",
        "refund_requested",
        "velocity_flag",
        "geolocation_mismatch",
        "device_fingerprint_new",
    ]

    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    return preprocessor


def get_transactions_dataset(
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Full helper for the transactions dataset:
    - loads raw data
    - builds X, y
    - splits into train/test
    - builds preprocessing pipeline

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_transactions()
    X, y = get_transactions_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_transactions_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
