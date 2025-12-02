"""
Leakage-free preprocessing pipeline for products dataset.
- Removes leaking columns
- Converts booleans
- Handles date field
- Encodes categoricals
- Scales numeric features
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TEST_SIZE, RANDOM_STATE
from .data_loading import load_products

NUMERIC_COLS = [
    "price",
    "views",
    "purchases",
    "wishlist_adds",
    "certification_badges",
    "warranty_months",
    "listing_days_since",
]

BOOL_COLS = [
    "contact_info_complete",
    "return_policy_clear",
    "bulk_orders",
    "unusual_payment_patterns",
    "ip_location_mismatch",
]

CATEGORICAL_COLS = [
    "category",
    "brand",
    "seller_country",
    "shipping_origin",
]

def get_products_features_and_target(df: pd.DataFrame):

    # Convert listing_date → numeric
    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        min_date = df["listing_date"].min()
        df["listing_days_since"] = (df["listing_date"] - min_date).dt.days.fillna(0)

    # Ensure boolean columns are 0/1
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    target_col = "is_counterfeit"

    feature_cols = NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS

    df_clean = df.dropna(subset=feature_cols + [target_col])

    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    return X, y

def build_products_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + BOOL_COLS),   # scale numeric + boolean
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    return preprocessor

def get_products_dataset(
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:

    df = load_products()
    X, y = get_products_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_products_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
