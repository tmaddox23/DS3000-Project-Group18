"""
preprocessing_products.py

Preprocessing pipeline for the products/sellers dataset.
- Selects features + target (is_counterfeit)
- Encodes categoricals
- Scales numeric features if needed
- Returns train/test splits and the preprocessing pipeline
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TEST_SIZE, RANDOM_STATE
from .data_loading import load_products


def get_products_features_and_target(df: pd.DataFrame):
    """
    Split products dataframe into features X and target y.
    Uses all non-ID, non-target columns as features.
    """
    target_col = "is_counterfeit"

    id_cols = ["product_id", "seller_id"]

    feature_cols = [
        "category",
        "brand",
        "price",
        "seller_rating",
        "seller_reviews",
        "product_images",
        "description_length",
        "shipping_time_days",
        "spelling_errors",
        "domain_age_days",
        "contact_info_complete",
        "return_policy_clear",
        "payment_methods_count",
        "listing_date",
        "seller_country",
        "shipping_origin",
        "views",
        "purchases",
        "wishlist_adds",
        "certification_badges",
        "warranty_months",
        "bulk_orders",
        "unusual_payment_patterns",
        "ip_location_mismatch",
    ]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def build_products_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build ColumnTransformer for the products dataset."""

    numeric_cols = [
        "price",
        "seller_rating",
        "seller_reviews",
        "description_length",
        "shipping_time_days",
        "spelling_errors",
        "domain_age_days",
        "payment_methods_count",
        "views",
        "purchases",
        "wishlist_adds",
        "warranty_months",
        "bulk_orders",
    ]

    categorical_cols = [
        "category",
        "brand",
        "product_images",
        "contact_info_complete",
        "return_policy_clear",
        "listing_date",
        "seller_country",
        "shipping_origin",
        "certification_badges",
        "unusual_payment_patterns",
        "ip_location_mismatch",
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


def get_products_dataset(
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Full helper:
    - loads data
    - splits into X/y
    - train/test split
    - builds preprocessor
    """
    df = load_products()
    X, y = get_products_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_products_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
