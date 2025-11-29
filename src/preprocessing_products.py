"""
preprocessing_products.py

Preprocessing pipeline for the products/sellers dataset.
- Selects features + target (is_counterfeit)
- Encodes categoricals with OneHotEncoder
- Scales numeric features with StandardScaler
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

    df : raw products DataFrame

    Returns
    -------
    X : DataFrame of features
    y : Series of target labels (0/1)
    """
    target_col = "is_counterfeit"  # binary target column

    # ID columns that uniquely identify rows but should not be used as features
    id_cols = ["product_id", "seller_id"]  # kept here just for clarity (not used below)

    # Feature columns used for modeling (all non-ID, non-target columns)
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

    # X = features, y = label (cast to int to ensure 0/1)
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def build_products_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build ColumnTransformer for the products dataset.

    - numeric_cols -> StandardScaler
    - categorical_cols -> OneHotEncoder
    """

    # Hand-picked numeric columns (will be standardized)
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

    # Hand-picked categorical/boolean columns (will be one-hot encoded)
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

    # Define transformers
    cat_transformer = OneHotEncoder(handle_unknown="ignore")  # ignore unseen categories
    num_transformer = StandardScaler()                        # mean=0, std=1

    # ColumnTransformer applies different transformers to different column subsets
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
    Full helper for the products dataset:
    - loads raw data
    - builds X, y
    - splits into train/test
    - builds preprocessing pipeline

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_products()  # load raw CSV via helper
    X, y = get_products_features_and_target(df)

    # Stratified split keeps class balance similar in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build preprocessor using training data columns
    preprocessor = build_products_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
