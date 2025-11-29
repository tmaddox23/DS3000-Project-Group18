"""
train_rf_products.py

Train a Random Forest model on the product/seller-level dataset
(counterfeit_products.csv) and evaluate its performance.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_products import get_products_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for products
    X_train, X_test, y_train, y_test, preprocessor = get_products_dataset()

    # Define the Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=200,  # number of trees
        max_depth=None,    # allow trees to grow fully
        random_state=42,   # reproducible results
        n_jobs=-1,         # parallelize across CPU cores
        class_weight="balanced",
    )

    # Pipeline: preprocessing -> RF model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="rf",
        dataset_name="products",
    )


if __name__ == "__main__":
    main()
