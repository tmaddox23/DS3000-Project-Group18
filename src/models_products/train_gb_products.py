"""
train_gb_products.py

Train a Gradient Boosting model on the product/seller-level dataset
(counterfeit_products.csv) and evaluate its performance.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_products import get_products_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for products
    X_train, X_test, y_train, y_test, preprocessor = get_products_dataset()

    # Define the Gradient Boosting classifier
    gb = GradientBoostingClassifier(
        random_state=42,  # reproducible
    )

    # Pipeline: preprocessing -> GB model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", gb),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="gb",
        dataset_name="products",
    )


if __name__ == "__main__":
    main()
