"""
train_logreg_products.py

Train a Logistic Regression model on the product/seller-level dataset
(counterfeit_products.csv) and evaluate its performance.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..preprocessing_products import get_products_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for products
    X_train, X_test, y_train, y_test, preprocessor = get_products_dataset()

    # Define Logistic Regression classifier (linear model for classification)
    logreg = LogisticRegression(
        max_iter=1000,   # increase iterations to ensure convergence
        random_state=42, # reproducible
        n_jobs=-1,       # use all cores (for some solvers)
        class_weight="balanced",
    )

    # Pipeline: preprocessing -> logistic regression
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", logreg),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on test data
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="logreg",
        dataset_name="products",
    )


if __name__ == "__main__":
    main()
