"""
train_knn_products.py

Train a K-Nearest Neighbors classifier on the product/seller-level dataset
(counterfeit_products.csv) and evaluate its performance.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_products import get_products_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for products
    X_train, X_test, y_train, y_test, preprocessor = get_products_dataset()

    # Define the KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=5,      # number of neighbors to consider
        weights="distance", # closer neighbors have more influence
        n_jobs=-1,          # parallelize distance computations
    )

    # Pipeline: preprocessing -> KNN model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", knn),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="knn",
        dataset_name="products",
    )


if __name__ == "__main__":
    main()
