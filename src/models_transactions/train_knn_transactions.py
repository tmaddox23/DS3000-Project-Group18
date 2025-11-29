"""
train_knn_transactions.py

Train a K-Nearest Neighbors classifier on the transaction-level dataset
(_counterfeit_transactions.csv) and evaluate its performance.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_transactions import get_transactions_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for transactions
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define the KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=5,     # look at 5 nearest neighbors
        weights="distance",# closer neighbors have more influence
        n_jobs=-1,         # use all CPU cores (for distance computations)
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
        dataset_name="transactions",
    )


if __name__ == "__main__":
    main()
