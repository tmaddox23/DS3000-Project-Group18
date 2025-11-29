"""
train_gb_transactions.py

Train a Gradient Boosting model on the transaction-level dataset
(_counterfeit_transactions.csv) and evaluate its performance.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_transactions import get_transactions_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for transactions
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define the Gradient Boosting classifier
    # (sequentially builds an ensemble of small decision trees)
    gb = GradientBoostingClassifier(
        random_state=42,  # for reproducibility
    )

    # Pipeline: preprocessing -> gradient boosting model
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
        dataset_name="transactions",
    )


if __name__ == "__main__":
    main()
