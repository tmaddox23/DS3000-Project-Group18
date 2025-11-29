"""
train_rf_transactions.py

Train a Random Forest model on the transaction-level dataset
(_counterfeit_transactions.csv) and evaluate its performance.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..preprocessing_transactions import get_transactions_dataset
from ..evaluation import evaluate_classifier


def main():
    # Load train/test splits + preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define the Random Forest model
    rf = RandomForestClassifier(
        n_estimators=200,      # number of trees
        max_depth=None,       # can tune later
        random_state=42,
        n_jobs=-1,            # use all CPU cores
    )

    # Full pipeline: preprocessing -> model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate and save metrics/plots
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="rf",
        dataset_name="transactions",
    )


if __name__ == "__main__":
    main()
