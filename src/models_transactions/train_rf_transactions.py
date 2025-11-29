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
    # Get train/test splits and preprocessing pipeline for transactions
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define the Random Forest model (ensemble of decision trees)
    rf = RandomForestClassifier(
        n_estimators=200,   # number of trees in the forest
        max_depth=None,     # trees can grow until leaves are pure or min_samples reached
        random_state=42,    # reproducible results
        n_jobs=-1,          # use all CPU cores
        class_weight="balanced",
    )

    # Build a Pipeline: first apply preprocessing, then fit the RF model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    # Fit the pipeline on training data (preprocessing is learned + model is trained)
    clf.fit(X_train, y_train)

    # Evaluate on test data and print/save metrics + plots
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="rf",
        dataset_name="transactions",
    )


if __name__ == "__main__":
    # This block runs when you call:
    # python -m src.models_transactions.train_rf_transactions
    main()
