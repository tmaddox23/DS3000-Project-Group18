"""
train_logreg_transactions.py

Train a Logistic Regression model on the transaction-level dataset
(_counterfeit_transactions.csv) and evaluate its performance.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Helper that returns X_train, X_test, y_train, y_test, and the preprocessing pipeline
from ..preprocessing_transactions import get_transactions_dataset
# Shared evaluation helper that prints metrics and saves plots
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for the transactions dataset
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define the Logistic Regression classifier (linear model for binary classification)
    logreg = LogisticRegression(
        max_iter=1000,   # allow more iterations so the optimizer can converge
        random_state=42, # make results reproducible
        n_jobs=-1,       # use all available CPU cores (for solvers that support it)
        class_weight="balanced",
    
    #For LogReg and SVM, you now see:
    #Recall = 1.0 → they catch all counterfeit transactions in the test set
    #Precision dipped a bit (~0.967) → they flag a few more false positives
    #That’s exactly what “balanced” does: it makes the model care more about the minority class, pushing for higher recall.
    )

    # Build a Pipeline:
    # 1) "preprocess" step: applies scaling + one-hot encoding to features
    # 2) "model" step: fits Logistic Regression on the transformed features
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", logreg),
    ])

    # Train the pipeline on the training data
    clf.fit(X_train, y_train)

    # Evaluate the trained model on the test data
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="logreg",       # used for printing and file naming
        dataset_name="transactions",
    )


# This block runs only when the script is executed directly,
# e.g. python -m src.models_transactions.train_logreg_transactions
if __name__ == "__main__":
    main()
