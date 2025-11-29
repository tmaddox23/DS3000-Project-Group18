"""
train_svm_transactions.py

Train an SVM classifier on the transaction-level dataset
(_counterfeit_transactions.csv) and evaluate its performance.
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from ..preprocessing_transactions import get_transactions_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for transactions
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    # Define an SVM classifier with an RBF kernel
    svm = SVC(
        kernel="rbf",       # non-linear kernel (Radial Basis Function)
        probability=True,   # enable probability estimates for ROC-AUC
        random_state=42,    # reproducibility
        class_weight="balanced",
    )

    # Pipeline: preprocessing -> SVM model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", svm),
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="svm",
        dataset_name="transactions",
    )


if __name__ == "__main__":
    main()
