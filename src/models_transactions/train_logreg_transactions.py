"""
train_logreg_transactions.py

Train a Logistic Regression model on the transactions dataset.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..preprocessing_transactions import get_transactions_dataset
from ..evaluation import evaluate_classifier


def main():
    X_train, X_test, y_train, y_test, preprocessor = get_transactions_dataset()

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)

    evaluate_classifier(
        clf,
        X_test,
        y_test,
        model_name="logreg",
        dataset_name="transactions",
    )


if __name__ == "__main__":
    main()
