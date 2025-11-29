"""
train_svm_products.py

Train an SVM classifier on the product/seller-level dataset
(counterfeit_products.csv) and evaluate its performance.
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from ..preprocessing_products import get_products_dataset
from ..evaluation import evaluate_classifier


def main():
    # Get train/test splits and preprocessing pipeline for products
    X_train, X_test, y_train, y_test, preprocessor = get_products_dataset()

    # Define SVM with RBF kernel
    svm = SVC(
        kernel="rbf",       # non-linear classifier
        probability=True,   # enable predict_proba for ROC-AUC
        random_state=42,    # reproducible
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
        dataset_name="products",
    )


if __name__ == "__main__":
    main()
