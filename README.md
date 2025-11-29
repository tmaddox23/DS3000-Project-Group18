# 🛍️ DS3000 Counterfeit Detection Project  
**Team 18 — Western University, DS3000: Introduction to Machine Learning**  

This project builds machine-learning classifiers to detect counterfeit behavior in online marketplaces using **two complementary datasets**:

- A **transaction-level dataset** focused on buyer behavior  
- A **product/seller-level dataset** focused on listing metadata and seller trust signals  

Each dataset trains its own family of models, and the results are compared to understand different dimensions of counterfeit detection.

---


---

## 🔧 Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

If you use conda:
```bash
conda create -n ds3000 python=3.11
conda activate ds3000
pip install -r requirements.txt
```

## Preprocessing overview
preprocessing_overview:
  transactions:
    description: "Preprocessing for the transaction-level dataset (_counterfeit_transactions.csv)."
    steps:
      - "Load raw data from data/raw/"
      - "Drop ID columns: transaction_id, customer_id"
      - "Identify target column: involves_counterfeit"
      - "Select relevant numeric and categorical features"
      - "Apply StandardScaler to numeric features"
      - "Apply OneHotEncoder to categorical and boolean features"
      - "Split into train/test sets (80/20 stratified)"
      - "Return X_train, X_test, y_train, y_test, preprocessor"

  products:
    description: "Preprocessing for the product/seller-level dataset (counterfeit_products.csv)."
    steps:
      - "Load raw data from data/raw/"
      - "Drop ID columns: product_id, seller_id"
      - "Identify target column: is_counterfeit"
      - "Select product-level and seller-level numeric features"
      - "Select appropriate categorical and boolean features"
      - "Apply StandardScaler to numeric features"
      - "Apply OneHotEncoder to categorical features"
      - "Split into train/test sets (80/20 stratified)"
      - "Return X_train, X_test, y_train, y_test, preprocessor"

running_models:
  transaction_models:
    logistic_regression: "python src/models_transactions/train_logreg_transactions.py"
    random_forest: "python src/models_transactions/train_rf_transactions.py"
    gradient_boosting: "python src/models_transactions/train_gb_transactions.py"
    svm: "python src/models_transactions/train_svm_transactions.py"
    knn: "python src/models_transactions/train_knn_transactions.py"

  product_models:
    logistic_regression: "python src/models_products/train_logreg_products.py"
    random_forest: "python src/models_products/train_rf_products.py"
    gradient_boosting: "python src/models_products/train_gb_products.py"
    svm: "python src/models_products/train_svm_products.py"
    knn: "python src/models_products/train_knn_products.py"

outputs:
  saved_to: "results/"
  files:
    - "metrics_transactions.csv"
    - "metrics_products.csv"
  directories:
    - "results/confusion_matrices/"
    - "results/roc_curves/"
  printed_metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc (when available)"

#