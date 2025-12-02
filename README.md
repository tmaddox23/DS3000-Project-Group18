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
---

## How to run

Navigate to the notebooks folder and run the .ipynbs

---

## Preprocessing Overview

This document summarizes the preprocessing logic for both datasets used in the counterfeit detection pipeline:  
- **Transaction-level dataset** (`_counterfeit_transactions.csv`)  
- **Product/Seller-level dataset** (`counterfeit_products.csv`)

Each preprocessing module loads raw data, selects safe/meaningful features, encodes/standardizes them, and returns train/test splits along with an sklearn `ColumnTransformer` preprocessor.

---

### 🧾 Transactions Preprocessing

**File:** `preprocessing_transactions.py`  
**Description:** Preprocessing for the transaction-level dataset (`_counterfeit_transactions.csv`).

#### Steps

1. **Load raw data** from `data/raw/`.
2. **Drop ID columns:**  
   `transaction_id`, `customer_id`
3. **Identify target column:**  
   `involves_counterfeit`
4. **Select relevant features:**  
   - Numeric features (e.g., `unit_price`, `customer_age`, `shipping_cost`, etc.)  
   - Categorical + boolean features (e.g., `payment_method`, `shipping_speed`, `velocity_flag`, etc.)
5. **Apply `StandardScaler`** to numeric features.
6. **Apply `OneHotEncoder`** to categorical & boolean features.
7. **Split into train/test sets** (80/20), stratified by `involves_counterfeit`.
8. **Return:**  
   `X_train`, `X_test`, `y_train`, `y_test`, `preprocessor`.

---

### 📦 Products Preprocessing

**File:** `preprocessing_products.py`  
**Description:** Preprocessing for the product/seller-level dataset (`counterfeit_products.csv`).

#### Steps

1. **Load raw data** from `data/raw/`.
2. **Drop ID columns:**  
   `product_id`, `seller_id`
3. **Identify target column:**  
   `is_counterfeit`
4. **Select numeric features:**  
   Product- and seller-level numerical information (e.g., price, views, warranty_months).
5. **Select categorical & boolean features:**  
   Category, brand, country fields, and fraud-related boolean flags.
6. **Convert date field** (`listing_date`) → derived numeric: `listing_days_since`.
7. **Apply `StandardScaler`** to numeric + boolean features.
8. **Apply `OneHotEncoder`** to categorical features.
9. **Split into train/test sets** (80/20), stratified by `is_counterfeit`.
10. **Return:**  
    `X_train`, `X_test`, `y_train`, `y_test`, `preprocessor`.
