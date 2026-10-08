import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

from xgboost import XGBClassifier


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "corporate_tax_risk.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "corporate_tax_risk_xgboost.joblib"
)


os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# 3. REMOVE COMPANY ID
# =========================================================

X = df.drop(
    columns=["Tax_Risk_Score", "Company_ID"]
)

y = df["Tax_Risk_Score"]


# =========================================================
# 4. ENCODE TARGET
# =========================================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = y.map(target_mapping)


# =========================================================
# 5. FEATURE TYPES
# =========================================================

categorical_features = [
    "Ownership_Type",
    "Market_Region"
]

numerical_features = [
    "Internal_Control_Score",
    "Effective_Tax_Rate",
    "Audit_Likelihood",
    "Offshore_Transactions",
    "Previous_Fines",
    "Board_Independence",
    "Revenue",
    "Profit_Before_Tax",
    "Leverage_Ratio",
    "Governance_Score",
    "Restatement_History",
    "Whistleblower_Reports"
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 7. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# 8. XGBOOST MODEL
# =========================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,

    subsample=0.9,
    colsample_bytree=0.9,

    random_state=42,

    eval_metric="mlogloss"
)


# =========================================================
# 9. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# 10. CLASS BALANCING
# =========================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# =========================================================
# 11. TRAIN
# =========================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)

print("Training completed.")


# =========================================================
# 12. EVALUATION
# =========================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low",
            "Medium",
            "High"
        ]
    )
)


# =========================================================
# 13. SAVE MODEL
# =========================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)