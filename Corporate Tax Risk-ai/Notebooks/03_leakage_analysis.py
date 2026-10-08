import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from xgboost import XGBClassifier


# =========================================================
# 1. PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "corporate_tax_risk.csv"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LEAKAGE-AWARE CORPORATE TAX RISK MODEL")
print("=" * 60)

print("Dataset shape:", df.shape)


# =========================================================
# 3. FEATURES TO EXCLUDE
# =========================================================

leakage_features = [
    "Previous_Fines",
    "Internal_Control_Score",
    "Effective_Tax_Rate",
    "Restatement_History"
]

print("\nExcluded features:")
for feature in leakage_features:
    print("-", feature)


# =========================================================
# 4. CREATE X AND Y
# =========================================================

X = df.drop(
    columns=[
        "Tax_Risk_Score",
        "Company_ID"
    ] + leakage_features
)

y = df["Tax_Risk_Score"]


# =========================================================
# 5. TARGET ENCODING
# =========================================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = y.map(target_mapping)


# =========================================================
# 6. FEATURE TYPES
# =========================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nRemaining numerical features:")
print(numerical_features)

print("\nRemaining categorical features:")
print(categorical_features)


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================================================
# 8. PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================================================
# 9. CLASS WEIGHTS
# =========================================================

class_counts = np.bincount(y_train)

total_samples = len(y_train)

number_of_classes = len(class_counts)

class_weights = {
    class_id:
        total_samples /
        (
            number_of_classes *
            count
        )

    for class_id, count
    in enumerate(class_counts)
}


sample_weights = np.array([
    class_weights[class_id]
    for class_id in y_train
])


print("\nClass counts:")
print(class_counts)

print("\nClass weights:")
print(class_weights)


# =========================================================
# 10. PREPROCESS DATA
# =========================================================

print("\nPreparing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# =========================================================
# 11. XGBOOST
# =========================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    reg_alpha=0.1,

    reg_lambda=1,

    random_state=42,

    eval_metric="mlogloss",

    tree_method="hist"
)


# =========================================================
# 12. TRAIN
# =========================================================

print("\nTraining leakage-aware XGBoost...")

model.fit(
    X_train_processed,
    y_train,
    sample_weight=sample_weights
)


# =========================================================
# 13. PREDICT
# =========================================================

predictions = model.predict(
    X_test_processed
)


# =========================================================
# 14. EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)


print("\n")
print("=" * 60)
print("LEAKAGE-AWARE RESULTS")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy, 4)
)

print(
    "Macro F1:",
    round(macro_f1, 4)
)


# =========================================================
# 15. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Low",
            "Medium",
            "High"
        ],
        zero_division=0
    )
)


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n")
print("=" * 60)
print("TOP FEATURES")
print("=" * 60)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 18. SAVE RESULTS
# =========================================================

output_path = (
    BASE_DIR
    / "models"
    / "leakage_aware_feature_importance.csv"
)

feature_importance.to_csv(
    output_path,
    index=False
)


print("\nFeature importance saved to:")
print(output_path)


print("\n")
print("=" * 60)
print("LEAKAGE ANALYSIS COMPLETE")
print("=" * 60)