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
    f1_score
)

from xgboost import XGBClassifier


# =========================================================
# 1. PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "corporate_tax_risk.csv"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CORPORATE TAX RISK - ABLATION STUDY")
print("=" * 70)

print("Dataset:", df.shape)


# =========================================================
# 3. FEATURES TO TEST
# =========================================================

risk_features = [
    "Previous_Fines",
    "Internal_Control_Score",
    "Effective_Tax_Rate",
    "Restatement_History"
]


# =========================================================
# 4. TARGET
# =========================================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = df["Tax_Risk_Score"].map(target_mapping)


# =========================================================
# 5. FUNCTION TO TRAIN MODEL
# =========================================================

def run_experiment(features, experiment_name):

    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)

    X = df[features].copy()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Process data
    # -----------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

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

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=sample_weights
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        X_test_processed
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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

    print("Accuracy:",
          round(accuracy, 4))

    print("Balanced Accuracy:",
          round(balanced_accuracy, 4))

    print("Macro F1:",
          round(macro_f1, 4))

    return {
        "Experiment": experiment_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": macro_f1
    }


# =========================================================
# 6. ALL FEATURES
# =========================================================

all_features = [
    column
    for column in df.columns
    if column not in [
        "Company_ID",
        "Tax_Risk_Score"
    ]
]

results = []

results.append(
    run_experiment(
        all_features,
        "ALL FEATURES"
    )
)


# =========================================================
# 7. REMOVE ONE FEATURE AT A TIME
# =========================================================

for feature in risk_features:

    remaining_features = [
        column
        for column in all_features
        if column != feature
    ]

    results.append(
        run_experiment(
            remaining_features,
            f"REMOVE: {feature}"
        )
    )


# =========================================================
# 8. REMOVE ALL FOUR
# =========================================================

remaining_features = [
    column
    for column in all_features
    if column not in risk_features
]

results.append(
    run_experiment(
        remaining_features,
        "REMOVE ALL FOUR RISK FEATURES"
    )
)


# =========================================================
# 9. RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# 10. SAVE RESULTS
# =========================================================

output_path = (
    BASE_DIR
    / "models"
    / "ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)


print("\n")
print("=" * 70)
print("ABLATION STUDY COMPLETE")
print("=" * 70)