import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

OUTPUT_DIR = (
    BASE_DIR
    / "models"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("=" * 70)
print("SHAP EXPLAINABILITY - CORPORATE TAX RISK")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# 3. TARGET ENCODING
# =========================================================

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

y = df["Tax_Risk_Score"].map(
    target_mapping
)


# =========================================================
# 4. FEATURES
# =========================================================

X = df.drop(
    columns=[
        "Tax_Risk_Score",
        "Company_ID"
    ]
)


# =========================================================
# 5. FEATURE TYPES
# =========================================================

categorical_features = (
    X
    .select_dtypes(include=["object"])
    .columns
    .tolist()
)

numerical_features = (
    X
    .select_dtypes(exclude=["object"])
    .columns
    .tolist()
)


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


# =========================================================
# 7. PREPROCESSING
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
# 8. TRANSFORM DATA
# =========================================================

print("\nPreparing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# =========================================================
# 9. CLASS WEIGHTS
# =========================================================

class_counts = np.bincount(
    y_train
)

total_samples = len(y_train)

number_of_classes = len(
    class_counts
)

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


print("\nClass weights:")
print(class_weights)


# =========================================================
# 10. TRAIN XGBOOST
# =========================================================

print("\nTraining XGBoost...")

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


print("XGBoost training complete.")


# =========================================================
# 11. FEATURE NAMES
# =========================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\nNumber of processed features:")
print(len(feature_names))


# =========================================================
# 12. SHAP EXPLAINER
# =========================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)

# Use a sample to keep SHAP computation manageable
sample_size = min(
    500,
    X_test_processed.shape[0]
)

X_shap = X_test_processed[
    :sample_size
]


print(
    "Calculating SHAP values for",
    sample_size,
    "companies..."
)


shap_values = explainer.shap_values(
    X_shap
)


print("SHAP calculation complete.")


# =========================================================
# 13. HANDLE SHAP OUTPUT FORMAT
# =========================================================

if isinstance(shap_values, list):

    # Older SHAP versions:
    # list of classes
    # each item = samples x features

    shap_array = np.stack(
        shap_values,
        axis=2
    )

else:

    shap_array = np.asarray(
        shap_values
    )

    # SHAP can return:
    #
    # samples x classes x features
    # OR
    # samples x features x classes
    #
    # We want:
    #
    # samples x features x classes

    if shap_array.ndim == 3:

        if shap_array.shape[1] == 3:

            # Current SHAP format:
            # samples x classes x features

            shap_array = np.transpose(
                shap_array,
                (0, 2, 1)
            )

        elif shap_array.shape[2] == 3:

            # Already:
            # samples x features x classes

            pass

        else:

            raise ValueError(
                f"Unexpected SHAP shape: "
                f"{shap_array.shape}"
            )

    else:

        raise ValueError(
            f"Unexpected SHAP dimensions: "
            f"{shap_array.shape}"
        )


print("\nSHAP array shape after processing:")
print(shap_array.shape)





# =========================================================
# 14. GLOBAL FEATURE IMPORTANCE
# =========================================================

print("\nCalculating global feature importance...")


mean_abs_shap = np.mean(
    np.abs(shap_array),
    axis=(0, 2)
)


global_importance = pd.DataFrame({

    "Feature": feature_names,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})


global_importance = (
    global_importance
    .sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

print(
    global_importance
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 15. SAVE GLOBAL IMPORTANCE
# =========================================================

global_path = (
    OUTPUT_DIR
    / "shap_global_importance.csv"
)

global_importance.to_csv(
    global_path,
    index=False
)


print("\nGlobal importance saved to:")
print(global_path)


# =========================================================
# 16. SHAP SUMMARY PLOT
# =========================================================

print("\nCreating SHAP summary plot...")


# Use mean SHAP magnitude across classes
summary_values = np.mean(
    np.abs(shap_array),
    axis=2
)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(
    summary_values,
    X_shap,
    feature_names=feature_names,
    show=False
)


plt.title(
    "SHAP Feature Importance - Corporate Tax Risk"
)


plt.tight_layout()


summary_path = (
    OUTPUT_DIR
    / "shap_summary.png"
)


plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nSummary plot saved to:")
print(summary_path)


# =========================================================
# 17. EXPLAIN ONE COMPANY
# =========================================================

company_index = 0

company_data = X_test.iloc[
    company_index
]

company_processed = X_test_processed[
    company_index
]


# Prediction
prediction = model.predict(
    company_processed.reshape(1, -1)
)[0]


probabilities = model.predict_proba(
    company_processed.reshape(1, -1)
)[0]


predicted_class = reverse_mapping[
    int(prediction)
]


print("\n")
print("=" * 70)
print("INDIVIDUAL COMPANY EXPLANATION")
print("=" * 70)


print(
    "\nPredicted Risk:",
    predicted_class
)


print("\nRisk probabilities:")

for class_id, probability in enumerate(
    probabilities
):

    print(
        f"{reverse_mapping[class_id]}: "
        f"{probability * 100:.2f}%"
    )


# =========================================================
# 18. INDIVIDUAL SHAP VALUES
# =========================================================

company_shap = shap_array[
    company_index
]


# SHAP importance across classes
company_importance = np.mean(
    np.abs(company_shap),
    axis=1
)


company_explanation = pd.DataFrame({

    "Feature": feature_names,

    "SHAP_Importance":
        company_importance

})


company_explanation = (
    company_explanation
    .sort_values(
        by="SHAP_Importance",
        ascending=False
    )
)


print("\nTop factors for this company:")

print(
    company_explanation
    .head(10)
    .to_string(index=False)
)


# =========================================================
# 19. SAVE INDIVIDUAL EXPLANATION
# =========================================================

individual_path = (
    OUTPUT_DIR
    / "company_1_shap_explanation.csv"
)

company_explanation.to_csv(
    individual_path,
    index=False
)


print("\nIndividual explanation saved to:")
print(individual_path)


# =========================================================
# 20. COMPANY DETAILS
# =========================================================

print("\n")
print("=" * 70)
print("COMPANY INPUT DATA")
print("=" * 70)

print(
    company_data.to_string()
)


print("\n")
print("=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)