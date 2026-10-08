import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

DATA_PATH = "data/corporate_tax_risk.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 2. BASIC DATA CHECK
# =========================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)


# =========================================================
# 3. TARGET DISTRIBUTION
# =========================================================

print("\n========== TARGET DISTRIBUTION ==========")

target_counts = df["Tax_Risk_Score"].value_counts()

print(target_counts)

print("\nPercentage:")
print(
    df["Tax_Risk_Score"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Plot target distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Tax_Risk_Score",
    order=["Low", "Medium", "High"]
)

plt.title("Corporate Tax Risk Distribution")
plt.xlabel("Tax Risk Level")
plt.ylabel("Number of Companies")

plt.tight_layout()
plt.show()


# =========================================================
# 4. REMOVE IDENTIFIER
# =========================================================

# Company_ID is only an identifier.
# It should not be used as an ML feature.

X = df.drop(
    columns=["Tax_Risk_Score", "Company_ID"]
)

y = df["Tax_Risk_Score"]


# =========================================================
# 5. IDENTIFY FEATURE TYPES
# =========================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\n========== FEATURE TYPES ==========")

print("Categorical features:")
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

print("\n========== SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 7. PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# 8. MODEL EVALUATION FUNCTION
# =========================================================

def evaluate_model(model, model_name):

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    print("Accuracy:", round(accuracy, 4))

    print(
        "Balanced Accuracy:",
        round(balanced_accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["Low", "Medium", "High"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "Medium", "High"]
    )

    print(cm)

    return model


# =========================================================
# 9. LOGISTIC REGRESSION
# =========================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

logistic_model = evaluate_model(
    logistic_model,
    "LOGISTIC REGRESSION"
)


# =========================================================
# 10. DECISION TREE
# =========================================================

tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

tree_model = evaluate_model(
    tree_model,
    "DECISION TREE"
)


# =========================================================
# 11. RANDOM FOREST
# =========================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model = evaluate_model(
    random_forest_model,
    "RANDOM FOREST"
)


# =========================================================
# 12. FEATURE RELATIONSHIP ANALYSIS
# =========================================================

print("\n========== NUMERICAL CORRELATION ==========")

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()

print(
    correlation["Tax_Risk_Score"]
    if "Tax_Risk_Score" in correlation.columns
    else "Target is categorical; ordinal correlation skipped."
)


# =========================================================
# 13. RISK-WISE FEATURE ANALYSIS
# =========================================================

print("\n========== RISK-WISE MEANS ==========")

important_features = [
    "Previous_Fines",
    "Internal_Control_Score",
    "Effective_Tax_Rate",
    "Restatement_History"
]

print(
    df.groupby("Tax_Risk_Score")[
        important_features
    ].mean()
)


# =========================================================
# 14. BOXPLOTS
# =========================================================

for feature in important_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Tax_Risk_Score",
        y=feature,
        order=["Low", "Medium", "High"]
    )

    plt.title(
        f"{feature} by Corporate Tax Risk"
    )

    plt.tight_layout()
    plt.show()


print("\n========== EDA + BASELINE COMPLETE ==========")