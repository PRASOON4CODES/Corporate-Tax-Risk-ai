import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "corporate_tax_risk_xgboost.joblib"
)


# Load trained XGBoost pipeline
model = joblib.load(MODEL_PATH)


def predict_company(data):
    """
    Convert API input into the exact feature format
    expected by the trained ML pipeline.
    """

    # Convert dictionary into DataFrame
    df = pd.DataFrame([data])

    # Rename API fields to original dataset column names
    df = df.rename(columns={
        "ownership_type": "Ownership_Type",
        "internal_control_score": "Internal_Control_Score",
        "effective_tax_rate": "Effective_Tax_Rate",
        "audit_likelihood": "Audit_Likelihood",
        "offshore_transactions": "Offshore_Transactions",
        "previous_fines": "Previous_Fines",
        "board_independence": "Board_Independence",
        "revenue": "Revenue",
        "profit_before_tax": "Profit_Before_Tax",
        "leverage_ratio": "Leverage_Ratio",
        "market_region": "Market_Region",
        "governance_score": "Governance_Score",
        "restatement_history": "Restatement_History",
        "whistleblower_reports": "Whistleblower_Reports"
    })

    # Exact order used during training
    expected_columns = [
        "Ownership_Type",
        "Internal_Control_Score",
        "Effective_Tax_Rate",
        "Audit_Likelihood",
        "Offshore_Transactions",
        "Previous_Fines",
        "Board_Independence",
        "Revenue",
        "Profit_Before_Tax",
        "Leverage_Ratio",
        "Market_Region",
        "Governance_Score",
        "Restatement_History",
        "Whistleblower_Reports"
    ]

    df = df[expected_columns]

    # ML prediction
    prediction = model.predict(df)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(df)[0]

    # Convert encoded classes to meaningful labels
    class_mapping = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    # Convert predicted class
    prediction_label = class_mapping[int(prediction)]

    # Convert probability classes
    probability_dict = {
        class_mapping[int(cls)]: float(probability)
        for cls, probability in zip(model.classes_, probabilities)
    }

    return prediction_label, probability_dict