from pydantic import BaseModel, Field


class CompanyInput(BaseModel):
    ownership_type: str
    internal_control_score: float = Field(..., ge=0, le=100)
    effective_tax_rate: float
    audit_likelihood: float
    offshore_transactions: int = Field(..., ge=0)
    previous_fines: float = Field(..., ge=0)
    board_independence: float = Field(..., ge=0, le=100)
    revenue: float = Field(..., ge=0)
    profit_before_tax: float
    leverage_ratio: float
    market_region: str
    governance_score: float = Field(..., ge=0, le=100)
    restatement_history: int = Field(..., ge=0)
    whistleblower_reports: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    risk: str
    risk_score: float
    probabilities: dict
    key_risk_factors: list[str]