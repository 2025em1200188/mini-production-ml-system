from pydantic import BaseModel, Field
from typing import Optional

class ChurnPredictionInput(BaseModel):
    tenure: float = Field(..., ge=0, le=120, description="Tenure in months")
    monthly_charges: float = Field(..., ge=0.0, le=1000.0, description="Monthly charges in USD")
    total_charges: Optional[float] = Field(default=None, ge=0.0, description="Total historical charges")
    support_tickets_30d: int = Field(..., ge=0, le=100, description="Support tickets in last 30 days")
    contract_type: str = Field(..., description="Contract duration type")
    payment_method: str = Field(..., description="Payment method used")
    paperless_billing: str = Field(..., description="Yes/No paperless billing")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12.0,
                "monthly_charges": 75.50,
                "total_charges": 906.00,
                "support_tickets_30d": 3,
                "contract_type": "Month-to-Month",
                "payment_method": "Electronic Check",
                "paperless_billing": "Yes"
            }
        }
    }

class ChurnPredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
    decision_threshold: float
    status: str = "success"
    fallback_applied: bool = False
