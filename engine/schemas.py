from pydantic import BaseModel, Field
from typing import Dict, List, Any


class ExperimentDesign(BaseModel):
    control: Dict[str, Any] = Field(..., description="Control variant configuration")
    treatment: Dict[str, Any] = Field(..., description="Treatment variant configuration")
    sample_size: int = Field(..., gt=0, description="Sample size per variant")
    metric: str = Field(..., description="Primary evaluation metric")
    
    class Config:
        json_schema_extra = {
            "example": {
                "control": {"cta_color": "blue", "discount": 0},
                "treatment": {"cta_color": "green", "discount": 10},
                "sample_size": 1000,
                "metric": "conversion_rate"
            }
        }


class BayesianResult(BaseModel):
    lift_mean: float = Field(..., description="Mean posterior lift (treatment - control)")
    prob_treatment_better: float = Field(..., ge=0.0, le=1.0, description="Probability that treatment > control")
    ci_95: List[float] = Field(..., min_length=2, max_length=2, description="95% credible interval [lower, upper]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lift_mean": 0.015,
                "prob_treatment_better": 0.87,
                "ci_95": [0.005, 0.025]
            }
        }
