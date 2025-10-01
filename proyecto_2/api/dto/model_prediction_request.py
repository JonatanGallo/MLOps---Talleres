from pydantic import BaseModel

class ModelPredictionRequest(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    year: int = 0
    
