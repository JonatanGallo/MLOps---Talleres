from pydantic import BaseModel

class ModelPredictionRequest(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island_Dream: int = 0
    island_Torgersen: int = 0
    sex_female: int = 0
    sex_male: int = 0
    
