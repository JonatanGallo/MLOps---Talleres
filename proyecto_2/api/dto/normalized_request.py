from pydantic import BaseModel
from dto.model_prediction_request import ModelPredictionRequest
import joblib

class NormalizedRequest(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island_Dream: int = 0
    island_Torgersen: int = 0
    sex_female: int = 0
    sex_male: int = 0


    @classmethod
    def from_prediction_request(cls, req: ModelPredictionRequest):
        """Create a NormalizedRequest from a ModelPredictionRequest.
        This matches the training one-hot encoding with two columns per categorical."""
        scaler = joblib.load("scaler.pkl")
        numerical_features = [[
            req.bill_length_mm,
            req.bill_depth_mm,
            req.flipper_length_mm,
            req.body_mass_g
        ]]
        scaled_features = scaler.transform(numerical_features)[0]

        # Normalize categorical variables (to match training one-hot encoding with two columns per categorical)
        island = str(req.island).strip().lower() if req.island is not None else ""
        sex = str(req.sex).strip().lower() if req.sex is not None else ""

        # One-hot encoding for island: Dream and Torgersen as separate columns (Biscoe/other -> both 0)
        island_Dream = 1 if island == "dream" else 0
        island_Torgersen = 1 if island == "torgersen" else 0

        # One-hot encoding for sex: female and male as separate columns (unknown/other -> both 0)
        sex_female = 1 if sex == "female" else 0
        sex_male = 1 if sex == "male" else 0

        return cls(
            bill_length_mm=scaled_features[0],
            bill_depth_mm=scaled_features[1],
            flipper_length_mm=scaled_features[2],
            body_mass_g=scaled_features[3],
            island_Dream=island_Dream,
            island_Torgersen=island_Torgersen,
            sex_female=sex_female,
            sex_male=sex_male
        )
