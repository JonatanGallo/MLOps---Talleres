from model import Model
import pandas as pd
import os
import glob
from fastapi import HTTPException
#Ruta base: misma carpeta que este archivo
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.environ.get("MODELS_DIR")

class ModelService:
    def __init__(self):
        self.model_obj = Model()

    def load_model(self, model_name: str):
        if os.path.exists(f"{MODELS_DIR}/{model_name}.pkl"):
            self.model_obj.load(f"{MODELS_DIR}/{model_name}.pkl")
            print(f"✅ Modelo {model_name} cargado desde {MODELS_DIR}")
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")

    def predict(self, features: dict):
        df = pd.DataFrame([features])
        return int(self.model_obj.predict(df)[0])

    def list_models(self):
        return [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(f"{MODELS_DIR}/*.pkl")
        ]
