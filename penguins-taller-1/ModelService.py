from model import Model
from models import ModelType
from dto.model_prediction_request import ModelPredictionRequest
import pandas as pd
import os

#Ruta base: misma carpeta que este archivo
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

class ModelService:
    def __init__(self):
        self.models = {}

    def load_models(self):
        for mtype in [ModelType.RANDOM_FOREST, ModelType.SVM, ModelType.NEURAL_NETWORK, ModelType.LINEAR_REGRESSION]:
            model_path = os.path.join(MODELS_DIR, f"model_{mtype.value}.pkl")
            if os.path.exists(model_path):
                model_obj = Model(mtype)
                model_obj.load(model_path)
                self.models[mtype.value] = model_obj
                print(f"✅ Modelo {mtype.value} cargado desde {model_path}")
            else:
                print(f"⚠ Modelo {mtype.value} no encontrado en {model_path}")

    def predict(self, model_name: ModelType, features: dict):
   
        df = pd.DataFrame([features])
        if model_name == ModelType.LINEAR_REGRESSION:
            pred = self.models[model_name.value].predict(df)
            return int(round(pred[0]))
        return int(self.models[model_name.value].predict(df)[0])

    def list_models(self):
        return list(self.models.keys())
