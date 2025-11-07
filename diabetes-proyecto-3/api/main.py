from fastapi import FastAPI, HTTPException, Depends
from ModelService import ModelService
from penguins import PenguinsType
from dto.model_prediction_request import ModelPredictionRequest
from contextlib import asynccontextmanager
from dto.normalized_request import NormalizedRequest
import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv
import joblib
import shutil

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

model_service = ModelService()
MODEL_STAGE = os.getenv("MODEL_STAGE", "prod")
MODELS_DIR = os.environ.get("MODELS_DIR","/app/models")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-model")
MODEL_PATH = os.path.join(MODELS_DIR, f"model_{MODEL_NAME}.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event logic (e.g., connect to database)
    print("Application startup: Initializing resources...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    yield
    # Shutdown event logic (e.g., close database connection)
    print("Application shutdown: Cleaning up resources...")
    
app = FastAPI(title="Cover Type API", version="1.0", lifespan=lifespan)

@app.get("/models")
def get_models():
    return {"available_models": ["random_forest"]}
         


def normalize_request(req: ModelPredictionRequest) -> NormalizedRequest:
    data_dict = req.model_dump()
    return NormalizedRequest.get_clean_data(data_dict, model_feature_names)

@app.post("/predict")
async def predict_model(
    normalized_req: NormalizedRequest = Depends(normalize_request)
):
    try:
        # Convertimos el objeto request a un diccionario
        model_path = os.path.join(MODELS_DIR, f"model_{model_name}.pkl")
        model = joblib.load(model_path)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
        print(f"Model {model_name} loaded from {model_path}")
        features = normalized_req        
        prediction = model.predict(features)
        return {
            "prediction": prediction[0]
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model/{model_name}")
def load_model(model_name: str):
    model_service.load_model(model_name)
    return {"message": f"Model {model_name} loaded successfully"}

@app.post("/model")
async def save_model():
    mlflow_model_url=f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    print("mlflow_model_url ", mlflow_model_url)
    model = mlflow.sklearn.load_model(mlflow_model_url)

    if os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} already exists, moving to {MODEL_PATH}.bak")
        shutil.move(MODEL_PATH, MODEL_PATH + ".bak")
    joblib.dump(model, MODEL_PATH)
    return {"message": f"Model {MODEL_NAME} saved successfully"}