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
import cloudpickle
import shutil
import pandas as pd
import joblib
import gzip
import numpy as np
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

model_service = ModelService()
MODEL_STAGE = os.getenv("MODEL_STAGE", "prod")
MODELS_DIR = os.environ.get("MODELS_DIR","/app/models")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-model")
MODEL_PATH = os.path.join(MODELS_DIR, f"model_{MODEL_NAME}.pkl")

PREP_PATH = os.getenv("PREP_PATH", "/app/models/preprocessor.pkl")  # plain pickle
PREP = None
GROUPS = None  # optional if you want to backfill missing columns

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PREP, GROUPS
    print("🔄 Loading preprocessor at startup...")
    with open(PREP_PATH, "rb") as f:
        payload = cloudpickle.load(f)
    PREP = payload["prep"]
    GROUPS = payload.get("groups")
    print("✅ Preprocessor loaded successfully")
    yield  # <-- this yields control to the app runtime
    print("🧹 Cleaning up resources at shutdown...")

app = FastAPI(title="Diabetes API", version="1.0", lifespan=lifespan)


@app.get("/models")
def get_models():
    return {"available_models": ["random_forest"]}
         


def normalize_request(req: ModelPredictionRequest):
    if PREP is None:
        raise HTTPException(500, "Preprocessor not loaded")
    data_dict = req.model_dump()
    df = pd.DataFrame([data_dict])
    X_new = PREP.transform(df)
    # Convert to plain Python so FastAPI can serialize it
    return X_new.astype(float).tolist()

@app.post("/predict")
async def predict_model(
    normalized_req: list[list[float]] = Depends(normalize_request)
):
    try:
        # Convertimos el objeto request a un diccionario
        model = joblib.load(MODEL_PATH)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {MODEL_NAME} not found.")
        print(f"Model {MODEL_NAME} loaded from {MODEL_PATH}")
        features = normalized_req        
        prediction = model.predict(np.array(features))
        print("prediction in predict_model", prediction)
        return {
           "prediction": int(prediction[0]),
           "features": features
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