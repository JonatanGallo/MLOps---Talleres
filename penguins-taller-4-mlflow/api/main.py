from fastapi import FastAPI, HTTPException, Depends
from ModelService import ModelService
from penguins import PenguinsType
from dto.model_prediction_request import ModelPredictionRequest
from contextlib import asynccontextmanager
from dto.normalized_request import NormalizedRequest
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

model_service = ModelService()
import os
MODEL_STAGE = os.getenv("MODEL_STAGE", "prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model_service.load_models()
    # Startup event logic (e.g., connect to database)
    print("Application startup: Initializing resources...")
    mlflow.set_tracking_uri("http://10.43.100.99:8003")
    yield
    # Shutdown event logic (e.g., close database connection)
    print("Application shutdown: Cleaning up resources...")
    
app = FastAPI(title="Palmer Penguins API", version="1.0", lifespan=lifespan)

@app.get("/models")
def get_models():
    print("in Get ModelS:")
    client = MlflowClient()
    list_models = list()
    print("registered models: ", client.search_registered_models())
    for rm in client.search_registered_models():
        print("Model:", rm.name)
        list_models.append(rm.name)
    return {"available_models": list_models}
         
# --- Nuevo endpoint para selección dinámica de modelo 

def normalize_request(req: ModelPredictionRequest) -> NormalizedRequest:
    return NormalizedRequest.from_prediction_request(req)

@app.post("/predict/{model_name}")
async def predict_model(
    model_name: str,
    normalized_req: NormalizedRequest = Depends(normalize_request)
):
    print(f"Received prediction request for model: {model_name}")
    try:
        mlflow_model_url=f"models:/{model_name}@{MODEL_STAGE}"
        print("mlflow_model_url ", mlflow_model_url)
        model = mlflow.sklearn.load_model(mlflow_model_url)

        # Convertimos el objeto request a un diccionario
        features = normalized_req.model_dump()
        print(f"Received prediction request for model: {features}")

        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        return {
            "model_used": model_name,
            "prediction": PenguinsType.get_penguins_by_value(prediction).value[0]
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/load_model/{model_name}")
def load_model(model_name: str):
    model_service.load_model(model_name)
    return {"message": f"Model {model_name} loaded successfully"}