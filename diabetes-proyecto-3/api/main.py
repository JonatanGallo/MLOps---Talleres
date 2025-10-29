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

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

model_service = ModelService()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://10.43.100.99:8003")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model_service.load_models()
    # Startup event logic (e.g., connect to database)
    print("Application startup: Initializing resources...")
    yield
    # Shutdown event logic (e.g., close database connection)
    print("Application shutdown: Cleaning up resources...")
    
app = FastAPI(title="Cover Type API", version="1.0", lifespan=lifespan)

@app.get("/models")
def get_models():
    return {"available_models": ["random_forest"]}
         
# --- Nuevo endpoint para selección dinámica de modelo 

model = mlflow.sklearn.load_model("models:/random-forest-regressor@prod")
model_feature_names = model.feature_names_in_


def normalize_request(req: ModelPredictionRequest) -> NormalizedRequest:
    data_dict = req.model_dump()
    return NormalizedRequest.get_clean_data(data_dict, model_feature_names)

@app.post("/predict")
async def predict_model(
    normalized_req: NormalizedRequest = Depends(normalize_request)
):
    try:
        # Convertimos el objeto request a un diccionario
        features = normalized_req        
        prediction = model.predict(features)
        return {
            "prediction": prediction[0]
        }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/load_model/{model_name}")
def load_model(model_name: str):
    model_service.load_model(model_name)
    return {"message": f"Model {model_name} loaded successfully"}