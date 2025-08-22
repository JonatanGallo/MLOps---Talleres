from fastapi import FastAPI, HTTPException, Depends
from models import ModelType
from ModelService import ModelService
from penguins import PenguinsType
from dto.model_prediction_request import ModelPredictionRequest
from contextlib import asynccontextmanager
from dto.normalized_request import NormalizedRequest

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_models()
    # Startup event logic (e.g., connect to database)
    print("Application startup: Initializing resources...")
    yield
    # Shutdown event logic (e.g., close database connection)
    print("Application shutdown: Cleaning up resources...")
    
app = FastAPI(title="Palmer Penguins API", version="1.0", lifespan=lifespan)

@app.get("/models")
def get_models():
    return {"available_models": model_service.list_models()}
         
# --- Nuevo endpoint para selección dinámica de modelo 

def normalize_request(req: ModelPredictionRequest) -> NormalizedRequest:
    return NormalizedRequest.from_prediction_request(req)

@app.post("/predict/{model_name}")
async def predict_model(
    model_name: ModelType,
    normalized_req: NormalizedRequest = Depends(normalize_request)
):
    print(f"Received prediction request for model: {model_name}")
    if model_name.value not in model_service.list_models():
        raise HTTPException(status_code=404, detail=f"Model {model_name.value} not found.")
    try:
        # Convertimos el objeto request a un diccionario
        features = normalized_req.model_dump()
        print(f"Received prediction request for model: {features}")

        
        prediction = model_service.predict(model_name, features)
        return {
            "model_used": model_name,
            "prediction": PenguinsType.get_penguins_by_value(prediction).value[0]
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))