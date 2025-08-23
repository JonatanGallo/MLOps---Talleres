from fastapi import FastAPI, HTTPException, Depends
from ModelService import ModelService
from penguins import PenguinsType
from dto.model_prediction_request import ModelPredictionRequest
from contextlib import asynccontextmanager
from dto.normalized_request import NormalizedRequest

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model_service.load_models()
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
    model_name: str,
    normalized_req: NormalizedRequest = Depends(normalize_request)
):
    print(f"Received prediction request for model: {model_name}")
    try:
        model_service.load_model(model_name)

        # Convertimos el objeto request a un diccionario
        features = normalized_req.model_dump()
        print(f"Received prediction request for model: {features}")

        
        prediction = model_service.predict(features)
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