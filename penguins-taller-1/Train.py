from etl import get_data
from model import Model
from models import ModelType
import os

# Ruta base: misma carpeta de este script
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Carpeta donde guardaremos los modelos entrenados

os.makedirs(MODELS_DIR, exist_ok=True)


# Cargar y preparar datos
X, y = get_data()

# Lista de modelos a entrenar
model_types = [
    ModelType.RANDOM_FOREST,
    ModelType.SVM,
    ModelType.NEURAL_NETWORK,
    ModelType.LINEAR_REGRESSION
]

# Entrenar y guardar cada modelo
for mtype in model_types:
    model_obj = Model(mtype)
    model_obj.train(X, y)
    model_path = os.path.join(MODELS_DIR, f"model_{mtype.value}.pkl")
    model_obj.save(model_path)
    print(f"✅ Modelo {mtype.value} guardado en {model_path}")