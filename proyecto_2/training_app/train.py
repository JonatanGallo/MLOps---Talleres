from .etl import get_clean_data
from .model import Model
from .models import ModelType
import os


MODELS_DIR = os.environ.get("MODELS_DIR")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(mtype, X, y):
    model_obj = Model(mtype)
    model_obj.train(X, y)
    model_path = os.path.join(MODELS_DIR, f"model_{mtype.value}.pkl")
    print(model_path)
    print(MODELS_DIR)
    model_obj.save(model_path)
    print(f"✅ Modelo {mtype.value} guardado en {model_path}")


def save_all_models():
  X, y = get_clean_data()
  save_model(ModelType.RANDOM_FOREST, X, y)
  save_model(ModelType.SVM, X, y)
  save_model(ModelType.NEURAL_NETWORK, X, y)

def hello():
  print("Hello World from train.py")