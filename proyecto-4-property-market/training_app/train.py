from .etl import get_clean_data
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import os
import requests

print(mlflow.__version__)
out_dir = "./models"

MODEL_NAME = "property-market-model"
ALIAS = "prod"
MAXIMIZE = True
METRIC_NAME = "test_score"

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_uri:
  raise ValueError("MLFLOW_TRACKING_URI is not set")
  
mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment("property_market_experiment_v2")
client = MlflowClient()
MIN_IMPROVE = 0.0

mlflow.autolog(log_input_examples= True, log_model_signatures = True, log_models = True, log_datasets = True,
              disable = False, exclusive = False, disable_for_unsupported_versions = False,
               silent = False)

def trainModel():

  params = {
        "fit_intercept": [True],
        "positive": [False]  # ponlo en True solo si sabes que y ≥ 0 y los coeficientes deben ser positivos
    }

  model = LinearRegression()

  X, y = get_clean_data()

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  grid = GridSearchCV(
      estimator=model,
      param_grid=params,
      cv=3,
      n_jobs=-1
  )

  with mlflow.start_run():
      grid.fit(X_train, y_train)

      test_score = grid.score(X_test, y_test)  # R^2 por defecto
      mlflow.log_metric("test_score", test_score)

      best_model = grid.best_estimator_
      signature = infer_signature(X_train, best_model.predict(X_train))

      result = mlflow.sklearn.log_model(
          best_model,
          "model",
          signature=signature,
          registered_model_name=MODEL_NAME,
      )
      new_version = result.registered_model_version

  return new_version, test_score


def _metric_from_run(client: MlflowClient, run_id: str, metric_name: str):
  r = client.get_run(run_id)
  return r.data.metrics.get(metric_name)

def _pick_best_version(client: MlflowClient, model_name: str, metric_name: str, maximize: bool = True):
  versions = client.search_model_versions(f"name = '{model_name}'")
  scored = []
  for mv in versions:
      mval = _metric_from_run(client, mv.run_id, metric_name)
      if mval is not None:
          scored.append((int(mv.version), mval, mv.run_id))
  if not scored:
      raise RuntimeError(f"No versions of '{model_name}' have metric '{metric_name}'.")
  # Tie-breaker: higher metric first (or lower if not maximize), then newer version
  if maximize:
      scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
  else:
      scored.sort(key=lambda t: (-t[1], t[0]), reverse=True)
  best_version, best_metric, best_run = scored[0]
  return best_version, best_metric, scored


def _current_alias_version_or_none(client, MODEL_NAME, ALIAS):
  try:
      return int(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)
  except Exception:
      return None

def send_preprocessor_file():
  file_path = os.path.join(out_dir, "preprocessor.pkl")
  print("file_path", file_path)
  url = f"{os.getenv('PREDICT_API_URL')}/upload_preprocessor"
  with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/octet-stream")}
    response = requests.post(url, files=files)
  if response.status_code == 200:
    print("Preprocessor file sent successfully")
  else:
    print("Failed to send preprocessor file")


def train_and_publish_best():
  print("Sending model to predict API")

  new_version, new_metric = trainModel()

  client = MlflowClient()

  best_version, best_metric, candidates = _pick_best_version(
      client, MODEL_NAME, METRIC_NAME, MAXIMIZE
  )

  current_ver = _current_alias_version_or_none(client, MODEL_NAME, ALIAS)
  should_flip = True
  if current_ver is not None and MIN_IMPROVE > 0.0:
      cur_mv = client.get_model_version(MODEL_NAME, str(current_ver))
      cur_metric = _metric_from_run(client, cur_mv.run_id, METRIC_NAME)
      if cur_metric is not None:
          if MAXIMIZE:
              should_flip = (best_metric >= cur_metric * (1.0 + MIN_IMPROVE))
          else:
              should_flip = (best_metric <= cur_metric * (1.0 - MIN_IMPROVE))
  url = f"{os.getenv('PREDICT_API_URL')}/model"
  if should_flip:
    print("Flipping to best version", best_version)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(best_version))
    alias_target = best_version
    alias_metric = best_metric
    flipped = True
    print("Sending model to predict API")
    send_preprocessor_file()
    requests.post(url)
  else:
    print("Keeping current version", current_ver)
    alias_target = current_ver
    alias_metric = cur_metric
    flipped = False
    print("Sending model to predict API")
    requests.post(url)
  if best_version == current_ver:
    print("Sending preprocessor file")
    send_preprocessor_file()