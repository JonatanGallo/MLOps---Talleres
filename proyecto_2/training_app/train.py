from .etl import get_clean_data
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

print(mlflow.__version__)

MODEL_NAME = "random-forest-regressor"
ALIAS = "prod"
MAXIMIZE = True
METRIC_NAME = "test_score"

mlflow.set_tracking_uri("http://10.43.100.99:8003")

mlflow.set_experiment("random_forest_experiment")
client = MlflowClient()
MIN_IMPROVE = 0.0

mlflow.autolog(log_input_examples= True, log_model_signatures = True, log_models = True, log_datasets = True,
              disable = False, exclusive = False, disable_for_unsupported_versions = False,
               silent = False)

def trainModel():
  params = {
    # "n_estimators": [33, 66, 200],
    # "max_depth": [2, 4, 6],
    # "max_features": [3, 4, 5]
    "n_estimators": [33],
    "max_depth": [2],
    "max_features": [3]
  }

  rf = RandomForestClassifier()
  searcher = GridSearchCV(estimator=rf, param_grid=params)

  X, y = get_clean_data()

  X_train, X_test, y_train, y_test = train_test_split(X, y)

  with mlflow.start_run(run_name="autolog_with_grid_search") as run:
      searcher.fit(X_train, y_train)
      best = searcher.best_estimator_
      test_score = best.score(X_test, y_test)
      mlflow.log_metric("test_score", float(test_score))
      sig = infer_signature(X_train, best.predict(X_train))
      input_ex = X_train[:5]

      # 💡 Register directly here by giving a registered model name
      result = mlflow.sklearn.log_model(
          sk_model=best,
          artifact_path="model",
          registered_model_name=MODEL_NAME,  # <-- your model name
          signature=sig,
          input_example=input_ex,
      )
      new_version = result.registered_model_version  # string like "7"
  return new_version, test_score


  # client = MlflowClient()
  # # client.set_registered_model_alias("random-forest-regressor", "staging", version)
  # # when ready:
  # client.set_registered_model_alias("random-forest-regressor", "prod", version)

  # model = mlflow.sklearn.load_model("models:/random-forest-regressor@prod")

  # # Use it like a normal sklearn estimator
  # y_pred = model.predict(X_test)
  # print(len(X_test))
  # print(y_pred)

  # client = MlflowClient()

  # for rm in client.search_registered_models():
  #     print("Model:", rm.name)
  #     for v in rm.latest_versions:
  #         print(f"  - version {v.version}, aliases={v.aliases}, run_id={v.run_id}")

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

def train_and_publish_best():
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

  if should_flip:
      print("Flipping to best version", best_version)
      client.set_registered_model_alias(MODEL_NAME, ALIAS, str(best_version))
      alias_target = best_version
      alias_metric = best_metric
      flipped = True
  else:
      alias_target = current_ver
      alias_metric = cur_metric
      flipped = False