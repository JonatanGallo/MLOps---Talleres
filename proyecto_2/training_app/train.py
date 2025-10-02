from .etl import get_clean_data
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

print(mlflow.__version__)


mlflow.set_tracking_uri("http://10.43.100.99:8003")

mlflow.set_experiment("random_forest_experiment")

mlflow.autolog(log_input_examples= True, log_model_signatures = True, log_models = True, log_datasets = True,
              disable = False, exclusive = False, disable_for_unsupported_versions = False,
               silent = False)

def trainModel():
  params = {
    "n_estimators": [33, 66, 200],
    "max_depth": [2, 4, 6],
    "max_features": [3, 4, 5]
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
          registered_model_name="random-forest-regressor",  # <-- your model name
          signature=sig,
          input_example=input_ex,
      )
      version = result.registered_model_version  # string like "7"

  client = MlflowClient()
  # client.set_registered_model_alias("random-forest-regressor", "staging", version)
  # when ready:
  client.set_registered_model_alias("random-forest-regressor", "prod", version)

  model = mlflow.sklearn.load_model("models:/random-forest-regressor@prod")

  # Use it like a normal sklearn estimator
  y_pred = model.predict(X_test)
  print(len(X_test))
  print(y_pred)

  client = MlflowClient()

  for rm in client.search_registered_models():
      print("Model:", rm.name)
      for v in rm.latest_versions:
          print(f"  - version {v.version}, aliases={v.aliases}, run_id={v.run_id}")