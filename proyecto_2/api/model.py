import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
  print("Model module")


class Model:
  def __init__(self, model_name = ""):
    self.model = None
    self.model_name = model_name
    self.training_data = 0.8 # 80% of the data for training
    self.testing_data = 0.1 # 10% of the data for testing
    self.validation_data = 0.1 # 10% of the data for validation
    self.random_state = 42

  def save(self, model_path):
    with open(model_path, 'wb') as f: # save the model to the file
      pickle.dump(self.model, f)
  #end of save

  def load(self, model_path):
    with open(model_path, 'rb') as f: # load the model from the file
      self.model = pickle.load(f)
  #end of load

  def evaluate(self, X, y):
    return self.model.score(X, y)

  def evaluate_train(self):
    return self.model.score(self.X_train, self.y_train)

  def evaluate_val(self):
    return self.model.score(self.X_val, self.y_val)

  def evaluate_test(self):
    return self.model.score(self.X_test, self.y_test)

  def _confusion_matrix(self, X, y, as_dataframe: bool = False):
      # Only valid for classification models
      if not hasattr(self.model, "classes_"):
          raise ValueError("Confusion matrix is only available for classification models.")
      y_pred = self.model.predict(X)
      cm = confusion_matrix(y, y_pred)
      if as_dataframe:
          labels = list(self.model.classes_)
          return pd.DataFrame(cm, index=labels, columns=labels)
      return cm

  def confusion_matrix_train(self, as_dataframe: bool = False):
      return self._confusion_matrix(self.X_train, self.y_train, as_dataframe)

  def confusion_matrix_val(self, as_dataframe: bool = False):
      return self._confusion_matrix(self.X_val, self.y_val, as_dataframe)

  def confusion_matrix_test(self, as_dataframe: bool = False):
      return self._confusion_matrix(self.X_test, self.y_test, as_dataframe)

  def classification_metrics(self, X, y):
      """Return accuracy, precision, recall, and F1 macro for the given set."""
      y_pred = self.model.predict(X)
      return {
          'accuracy': accuracy_score(y, y_pred),
          'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
          'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
          'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0)
      }

  def metrics_train(self):
      return self.classification_metrics(self.X_train, self.y_train)

  def metrics_val(self):
      return self.classification_metrics(self.X_val, self.y_val)

  def metrics_test(self):
      return self.classification_metrics(self.X_test, self.y_test)
  def get_model(self):
    return self.model
  


  def predict(self, X):
    return self.model.predict(X)