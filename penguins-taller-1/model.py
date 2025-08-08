from etl import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import pickle
from models import ModelType

if __name__ == '__main__':
  X, y = get_data()
  print(X.head())
  print(y.head())


class Model:
  def __init__(self, model_name):
    self.model = None
    self.model_name = model_name

  def train(self, X, y):
    match self.model_name:
      case ModelType.RANDOM_FOREST:
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
      case ModelType.SVM:
        self.model = SVC(kernel='rbf', C=1.0)
      case ModelType.NEURAL_NETWORK:
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
      case ModelType.LINEAR_REGRESSION:
        self.model = LinearRegression()
      case _:
        raise ValueError(f"Model type {self.model_name} not supported")

    self.model.fit(X, y)

  def save(self, model_path):
    with open(model_path, 'wb') as f:
      pickle.dump(self.model, f)

  def load(self, model_path):
    with open(model_path, 'rb') as f:
      self.model = pickle.load(f)

  def evaluate(self, X, y):
    return self.model.score(X, y)

  def get_model(self):
    return self.model
  


  def predict(self, X):
    return self.model.predict(X)