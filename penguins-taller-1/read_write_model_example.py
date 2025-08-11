from etl import get_data
from model import Model
from models import ModelType
import pandas as pd

X, y = get_data()

model = Model(ModelType.NEURAL_NETWORK)
model.train(X, y)

model.load("model.pkl")

# print("Train confusion matrix:")
# print(model.confusion_matrix_train(as_dataframe=True))

# print("Validation confusion matrix:")
# print(model.confusion_matrix_val(as_dataframe=True))

# print("Test confusion matrix:")
# print(model.confusion_matrix_test(as_dataframe=True))

# print("Train metrics:", model.metrics_train())
# print("Validation metrics:", model.metrics_val())
# print("Test metrics:", model.metrics_test())

# model.save("model.pkl")

# Example: take the first row of X as a sample
sample = X.iloc[[0]]  # double brackets keep it as DataFrame
pred = model.predict(sample)
print("Prediction for first sample:", pred)
