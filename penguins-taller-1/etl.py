import pandas as pd
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler

# load the data

# print(penguins.head())
endl = "#" * 100


def feature_engineering(penguins):
  """Applies feature scaling and other transformations to numerical features."""
  numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  scaler = StandardScaler()
  penguins[numerical_features] = scaler.fit_transform(penguins[numerical_features])
  return penguins

# Cleans, transforms, encodes, and scales the penguins dataset
def clean_data(penguins):
  penguins['bill_length_mm'] = penguins['bill_length_mm'].fillna(penguins['bill_length_mm'].mean())
  penguins['sex'] = penguins['sex'].fillna('Unknown')
  penguins = pd.get_dummies(penguins, columns=['island', 'sex'], drop_first=True)
  penguins = feature_engineering(penguins)
  # Drop any rows that still have NaNs after imputation and encoding
  penguins.dropna(inplace=True)

  print("Null values", penguins.isnull().sum())
  print(endl)
  print("Data types", penguins.dtypes)
  print(endl)
  print("Species", penguins['species'].value_counts(normalize=True))
  print(endl)
  return penguins

def show_initial_data(penguins):
  print(penguins.head())
  print(penguins.size)
  print(penguins.shape)
  print(endl)

def get_data():
  penguins = load_penguins()
  show_initial_data(penguins)
  # clean the data with NaN values
  penguins = clean_data(penguins)

  y = penguins['species']
  X = penguins.drop(columns=['species'])
  return X, y
