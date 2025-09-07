import pandas as pd
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from .db import create_table, insert_data, get_rows, clear_table, get_rows_with_columns
# load the data

# print(penguins.head())
endl = "#" * 100


def feature_engineering(penguins):
  """Applies feature scaling and other transformations to numerical features."""
  numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  scaler = StandardScaler()
  penguins[numerical_features] = scaler.fit_transform(penguins[numerical_features])
  joblib.dump(scaler, "scaler.pkl")
  return penguins

# Cleans, transforms, encodes, and scales the penguins dataset
def clean_data(penguins):
  penguins['bill_length_mm'] = penguins['bill_length_mm'].fillna(penguins['bill_length_mm'].mean())
  penguins['sex'] = penguins['sex'].fillna('Unknown')
  penguins = pd.get_dummies(penguins, columns=['island', 'sex'], drop_first=True)
  # Convert boolean columns to 0/1 integers
  bool_cols = penguins.select_dtypes(include=['bool']).columns
  penguins[bool_cols] = penguins[bool_cols].astype(int)
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
  print("Initial data")
  print(penguins.head())
  print(penguins.size)
  print(penguins.shape)
  print(endl)

def show_after_cleaning(X, y):
  print("After cleaning")
  print(X.head())
  print(y.head())
  print(X.shape)
  print(y.shape)
  print(endl)

def get_data():
  penguins = load_penguins()
  show_initial_data(penguins)
  # clean the data with NaN values
  penguins = clean_data(penguins)

  le = LabelEncoder()
  y = pd.Series(le.fit_transform(penguins['species']), index=penguins.index, name='species')
  X = penguins.drop(columns=["species", "year"])
  show_after_cleaning(X, y)
  return X, y

def store_raw_data():
  penguins = load_penguins()
  penguins.to_csv("raw_data.csv", index=False)
  create_table("raw_data", penguins)
  insert_data("raw_data", penguins)

def clear_raw_data():
  penguins = load_penguins()
  create_table("raw_data", penguins)
  clear_table("raw_data")

def get_raw_data():
  penguins = load_penguins()
  rows = get_rows("raw_data")
  columns = penguins.columns.tolist()
  df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  print(df.head())
  return df

def get_clean_data():
  rows, columns = get_rows_with_columns("clean_data")
  columns = columns[1:]  # remove id column
  le = LabelEncoder()
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  y = pd.Series(le.fit_transform(clean_data_df['species']), index=clean_data_df.index, name='species')
  X = clean_data_df.drop(columns=["species", "year"])

  show_after_cleaning(X, y)
  return X, y