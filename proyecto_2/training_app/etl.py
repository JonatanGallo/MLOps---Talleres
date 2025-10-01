import pandas as pd
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from .db import create_table, insert_data, get_rows, clear_table, get_rows_with_columns, get_table_columns, delete_table
# load the data

# print(penguins.head())
endl = "#" * 100


# Cleans, transforms, encodes, and scales the penguins dataset
def clean_data(covertype_data):
  exclude = [ "Wilderness_Area", "Soil_Type"]
  covertype_data[covertype_data.columns.difference(exclude)] = covertype_data[covertype_data.columns.difference(exclude)].apply(
      pd.to_numeric, errors="coerce"
  )
  covertype_data = pf.get_dummies(covertype_data, columns=['Wilderness_Area','Soil_Type'], drop_first=True)
  # Drop any rows that still have NaNs after imputation and encoding
  covertype_data.dropna(inplace=True)
  print("data after cleaning", covertype_data.head())
  return covertype_data

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

def store_raw_data(dataFrame):
  penguins = load_penguins()
  penguins.to_csv("raw_data.csv", index=False)
  create_table("raw_data", penguins)
  insert_data("raw_data", penguins)

def clear_raw_data():
  delete_table("raw_data")

def get_raw_data():
  rows, columns = get_rows_with_columns("raw_data")
  df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  print(df.head())
  return df

def clear_clean_data():
  delete_table("clean_data")

def save_clean_data():
  raw_data = get_raw_data()
  clean_data_df = clean_data(raw_data)
  create_table("clean_data", clean_data_df)
  columns = get_table_columns("clean_data")
  insert_data("clean_data", clean_data_df)

def get_clean_data():
  rows, columns = get_rows_with_columns("clean_data")
  le = LabelEncoder()
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  y = clean_data_df['Cover_Type']
  X = clean_data_df.drop(columns=["Cover_Type"])

  show_after_cleaning(X, y)
  return X, y