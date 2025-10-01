import pandas as pd
from palmerpenguins import load_penguins
from .dataService import fetch_data, get_raw_column_names
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
  covertype_data = pd.get_dummies(covertype_data, columns=['Wilderness_Area','Soil_Type'], drop_first=True)
 
  bool_cols = covertype_data.select_dtypes(include=['bool']).columns
  covertype_data[bool_cols] = covertype_data[bool_cols].astype(int)
 
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

def store_raw_data():
  rawData = pd.DataFrame(fetch_data(), columns=get_raw_column_names())
  create_table("raw_data", rawData)
  insert_data("raw_data", rawData)

def clear_raw_data():
  delete_table("raw_data")

def get_raw_data():
  print("Getting raw data from DB")
  rows, columns = get_rows_with_columns("raw_data")
  print("columns raw data", columns)
  df = pd.DataFrame([row[1:] for row in rows], columns=columns[1:])
  df = df.drop(columns=["row_hash"])
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
  print('✅ Clean data saved to DB', clean_data_df.head())

def get_clean_data():
  rows, columns = get_rows_with_columns("clean_data")
  columns = columns[1:]  # Exclude 'id' column
  le = LabelEncoder()
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  y = clean_data_df['Cover_Type']
  X = clean_data_df.drop(columns=["Cover_Type"])

  show_after_cleaning(X, y)
  return X, y