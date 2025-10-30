import pandas as pd
from .dataService import fetch_data, get_raw_column_names
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from .db import create_table, insert_data, get_rows, clear_table, get_rows_with_columns, get_table_columns, delete_table
# load the data

# print(penguins.head())
endl = "#" * 100

batch_size = 15000

# Cleans, transforms, encodes, and scales the penguins dataset
# def clean_data(covertype_data):
#   exclude = [ "Wilderness_Area", "Soil_Type"]
#   covertype_data[covertype_data.columns.difference(exclude)] = covertype_data[covertype_data.columns.difference(exclude)].apply(
#       pd.to_numeric, errors="coerce"
#   )
#   covertype_data = pd.get_dummies(covertype_data, columns=['Wilderness_Area','Soil_Type'], drop_first=True)
 
#   bool_cols = covertype_data.select_dtypes(include=['bool']).columns
#   covertype_data[bool_cols] = covertype_data[bool_cols].astype(int)
 
#   # Drop any rows that still have NaNs after imputation and encoding
#   covertype_data.dropna(inplace=True)
#   print("data after cleaning", covertype_data.head())
#   return covertype_data

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

# Save raw data functions
def store_data(fetchData ,table_name, columns):
    data = pd.DataFrame(fetchData, columns=columns)
    create_table(table_name, data)
    insert_data(table_name, data)

def store_raw_data(count):
  raw_data = load_raw_data(count, batch_size)
  store_data(raw_data['train'], "raw_data_train", get_raw_column_names())
  store_data(raw_data['validate'], "raw_data_validate", get_raw_column_names())
  store_data(raw_data['test'], "raw_data_test", get_raw_column_names())

#clears raw data functions
def clear_raw_data():
  delete_table("raw_data_train")
  delete_table("raw_data_validate")
  delete_table("raw_data_test")

def clean_all_data():
  clear_raw_data()
  clear_clean_data()

# Get raw data from DB
def get_raw_data(table_name):
  print(f"Getting raw data from {table_name} in DB")
  rows, columns = get_rows_with_columns(table_name)
  print("columns raw data", columns)
  df = pd.DataFrame([row[1:] for row in rows], columns=columns[1:])
  df = df.drop(columns=["row_hash"])
  print(df.head())
  return df

# CLEAN DATA FUNCTIONS

# clear clean data functions
def clear_clean_data():
    delete_table("clean_data_train")
    delete_table("clean_data_validate")
    delete_table("clean_data_test")

def save_clean_data(table_sufix, must_balance=False):
    raw_data = get_raw_data("raw_data_" + table_sufix)
    clean_data_df, column_list = clean_data(raw_data, must_balance)
    create_table("clean_data_" + table_sufix, clean_data_df, column_list)
    columns = get_table_columns("clean_data_" + table_sufix)
    insert_data("clean_data_" + table_sufix, clean_data_df)
    print('✅ Clean data saved to DB', clean_data_df.head())

def save_all_clean_data():
    save_clean_data("train", True)
    save_clean_data("validate")
    save_clean_data("test")

def get_clean_data(table_sufix):
  rows, columns = get_rows_with_columns("clean_data_" + table_sufix)
  columns = columns[1:]  # Exclude 'id' column
  le = LabelEncoder()
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  clean_data_df = clean_data_df.drop(columns=["row_hash"])
  y = clean_data_df['readmitted']
  X = clean_data_df.drop(columns=["readmitted"])

  show_after_cleaning(X, y)
  return X, y