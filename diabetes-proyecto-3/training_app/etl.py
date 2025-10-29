import pandas as pd
from .dataService import fetch_data, get_raw_column_names
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from .db import create_table, insert_data, get_rows, clear_table, get_rows_with_columns, get_table_columns, delete_table
# load the data

# print(penguins.head())
endl = "#" * 100

column_list =  [
  "Elevation", 
  "Aspect", 
  "Slope", 
  "Horizontal_Distance_To_Hydrology", 
  "Vertical_Distance_To_Hydrology", 
  "Horizontal_Distance_To_Roadways", 
  "Hillshade_9am", 
  "Hillshade_Noon", 
  "Hillshade_3pm", 
  "Horizontal_Distance_To_Fire_Points", 
  "Cover_Type",
  "Wilderness_Area_Rawah",
  "Wilderness_Area_Neota",
  "Wilderness_Area_Commanche",
  "Wilderness_Area_Cache",
  "Soil_Type_C2702",
  "Soil_Type_C2703",
  "Soil_Type_C2704",
  "Soil_Type_C2705",
  "Soil_Type_C2706",
  "Soil_Type_C2717",
  "Soil_Type_C3501",
  "Soil_Type_C3502",
  "Soil_Type_C4201" ,
  "Soil_Type_C4703",
  "Soil_Type_C4704",
  "Soil_Type_C4744",
  "Soil_Type_C4758",
  "Soil_Type_C5101",
  "Soil_Type_C5151",
  "Soil_Type_C6101",
  "Soil_Type_C6102",
  "Soil_Type_C6731",
  "Soil_Type_C7101",
  "Soil_Type_C7102",
  "Soil_Type_C7103",
  "Soil_Type_C7201",
  "Soil_Type_C7202",
  "Soil_Type_C7700",
  "Soil_Type_C7701",
  "Soil_Type_C7702",
  "Soil_Type_C7709",
  "Soil_Type_C7710",
  "Soil_Type_C7745",
  "Soil_Type_C7746",
  "Soil_Type_C7755",
  "Soil_Type_C7756",
  "Soil_Type_C7757",
  "Soil_Type_C7790",
  "Soil_Type_C8703",
  "Soil_Type_C8707",
  "Soil_Type_C8708",
  "Soil_Type_C8771",
  "Soil_Type_C8772",
  "Soil_Type_C8776"
]

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
  data, batch_number = fetch_data()
  rawData = pd.DataFrame(data, columns=get_raw_column_names())
  create_table("raw_data", rawData)
  insert_data("raw_data", rawData)
  return batch_number

def clear_raw_data():
  delete_table("raw_data")


def clean_all_data():
  clear_raw_data()
  clear_clean_data()

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
  create_table("clean_data", clean_data_df, column_list)
  columns = get_table_columns("clean_data")
  insert_data("clean_data", clean_data_df)
  print('✅ Clean data saved to DB', clean_data_df.head())

def get_clean_data():
  rows, columns = get_rows_with_columns("clean_data")
  columns = columns[1:]  # Exclude 'id' column
  le = LabelEncoder()
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  clean_data_df = clean_data_df.drop(columns=["row_hash"])
  y = clean_data_df['Cover_Type']
  X = clean_data_df.drop(columns=["Cover_Type"])

  show_after_cleaning(X, y)
  return X, y