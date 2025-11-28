import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import resample
import scipy as sp
from matplotlib.colors import ListedColormap
import requests

from sklearn.pipeline import Pipeline

from .db import *

import os, joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import cloudpickle
import gzip
import training_app
import sys

SRC_DATA_URL = os.getenv("SRC_DATA_URL", "http://10.43.100.103:8000/data?group_number=4&day=Tuesday")
RESTART_DATA_URL = os.getenv("RESTART_DATA_URL", "http://10.43.100.103:8000/restart_data_generation?group_number=4&day=Tuesday")


pd.set_option('future.no_silent_downcasting', True)

# load the data
endl = "#" * 100

# Property dataset categorical features
BASE_CAT = [
    "brokered_by",      # agency/broker categorically coded
    "status",           # housing status: listed for sale or listed for construction
    "street",           # street address categorically coded
    "city",             # city name
    "state",            # state name
    "zip_code",         # area zip code (can be treated as categorical)
]

# Numeric features for property dataset
NUMERIC_FEATURES = [
    "bed",              # Number of beds
    "bath",             # Number of bathrooms
    "acre_lot",         # Lot size/Property in acres
    "house_size",       # House area/size/living space in square feet
]

# Date features
DATE_FEATURES = [
    "prev_sold_date",   # Previous sale date
]

# No ID columns for property dataset (or add if needed)
ID_COLUMNS = set()
TARGET = "price"  # Housing price - target variable for regression

def to_str_df(X):
    import pandas as pd
    return pd.DataFrame(X).astype(str)


def load_raw_data(df_temporary = None):
    if(df_temporary):
        return split_data(df_temporary, TARGET)
    response = requests.get(SRC_DATA_URL)
    print("fetch data is starting...")
    columns = set()
    data = response.json()['data']
    print(f"✅ Data {len(data)}")
    df = pd.DataFrame(data)

    return split_data(df, TARGET)


def get_batch_amount(batch_size):
  print("batch_size type", type(batch_size))
  df = pd.read_csv("https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC")
  batch_amount = (df.shape[0] // batch_size) + 1
  return batch_amount
  

def reset_data_generation():
    restart_response = requests.get(RESTART_DATA_URL)
    print(f"All data fetched. Columns saved, restart response: {restart_response}")

# Note: balance_dataset removed - not needed for regression tasks
# For property price prediction (regression), we don't need class balancing

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n--- {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = pd.crosstab(
        pd.Series(y_test, name="Actual"),
        pd.Series(y_pred, name="Predict"),
        margins=True,
    )
    print(cm)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

def plot_model_comparison(results):
    plt.figure(figsize=(14, 7))
    x = np.arange(len(results))
    width = 0.25

    metrics = ["accuracy", "precision", "recall"]
    colors = ["red", "blue", "green"]
    

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[model_name][metric] for model_name in results.keys()]
        plt.bar(
            x + i * width, values, width=width, color=color, alpha=0.7, label=metric
        )

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.xticks(x + width, results.keys())
    plt.legend()
    plt.tight_layout()
    plt.show()

def split_data(df, target_col, test_size=0.10, val_size=0.20, random_state=42):
    # Basic validations
    print("amount of data in split_data ", len(df))
    if (len(df) < 10000):
        return {
            "train": df,
            "validate": [],
            "test": [],
            "to_train": False
        }
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be in (0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' not found in dataframe columns")

    # For regression (continuous target), we don't use stratify
    df_train, df_temp = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
    )

    rel_test_size = test_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=rel_test_size,
        random_state=random_state,
    )

    return {
        "train": df_train.reset_index(drop=True),
        "validate": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
        "to_train": True
    }
def shrink_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    bool_like = []
    for c in df.columns:
        if df[c].dtype == bool:
            bool_like.append(c)
        elif df[c].dtype == object:
            # convert 'True'/'False'/1/0 to 0/1 if applicable
            vals = set(df[c].dropna().unique())
            if vals <= {True, False, 'True', 'False', 1, 0, '1', '0'}:
                df[c] = df[c].map({'True':1,'False':0,True:1,False:0,'1':1,'0':0,1:1,0:0}).astype('int8')

    if bool_like:
        df[bool_like] = df[bool_like].astype('int8')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def prepare_features(df):
    """
    Prepare features for property dataset.
    Handles date features, converts categoricals, and standardizes numerics.
    """
    df2 = df.copy()
    
    # Handle date features - extract useful date components
    for date_col in DATE_FEATURES:
        if date_col in df2.columns:
            # Convert to datetime if not already
            df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
            # Extract year, month, day of year as numeric features
            df2[f"{date_col}_year"] = df2[date_col].dt.year
            df2[f"{date_col}_month"] = df2[date_col].dt.month
            df2[f"{date_col}_day_of_year"] = df2[date_col].dt.dayofyear
            # Calculate days since sale (if applicable)
            if date_col == "prev_sold_date":
                df2["days_since_sale"] = (pd.Timestamp.now() - df2[date_col]).dt.days
            # Drop original date column (we'll use extracted features)
            df2 = df2.drop(columns=[date_col])
    
    # Convert categorical features to 'object' type
    for col in BASE_CAT:
        if col in df2.columns:
            df2[col] = df2[col].astype("object")
    
    # Convert zip_code to string if it's numeric (to treat as categorical)
    if "zip_code" in df2.columns:
        df2["zip_code"] = df2["zip_code"].astype(str)
    
    print("df in prepare_features", df2.head())
    
    # Get list of numeric features (excluding target)
    numerics = list(set(list(df2.select_dtypes(include=[np.number]).columns)))
    if TARGET in numerics:
        numerics.remove(TARGET)
    print("numerics", numerics)

    # Standardize numeric features
    # First convert numeric columns to float to avoid warnings
    for col in numerics:
        if col in df2.columns:
            df2[col] = df2[col].astype(float)
    
    print("df2 in prepare_features", df2.head())

    # Standardize numeric features
    if numerics:
        std = (np.std(df2[numerics], axis=0)).replace(0, 1)
        mean = np.mean(df2[numerics], axis=0)
        df2.loc[:, numerics] = (df2[numerics] - mean) / std
    
    # Create dummy variables for categorical columns
    categorical_columns = [col for col in BASE_CAT if col in df2.columns]
    
    if categorical_columns:
        df_encoded = pd.get_dummies(df2, dtype='int8', columns=categorical_columns, drop_first=True)
    else:
        df_encoded = df2.copy()
    
    print("df_encoded in prepare_features", df_encoded.head())

    # Get all feature columns (excluding target)
    feature_set = [col for col in df_encoded.columns if col != TARGET]

    return df_encoded, feature_set

# Note: map_readmitted_series removed - not needed for regression
# Price is a continuous numeric target, no mapping needed

def process_date_features(X_processed):
    print("X_processed in process_date_features", X_processed.head())
    for date_col in DATE_FEATURES:
        if date_col in X_processed.columns:
            # Convert to datetime if not already
            X_processed[date_col] = pd.to_datetime(X_processed[date_col], errors='coerce')
            # Extract year, month, day of year as numeric features
            X_processed[f"{date_col}_year"] = X_processed[date_col].dt.year
            X_processed[f"{date_col}_month"] = X_processed[date_col].dt.month
            X_processed[f"{date_col}_day_of_year"] = X_processed[date_col].dt.dayofyear
            # Calculate days since sale
            if date_col == "prev_sold_date":
                X_processed["days_since_sale"] = (pd.Timestamp.now() - X_processed[date_col]).dt.days
    return X_processed

def clear_data(raw_data, for_balancing=False):
    """
    Clean and transform raw property data.
    For regression, we don't need balancing, but keep parameter for compatibility.
    """
    print("raw_data -->", raw_data.head())
    
    # Extract target (price) - keep as numeric for regression
    y = raw_data[TARGET].astype(float)
    X = raw_data.drop(columns=[TARGET])
    
    # Build and fit preprocessor on processed data
    prep, groups = build_full_preprocessor(X)
    X_transformed = prep.fit_transform(X)

    print("X_transformed in clear_data", X_transformed)
    
    out_dir = "./models"
    print("out_dir in clear_data:", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print("exists after makedirs?", os.path.isdir(out_dir))
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    print("prep in clear_data", prep)

    if for_balancing:
        print("Saving preprocessor in:", os.path.join(out_dir, "preprocessor.pkl"))
        with open(os.path.join(out_dir, "preprocessor.pkl"), "wb") as f:
            cloudpickle.dump({"prep": prep, "groups": groups}, f)
    
    
    feature_names = prep.get_feature_names_out()
    print("feature_names in clear_data", feature_names)
    # Convert X_transformed to a DataFrame
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    print("X_df in clear_data", X_df.head())
    # Add target back to dataframe
    X_df[TARGET] = y.reset_index(drop=True)
    print("X_df", X_df.head())
    return X_df, feature_names


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
  clean_data_df = clear_data(raw_data)
  create_table("clean_data", clean_data_df, column_list)
  columns = get_table_columns("clean_data")
  insert_data("clean_data", clean_data_df)
  print('✅ Clean data saved to DB', clean_data_df.head())

def get_clean_data():
  rows, columns = get_rows_with_columns("clean_data_train")
  columns = columns[1:]  # Exclude 'id' column
  clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  clean_data_df = clean_data_df.drop(columns=["row_hash"])
  y = clean_data_df[TARGET]  # Use TARGET variable (price)
  X = clean_data_df.drop(columns=[TARGET])
  print("columns deleted", clean_data_df.columns)
  show_after_cleaning(X, y)
  return X, y

def get_raw_data_columns():
    rows, columns = get_rows_with_columns("raw_data_columns")
    columns = columns[1:]  # Exclude 'id' column
    print("columns in get_raw_data_colums", columns)
    clean_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
    clean_data_df = clean_data_df.drop(columns=["row_hash"])
    return clean_data_df

def get_temporary_data():
  rows, columns = get_rows_with_columns("raw_data_temporary")
  columns = columns[1:]  # Exclude 'id' column
  temporary_data_df = pd.DataFrame([row[1:] for row in rows], columns=columns)
  temporary_data_df = temporary_data_df.drop(columns=["row_hash"])
  return temporary_data_df

def _date_and_zip_transform(X: pd.DataFrame) -> pd.DataFrame:
    X = process_date_features(X)
    if "zip_code" in X.columns and X["zip_code"].dtype != "object":
        X["zip_code"] = X["zip_code"].astype(str)
    return X

def build_full_preprocessor(df_raw: pd.DataFrame):
    # Aplica una vez para descubrir columnas
    df_processed = _date_and_zip_transform(df_raw)
    column_prep, groups = build_preprocessor(df_processed)

    full_prep = Pipeline(steps=[
        ("date_zip", FunctionTransformer(_date_and_zip_transform, feature_names_out="one-to-one")),
        ("columns", column_prep),
    ])

    return full_prep, groups

# Note: Removed diabetes-specific functions:
# - _simplify_icd_series: ICD code simplification (not needed for property data)
# - simplify_icd_frame: ICD frame processing (not needed for property data)
# - ordinal_categories: Medication ordinal encoding (not needed for property data)

def build_preprocessor(df: pd.DataFrame):
    """
    Build preprocessor for property dataset.
    Assumes date features have already been processed (extracted to numeric features).
    Handles numeric features and categorical features.
    """
    # Get present categorical features
    print("df in build_preprocessor", df.head())
    present_base = [c for c in BASE_CAT if c in df.columns]
    cat_ohe_cols = [c for c in present_base if c in ["status", "city", "state"]]
    print("present_base", present_base)
    # Get date-derived numeric feature names (they should already be in df)
    date_derived_numeric = []
    for date_col in DATE_FEATURES:
        date_derived_numeric.extend([f"{date_col}_year", f"{date_col}_month", f"{date_col}_day_of_year"])
        if date_col == "prev_sold_date":
            date_derived_numeric.append("days_since_sale")
    print("date_derived_numeric", date_derived_numeric)
    # Get numeric columns (excluding IDs, target, and forced categoricals)
    forced_cats = set(present_base)
    num_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols_all
                if c not in ID_COLUMNS
                and c != TARGET
                and c not in forced_cats]
    print("num_cols", num_cols)
    # Filter to only include date-derived features that actually exist in df
    date_derived_present = [c for c in date_derived_numeric if c in df.columns]
    print("date_derived_present", date_derived_present)
    # Base categoricals (One-Hot Encoding)
    # cat_ohe_cols = present_base
    
    # Build preprocessor
    transformers = []
    
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    
    if cat_ohe_cols:
        transformers.append(("base_cat", OneHotEncoder(
            handle_unknown="ignore", 
            sparse_output=False,
            min_frequency=10,  # Handle infrequent categories
            max_categories=50
        ), cat_ohe_cols))
    print("transformers in build_preprocessor", transformers)
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    print("preprocessor in build_preprocessor", preprocessor)
    groups = {
        "numeric": num_cols,
        "base_cat": cat_ohe_cols,
        "date_derived": date_derived_present,
    }

    print("preprocessor", preprocessor)
    print("groups", groups)
    return preprocessor, groups

