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
from db import *

pd.set_option('future.no_silent_downcasting', True)

# load the data
endl = "#" * 100


def load_raw_data(batch_number, batch_size):
    df = pd.read_csv("https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC")
    start_index = (batch_number - 1) * batch_size
    end_index = start_index + batch_size if start_index + batch_size < df.shape[0] else df.shape[0]
    df = df.iloc[start_index:end_index]
    return split_data(df, "readmitted")
    

def get_batch_amount(batch_size):
  df = pd.read_csv("https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC")
  batch_amount = (df.shape[0] // batch_size) + 1
  return batch_amount
  

def load_and_preprocess_data(raw_data):
    df = raw_data.copy()
    # Drop rows with missing values or invalid data
    mask = (
        (df[["diag_1", "diag_2", "diag_3"]] == "?").any(axis=1)
        | (df["race"] == "?")
        | (df["discharge_disposition_id"] == 11)  # Expired
        | (df["gender"] == "Unknown/Invalid")
    )
    df = df[~mask]

    # Re-encoding admission type, discharge type and admission source into fewer categories
    df["admission_type_id"] = df["admission_type_id"].replace({2: 1, 7: 1, 6: 5, 8: 5})

    discharge_mappings = {
        6: 1,
        8: 1,
        9: 1,
        13: 1,
        3: 2,
        4: 2,
        5: 2,
        14: 2,
        22: 2,
        23: 2,
        24: 2,
        12: 10,
        15: 10,
        16: 10,
        17: 10,
        25: 18,
        26: 18,
    }
    df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(
        discharge_mappings
    )

    admission_mappings = {
        2: 1,
        3: 1,
        5: 4,
        6: 4,
        10: 4,
        22: 4,
        25: 4,
        15: 9,
        17: 9,
        20: 9,
        21: 9,
        13: 11,
        14: 11,
    }
    df["admission_source_id"] = df["admission_source_id"].replace(admission_mappings)

    # Encode categorical variables
    categorical_mappings = {
        "change_m": {"Ch": 1, "No": 0},
        "gender": {"Male": 1, "Female": 0},
        "diabetesMed": {"Yes": 1, "No": 0},
        "A1Cresult": {">7": 1, ">8": 1, "Norm": 0, "None": -99},
        "max_glu_serum": {">200": 1, ">300": 1, "Norm": 0, "None": -99},
        "readmitted": {">30": 0, "<30": 1, "NO": 0},
    }

    
    for col, mapping in categorical_mappings.items():
        df[col] = df[col].replace(mapping).infer_objects(copy=False)

    print("readmitted unique", df["readmitted"].unique())
    print("readmitted unique", df["readmitted"].value_counts())

    # Encode age intervals [0-10) - [90-100) from 1-10
    age_mapping = {f"[{i*10}-{(i+1)*10})": i + 1 for i in range(10)}
    df["age"] = df["age"].replace(age_mapping).infer_objects(copy=False)

    # Keep only first encounter per patient
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")

    # Drop columns with many missing values
    df = df.drop(["weight", "payer_code", "medical_specialty"], axis=1)
    return df


def balance_dataset(X, y):
    # Separate majority and minority classes
    df_majority = X[y == 0]
    df_minority = X[y == 1]

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=20
    )

    # Combine majority class with upsampled minority class
    X_balanced = pd.concat([df_majority, df_minority_upsampled])
    y_balanced = pd.Series([0] * len(df_majority) + [1] * len(df_minority_upsampled))

    return X_balanced, y_balanced

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
    print("amoun of data in splt_data ", len(df))
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be in (0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' not found in dataframe columns")

    df_train, df_temp = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    rel_test_size = test_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=rel_test_size,
        stratify=df_temp[target_col],
        random_state=random_state,
    )

    # Optional: reset indices for clean downstream merges/joins
    return {
        "train": df_train.reset_index(drop=True),
        "validate": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True)
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
    # Convert data type of nominal features to 'object' type
    nominal_features = [
        "encounter_id",
        "patient_nbr",
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "A1Cresult",
        "age",
        "max_glu_serum",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    # Add medication columns
    med_columns = [
        col
        for col in df.columns
        if col
        in [
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "insulin",
            "glyburide_metformin",
            "glipizide_metformin",
            "glimepiride_pioglitazone",
            "metformin_rosiglitazone",
            "metformin_pioglitazone",
            "change_m",
            "diabetesMed",
        ]
    ]


    # Convert only existing columns
    for col in nominal_features:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Get list of only numeric features
    numerics = list(set(list(df._get_numeric_data().columns)))
    print("numerics", numerics)
    numerics.remove("readmitted")
    print("numerics after removing readmitted", numerics)

    # Standardize numeric features
    df2 = df.copy()

    # First convert numeric columns to float to avoid warnings
    for col in numerics:
        df2[col] = df2[col].astype(float)

    # Now standardize
    df2.loc[:, numerics] = (df2[numerics] - np.mean(df2[numerics], axis=0)) / np.std(
        df2[numerics], axis=0
    )

    # Remove outliers
    df2 = df2[(np.abs(sp.stats.zscore(df2[numerics])) < 3).all(axis=1)]

    # Define categorical columns for dummy variables
    categorical_columns = [
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "max_glu_serum",
        "A1Cresult",
        "race",
    ]

    categorical_columns.extend(med_columns)

    # Create dummy variables
    df_encoded = pd.get_dummies(df2, dtype='int8', columns=categorical_columns, drop_first=True)

    # Define feature sets
    numeric_features = [
        "age",
        "time_in_hospital",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    # Get all dummy columns for categorical variables
    dummy_columns = [
        col
        for col in df_encoded.columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "gender_",
                "admission_type_id_",
                "discharge_disposition_id_",
                "admission_source_id_",
                "max_glu_serum_",
                "A1Cresult_",
                "race_",
            ]
            + [f"{med}_" for med in med_columns]
        )
    ]

    # Combine numeric and dummy features
    feature_set = numeric_features + dummy_columns

    return df_encoded, feature_set

def clear_data(raw_data, for_balancing=False):
  df = load_and_preprocess_data(raw_data)
  prepared_features, feature_set = prepare_features(df)
  df_encoded, feature_set = prepare_features(df)
  X = df_encoded[feature_set]
  y = df_encoded["readmitted"]
  
  print("df_encoded readmitted", y.head())
  if for_balancing:
    # Balance the dataset
    print("Balancing dataset...", y.head())
    X_balanced, y_balanced = balance_dataset(X, y)
    y_balanced = y_balanced.rename("readmitted")
    y_balanced = y_balanced.set_axis(X_balanced.index)
    return X_balanced.join(y_balanced), feature_set
  return df_encoded, feature_set


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
  y = clean_data_df['readmitted']
  X = clean_data_df.drop(columns=["readmitted"])

  show_after_cleaning(X, y)
  return X, y