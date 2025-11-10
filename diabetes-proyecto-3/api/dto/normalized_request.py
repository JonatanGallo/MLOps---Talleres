from pydantic import BaseModel
from dto.model_prediction_request import ModelPredictionRequest
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, inspect

engine = create_engine("mysql+pymysql://user:password@10.43.100.86:8005/training")

clean_data_columns = ['gender_1', 'admission_type_id_3', 'admission_type_id_4', 'admission_type_id_5',
'discharge_disposition_id_2', 'discharge_disposition_id_7', 'discharge_disposition_id_10', 'discharge_disposition_id_18',
'admission_source_id_4', 'admission_source_id_7', 'admission_source_id_9', 'max_glu_serum_1', 'max_glu_serum_', 'A1Cresult_1',
'A1Cresult_', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other', 'metformin_No', 'metformin_Steady',
'metformin_Up', 'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up', 'nateglinide_Steady', 'chlorpropamide_No',
'chlorpropamide_Steady', 'chlorpropamide_Up', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up', 'glipizide_No',
'glipizide_Steady', 'glipizide_Up', 'glyburide_No', 'glyburide_Steady', 'glyburide_Up', 'tolbutamide_Steady',
'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up', 'rosiglitazone_No', 'rosiglitazone_Steady', 'rosiglitazone_Up',
'acarbose_Steady', 'acarbose_Up', 'troglitazone_Steady', 'tolazamide_Steady', 'tolazamide_Up', 'insulin_No',
'insulin_Steady', 'insulin_Up', 'diabetesMed_0.6069480308903191']

class NormalizedRequest(BaseModel):
    encounter_id: int | None = None
    patient_nbr: int | None = None
    race: str | None
    gender: str | None
    age: str | None
    weight: str | None
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str | None
    medical_specialty: str | None
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str | None
    diag_2: str | None
    diag_3: str | None
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str

    # ---------- medication flags (full list)
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str

    # combinations
    glyburide_metformin: float
    glipizide_metformin: float
    glimepiride_pioglitazone: float
    metformin_rosiglitazone: float
    metformin_pioglitazone: float

    # flags
    change_m: float | None = None
    diabetesMed: str

    
   

    def get_clean_data_columns():
        inspector = inspect(engine)
        excluded = ["id", "row_hash"] 
        columns = [col["name"] for col in inspector.get_columns("clean_data") if col["name"] not in excluded]
        # print("Columns in clean_data:", columns)
        return columns

    def load_and_preprocess_data(raw_data):
        df = raw_data.copy()
        # Drop rows with missing values or invalid data

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
        print("df after replacing admission_source_id", len(df.columns))
        # Encode categorical variables
        categorical_mappings = {
            "change_m": {"Ch": 1, "No": 0},
            "gender": {"Male": 1, "Female": 0},
            "diabetesMed": {"Yes": 1, "No": 0},
            "A1Cresult": {">7": 1, ">8": 1, "Norm": 0, "None": -99},
            "max_glu_serum": {">200": 1, ">300": 1, "Norm": 0, "None": -99}
        }

        
        for col, mapping in categorical_mappings.items():
            df[col] = df[col].replace(mapping).infer_objects(copy=False)


        # Encode age intervals [0-10) - [90-100) from 1-10
        age_mapping = {f"[{i*10}-{(i+1)*10})": i + 1 for i in range(10)}
        df["age"] = df["age"].replace(age_mapping).infer_objects(copy=False)

        # Drop columns with many missing values
        df = df.drop(["weight", "payer_code", "medical_specialty"], axis=1)
        return df
        
    def prepare_features(df):
        print("df in prepare_features head", df.head())
        print("df in prepare_features", len(df.columns))
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

        print("med_columns in prepare_features", len(med_columns))


        # Convert only existing columns
        for col in nominal_features:
            if col in df.columns:
                df[col] = df[col].astype("object")

        print("df in prepare_features", df.head())
        # Get list of only numeric features
        numerics = list(set(list(df._get_numeric_data().columns)))

        # Standardize numeric features
        df2 = df.copy()

        # First convert numeric columns to float to avoid warnings
        for col in numerics:
            df2[col] = df2[col].astype(float)
        print("df2 in prepare_features", df2.head())

        # Now standardize
        std =  (np.std(df2[numerics], axis=0)).replace(0, 1)
        # if np.std(df2[numerics], axis=0) == 0:
        # std = std.replace(0, 1)
        mean = np.mean(df2[numerics], axis=0)
        df2.loc[:, numerics] = (df2[numerics] - mean) / std
        # Remove outliers
        # TODO: remove outliers

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

        print("categorical_columns in prepare_features", len(categorical_columns))
        numeric_features = [    "encounter_id",
            "age",
            "time_in_hospital",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]
        # Create dummy variables
        df_numeric = df2[numeric_features]
        print("df_numeric in prepare_features", df_numeric.iloc[0].to_string())
        df_encoded = pd.get_dummies(df2, dtype='int8', columns=categorical_columns, drop_first=True)
        df_encoded = df_encoded.reindex(columns=clean_data_columns, fill_value=0)
        print("df_encoded in prepare_features", len(df_encoded.columns))

        # Define feature sets
        
   
        # Get all dummy columns for categorical variables
        # dummy_columns = [
        #     col
        #     for col in df_encoded.columns
        #         if any(
        #             col.startswith(prefix)
        #             for prefix in [
        #                 "gender_",
        #                 "admission_type_id_",
        #                 "discharge_disposition_id_",
        #                 "admission_source_id_",
        #                 "max_glu_serum_",
        #                 "A1Cresult_",
        #                 "race_",
        #             ]
        #             + [f"{med}_" for med in med_columns]
        #         )
        # ]
        dummy_columns = []
        for col in df_encoded.columns:
            print("df_encoded columns:", col)
        print("df_encoded after getting dummy columns", len(df_encoded.columns))
        
        # Combine numeric and dummy features
        feature_set = numeric_features + dummy_columns
        print("you are here :D feature_set in prepare_features", feature_set)

        df_encoded = pd.concat([df_numeric, df_encoded], axis=1)
        print("df_encoded final in prepare_features", df_encoded.head())
        print("df_encoded final columns in prepare_features", len(df_encoded.columns))
        # Print the first row of df_encoded completely
        print("First row of df_encoded:", df_encoded.iloc[0].to_string())
        return df_encoded, feature_set


    @classmethod
    def get_clean_data(cls, raw_data):
        print("raw_data in get_clean_data", raw_data.head())
        print("raw_data in get_clean_data", len(raw_data.columns))
        df = cls.load_and_preprocess_data(raw_data)
        df_encoded, feature_set = cls.prepare_features(df)
        X = df_encoded[feature_set]
        return X
    
