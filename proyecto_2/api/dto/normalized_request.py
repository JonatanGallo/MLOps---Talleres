from pydantic import BaseModel
from dto.model_prediction_request import ModelPredictionRequest
import joblib
import pandas as pd
from sqlalchemy import create_engine, inspect

engine = create_engine("mysql+pymysql://user:password@10.43.100.86:8005/training")

class NormalizedRequest(BaseModel):
    Elevation: int
    Aspect: int
    Slope: int
    Horizontal_Distance_To_Hydrology: int
    Vertical_Distance_To_Hydrology: int
    Horizontal_Distance_To_Roadways: int
    Hillshade_9am: int
    Hillshade_Noon: int
    Hillshade_3pm: int
    Horizontal_Distance_To_Fire_Points: int
    Wilderness_Area: str
    Soil_Type: str

    def get_clean_data_columns():
        inspector = inspect(engine)
        excluded = ["id", "row_hash"] 
        columns = [col["name"] for col in inspector.get_columns("clean_data") if col["name"] not in excluded]
        print("Columns in clean_data:", columns)
        return columns

    @classmethod
    def get_clean_data(cls, covertype_data, feature_names_in_):
        cols_from_db = cls.get_clean_data_columns()
        print("cols --->", cols_from_db)
        print("covertype_data", covertype_data)

        new_wilderness_area_colum = "Wilderness_Area_" + covertype_data["Wilderness_Area"]
        new_soil_type_column = "Soil_Type_" + covertype_data["Soil_Type"]
        
        cols = list(ModelPredictionRequest.model_fields.keys())
        covertype_data = pd.DataFrame([covertype_data], columns=cols)
        exclude = [ "Wilderness_Area", "Soil_Type"]
        covertype_data[covertype_data.columns.difference(exclude)] = covertype_data[covertype_data.columns.difference(exclude)].apply(
            pd.to_numeric, errors="coerce"
        )

        covertype_data[new_wilderness_area_colum] = 1
        covertype_data[new_soil_type_column] = 1

        # covertype_data = covertype_data.drop(columns=exclude)

        print("covertype_data after dropping columns", covertype_data)

        bool_cols = covertype_data.select_dtypes(include=['bool']).columns
        covertype_data[bool_cols] = covertype_data[bool_cols].astype(int)

        covertype_data = covertype_data.drop(columns=exclude)

        for col in feature_names_in_:
            if col not in covertype_data.columns:
                covertype_data[col] = 0
        

        covertype_data = covertype_data[feature_names_in_]
        
        print("covertype_data after adding columns", covertype_data)        
        
        # Drop any rows that still have NaNs after imputation and encoding
        covertype_data.dropna(inplace=True)
        print("data after cleaning", covertype_data.head())
        # return without headers
        return covertype_data
        