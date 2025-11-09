from pydantic import BaseModel

class ModelPredictionRequest(BaseModel):
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
    metformin: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    change: str
    diabetesMed: str
    
