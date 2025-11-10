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
