import requests

RAW_COLUMN_NAMES = [
    'encounter_id',
    'patient_nbr',
    'race',
    'gender',
    'age',
    'weight',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'time_in_hospital',
    'payer_code',
    'medical_specialty',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'diag_1',
    'diag_2',
    'diag_3',
    'number_diagnoses',
    'max_glu_serum',
    'A1Cresult',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'change',
    'diabetesMed',
    'readmitted'
    ]

URL = 'http://10.43.100.103:8080/data?group_number=4'

def get_raw_column_names():
    return RAW_COLUMN_NAMES

def fetch_data():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()['data']
        batch_number = response.json()['batch_number']
        print(f"✅ Batch number {batch_number}")
        print(f"✅ Data {data}")
        return data, batch_number
    else:
        print(f"❌ Failed to fetch data: {response.status_code}")

# def store_raw_data():
#   rawData = pd.DataFrame(fetch_data(), columns=get_raw_column_names())
#   create_table("raw_data", rawData)
#   insert_data("raw_data", rawData)

# print("raw data")
# store_raw_data()