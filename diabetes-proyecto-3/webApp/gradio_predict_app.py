import os
import json
import requests
import gradio as gr

# Configure the API endpoint (override with env var PREDICT_API_URL)
API_URL = os.getenv("PREDICT_API_URL") + "/predict"
PORT = int(os.getenv("GRADIO_PORT"))

def _none_if_empty(s: str):
    """Return None if s is empty/blank; else return s."""
    if s is None:
        return None
    s = str(s).strip()
    return None if s == "" else s

def _int_or_none(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(v)
    except Exception:
        return None

def _float_or_zero(v):
    try:
        if v is None or str(v).strip() == "":
            return 0.0
        return float(v)
    except Exception:
        return 0.0

def predict(
    encounter_id,
    patient_nbr,
    race,
    gender,
    age,
    weight,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    payer_code,
    medical_specialty,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    diag_1,
    diag_2,
    diag_3,
    number_diagnoses,
    max_glu_serum,
    A1Cresult,
    metformin,
    repaglinide,
    nateglinide,
    chlorpropamide,
    glimepiride,
    acetohexamide,
    glipizide,
    glyburide,
    tolbutamide,
    pioglitazone,
    rosiglitazone,
    acarbose,
    miglitol,
    troglitazone,
    tolazamide,
    examide,
    citoglipton,
    insulin,
    glyburide_metformin,
    glipizide_metformin,
    glimepiride_pioglitazone,
    metformin_rosiglitazone,
    metformin_pioglitazone,
    change_m,
    diabetesMed,
):
    # Build payload matching your API schema
    payload = {
        "encounter_id": _int_or_none(encounter_id),
        "patient_nbr": _int_or_none(patient_nbr),
        "race": race,
        "gender": gender,
        "age": age,
        "weight": _none_if_empty(weight),  # null if blank
        "admission_type_id": _int_or_none(admission_type_id),
        "discharge_disposition_id": _int_or_none(discharge_disposition_id),
        "admission_source_id": _int_or_none(admission_source_id),
        "time_in_hospital": _int_or_none(time_in_hospital),
        "payer_code": _none_if_empty(payer_code),  # null if blank
        "medical_specialty": _none_if_empty(medical_specialty),
        "num_lab_procedures": _int_or_none(num_lab_procedures),
        "num_procedures": _int_or_none(num_procedures),
        "num_medications": _int_or_none(num_medications),
        "number_outpatient": _int_or_none(number_outpatient),
        "number_emergency": _int_or_none(number_emergency),
        "number_inpatient": _int_or_none(number_inpatient),
        "diag_1": _none_if_empty(diag_1),
        "diag_2": _none_if_empty(diag_2),
        "diag_3": _none_if_empty(diag_3),
        "number_diagnoses": _int_or_none(number_diagnoses),
        "max_glu_serum": _none_if_empty(max_glu_serum) if max_glu_serum is not None else "",
        "A1Cresult": _none_if_empty(A1Cresult) if A1Cresult is not None else "",
        # Medication directions: typical values are "No", "Steady", "Up", "Down"
        "metformin": metformin,
        "repaglinide": repaglinide,
        "nateglinide": nateglinide,
        "chlorpropamide": chlorpropamide,
        "glimepiride": glimepiride,
        "acetohexamide": acetohexamide,
        "glipizide": glipizide,
        "glyburide": glyburide,
        "tolbutamide": tolbutamide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "acarbose": acarbose,
        "miglitol": miglitol,
        "troglitazone": troglitazone,
        "tolazamide": tolazamide,
        "examide": examide,
        "citoglipton": citoglipton,
        "insulin": insulin,
        # Combo meds & flags as floats (per your example)
        "glyburide_metformin": _float_or_zero(glyburide_metformin),
        "glipizide_metformin": _float_or_zero(glipizide_metformin),
        "glimepiride_pioglitazone": _float_or_zero(glimepiride_pioglitazone),
        "metformin_rosiglitazone": _float_or_zero(metformin_rosiglitazone),
        "metformin_pioglitazone": _float_or_zero(metformin_pioglitazone),
        "change_m": _float_or_zero(change_m),
        "diabetesMed": diabetesMed,  # "YES"/"NO"
    }

    status = f"POST {API_URL}"
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        status += f"\nStatus: {resp.status_code}"
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}

        # Try to surface a likely prediction field
        pretty = None
        if isinstance(data, dict):
            for key in ("prediction", "pred", "class", "label", "result", "readmitted"):
                if key in data:
                    pretty = f"**Model output ({key})**: {data[key]}"
                    break
        if pretty is None:
            pretty = "**Model output**: (see JSON below)"

        return payload, data, pretty, status

    except requests.exceptions.RequestException as e:
        return payload, {"error": str(e)}, "**Request failed** — check API_URL and server logs.", status

MED_DIRS = ["No", "Steady", "Up", "Down"]

AGE_BUCKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

RACES = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"]
GENDERS = ["Male", "Female", "Unknown/Invalid", "?"]

with gr.Blocks(title="Diabetes Readmission — via /predict") as demo:
    gr.Markdown(
        """
        # 🩺 Diabetes Readmission — via `/predict`
        Fill the fields and click **Predict**.  
        This app will POST the JSON body to your API and display the response.

        > Configure the endpoint with env var: `PREDICT_API_URL`  
        > Default: `http://localhost:8000/predict`
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Patient & Encounter")
            encounter_id = gr.Number(label="encounter_id", value=39877476, precision=0)
            patient_nbr = gr.Number(label="patient_nbr", value=4226301, precision=0)
            race = gr.Dropdown(label="race", choices=RACES, value="Caucasian")
            gender = gr.Dropdown(label="gender", choices=GENDERS, value="Male")
            age = gr.Dropdown(label="age", choices=AGE_BUCKETS, value="[50-60)")
            weight = gr.Textbox(label="weight (blank = null)", value="")  # send null if blank

            gr.Markdown("### Admission / Discharge")
            admission_type_id = gr.Number(label="admission_type_id", value=1, precision=0)
            discharge_disposition_id = gr.Number(label="discharge_disposition_id", value=1, precision=0)
            admission_source_id = gr.Number(label="admission_source_id", value=7, precision=0)
            time_in_hospital = gr.Number(label="time_in_hospital", value=2, precision=0)
            payer_code = gr.Textbox(label="payer_code (blank = null)", value="")
            medical_specialty = gr.Textbox(label="medical_specialty", value="Family/GeneralPractice")

            gr.Markdown("### Procedure / Meds Counts")
            num_lab_procedures = gr.Number(label="num_lab_procedures", value=35, precision=0)
            num_procedures = gr.Number(label="num_procedures", value=0, precision=0)
            num_medications = gr.Number(label="num_medications", value=7, precision=0)
            number_outpatient = gr.Number(label="number_outpatient", value=0, precision=0)
            number_emergency = gr.Number(label="number_emergency", value=0, precision=0)
            number_inpatient = gr.Number(label="number_inpatient", value=0, precision=0)

            gr.Markdown("### Diagnoses")
            diag_1 = gr.Textbox(label="diag_1", value="434")
            diag_2 = gr.Textbox(label="diag_2", value="250.52")
            diag_3 = gr.Textbox(label="diag_3", value="250.42")
            number_diagnoses = gr.Number(label="number_diagnoses", value=9, precision=0)

            gr.Markdown("### Lab Results")
            max_glu_serum = gr.Textbox(label='max_glu_serum (e.g., "None", ">200", ">300", or blank)', value="")
            A1Cresult = gr.Textbox(label='A1Cresult (e.g., "None", ">7", ">8", or blank)', value="")

            gr.Markdown("### Medications (direction)")
            metformin = gr.Dropdown(label="metformin", choices=MED_DIRS, value="No")
            repaglinide = gr.Dropdown(label="repaglinide", choices=MED_DIRS, value="No")
            nateglinide = gr.Dropdown(label="nateglinide", choices=MED_DIRS, value="No")
            chlorpropamide = gr.Dropdown(label="chlorpropamide", choices=MED_DIRS, value="No")
            glimepiride = gr.Dropdown(label="glimepiride", choices=MED_DIRS, value="No")
            acetohexamide = gr.Dropdown(label="acetohexamide", choices=MED_DIRS, value="No")
            glipizide = gr.Dropdown(label="glipizide", choices=MED_DIRS, value="No")
            glyburide = gr.Dropdown(label="glyburide", choices=MED_DIRS, value="No")
            tolbutamide = gr.Dropdown(label="tolbutamide", choices=MED_DIRS, value="No")
            pioglitazone = gr.Dropdown(label="pioglitazone", choices=MED_DIRS, value="No")
            rosiglitazone = gr.Dropdown(label="rosiglitazone", choices=MED_DIRS, value="No")
            acarbose = gr.Dropdown(label="acarbose", choices=MED_DIRS, value="No")
            miglitol = gr.Dropdown(label="miglitol", choices=MED_DIRS, value="No")
            troglitazone = gr.Dropdown(label="troglitazone", choices=MED_DIRS, value="No")
            tolazamide = gr.Dropdown(label="tolazamide", choices=MED_DIRS, value="No")
            examide = gr.Dropdown(label="examide", choices=MED_DIRS, value="No")
            citoglipton = gr.Dropdown(label="citoglipton", choices=MED_DIRS, value="No")
            insulin = gr.Dropdown(label="insulin", choices=MED_DIRS, value="No")

            gr.Markdown("### Combo Meds & Flags")
            glyburide_metformin = gr.Number(label="glyburide_metformin", value=0.0)
            glipizide_metformin = gr.Number(label="glipizide_metformin", value=0.0)
            glimepiride_pioglitazone = gr.Number(label="glimepiride_pioglitazone", value=0.0)
            metformin_rosiglitazone = gr.Number(label="metformin_rosiglitazone", value=0.0)
            metformin_pioglitazone = gr.Number(label="metformin_pioglitazone", value=0.0)
            change_m = gr.Number(label="change_m", value=0.0)
            diabetesMed = gr.Dropdown(label='diabetesMed', choices=["YES", "NO"], value="YES")

            predict_btn = gr.Button("🚀 Predict", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Request preview")
            req_json = gr.JSON()
            gr.Markdown("### Response from API")
            resp_json = gr.JSON()
            summary_md = gr.Markdown()
            status_md = gr.Markdown()

    predict_btn.click(
        fn=predict,
        inputs=[
            encounter_id, patient_nbr, race, gender, age, weight,
            admission_type_id, discharge_disposition_id, admission_source_id,
            time_in_hospital, payer_code, medical_specialty,
            num_lab_procedures, num_procedures, num_medications,
            number_outpatient, number_emergency, number_inpatient,
            diag_1, diag_2, diag_3, number_diagnoses,
            max_glu_serum, A1Cresult,
            metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
            glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone,
            acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton, insulin,
            glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone,
            metformin_rosiglitazone, metformin_pioglitazone, change_m, diabetesMed,
        ],
        outputs=[req_json, resp_json, summary_md, status_md],
        api_name="predict_diabetes",
    )

    gr.Examples(
        label="Examples",
        examples=[
            [39877476, 4226301, "Caucasian", "Male", "[50-60)", "", 1, 1, 7, 2, "", "Family/GeneralPractice",
             35, 0, 7, 0, 0, 0, "434", "250.52", "250.42", 9, "", "",
             "No", "No", "No", "No", "No", "No", "No", "No", "No", "No",
             "No", "No", "No", "No", "No", "No", "No", "No",
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "YES"],
        ],
        inputs=[
            encounter_id, patient_nbr, race, gender, age, weight,
            admission_type_id, discharge_disposition_id, admission_source_id,
            time_in_hospital, payer_code, medical_specialty,
            num_lab_procedures, num_procedures, num_medications,
            number_outpatient, number_emergency, number_inpatient,
            diag_1, diag_2, diag_3, number_diagnoses,
            max_glu_serum, A1Cresult,
            metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
            glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone,
            acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton, insulin,
            glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone,
            metformin_rosiglitazone, metformin_pioglitazone, change_m, diabetesMed,
        ],
    )

    with gr.Accordion("cURL example", open=False):
        curl_md = gr.Markdown(
            f"""
```bash
curl -X POST {API_URL} \\
  -H 'Content-Type: application/json' \\
  -d '{json.dumps({
    "encounter_id": 39877476,
    "patient_nbr": 4226301,
    "race": "Caucasian",
    "gender": "Male",
    "age": "[50-60)",
    "weight": None,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 2,
    "payer_code": None,
    "medical_specialty": "Family/GeneralPractice",
    "num_lab_procedures": 35,
    "num_procedures": 0,
    "num_medications": 7,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "434",
    "diag_2": "250.52",
    "diag_3": "250.42",
    "number_diagnoses": 9,
    "max_glu_serum": "",
    "A1Cresult": "",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "No",
    "glyburide_metformin": 0.0,
    "glipizide_metformin": 0.0,
    "glimepiride_pioglitazone": 0.0,
    "metformin_rosiglitazone": 0.0,
    "metformin_pioglitazone": 0.0,
    "change_m": 0.0,
    "diabetesMed": "YES"
}, ensure_ascii=False)}'
 """
    )

print(f"Launching Gradio on port {PORT}")
demo.launch(server_name="0.0.0.0", server_port=PORT)
