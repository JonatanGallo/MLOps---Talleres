from locust import task, between
from locust.contrib.fasthttp import FastHttpUser

class UsuarioDeCarga(FastHttpUser):
    # Tiempo de espera entre tareas por usuario simulado (en segundos)
    wait_time = between(1, 3)
    connections = 800         # pool grande por worker
    max_reqs_per_conn = 0 
    @task
    def hacer_inferencia(self):
        payload = {
            "encounter_id": 1,
            "patient_nbr": 27303720,
            "race": "Caucasian",
            "gender": "Female",
            "age": "[60-70)",
            "weight": "[75-100)",
            "admission_type_id": 6,
            "discharge_disposition_id": 18,
            "admission_source_id": 5,
            "time_in_hospital": 3,
            "payer_code": null,
            "medical_specialty": "InternalMedicine",
            "num_lab_procedures": 42,
            "num_procedures": 0,
            "num_medications": 1,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "diag_1": "486",
            "diag_2": "401",
            "diag_3": "250",
            "number_diagnoses": 7,
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
            "diabetesMed": "NO"
        }
        # Enviar una petición POST al endpoint /predict
        # response = self.client.get("/models", json=payload)
        response = self.client.post("/predict", json=payload)
        # Opcional: validación de respuesta
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
