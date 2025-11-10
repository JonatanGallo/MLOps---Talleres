from locust import task, between
from locust.contrib.fasthttp import FastHttpUser
import random

class UsuarioDeCarga(FastHttpUser):
    # Tiempo de espera entre tareas por usuario simulado (en segundos)
    wait_time = between(1, 3)
    connections = 800         # pool grande por worker
    max_reqs_per_conn = 0 
    @task
    def hacer_inferencia(self):
        payload = {
            "encounter_id": random.randint(10000, 99999),
            "patient_nbr": random.randint(10000, 99999),
            "race": random.choice(["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]),
            "gender": random.choice(["Male", "Female"]),
            "age": random.choice([f"[{i*10}-{(i+1)*10})" for i in range(10)]),
            "admission_type_id": random.randint(1, 8),
            "discharge_disposition_id": random.randint(1, 30),
            "admission_source_id": random.randint(1, 25),
            "time_in_hospital": random.randint(1, 14),
            "num_lab_procedures": random.randint(1, 100),
            "num_procedures": random.randint(0, 20),
            "num_medications": random.randint(1, 50),
            "number_outpatient": random.randint(0, 10),
            "number_emergency": random.randint(0, 10),
            "number_inpatient": random.randint(0, 10),
            "diag_1": "250.00",
            "diag_2": "250.00",
            "diag_3": "250.00",
            "number_diagnoses": random.randint(1, 10),
            "max_glu_serum": random.choice([">200", ">300", "Norm", "None"]),
            "A1Cresult": random.choice([">7", ">8", "Norm", "None"]),
            "metformin": random.choice(["No", "Down", "Up", "Steady"]),
            "insulin": random.choice(["No", "Down", "Up", "Steady"]),
            "change": random.choice(["No", "Ch"]),
            "diabetesMed": random.choice(["Yes", "No"])
        }
        # Enviar una petición POST al endpoint /predict
        # response = self.client.get("/models", json=payload)
        response = self.client.post("/predict", json=payload)
        # Opcional: validación de respuesta
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
