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
            "bath": 3.0,
            "prev_sold_date": "2016-09-06",
            "state": "Georgia",
            "status": "Georgia",
            "city": "Lawrenceville",
            "zip_code": "30046.0",
            "house_size": 1776.0,
            "bed": 3.0,
            "brokered_by": "83562.0",
            "street": "512456.0",
            "acre_lot": 0.07
        }
 
        # Enviar una petición POST al endpoint /predict
        # response = self.client.get("/models", json=payload)
        response = self.client.post("/predict", json=payload)
        # Opcional: validación de respuesta
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
