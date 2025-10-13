from locust import task, between
from locust.contrib.fasthttp import FastHttpUser

class UsuarioDeCarga(FastHttpUser):
    # Tiempo de espera entre tareas por usuario simulado (en segundos)
    wait_time = between(4, 9)
    connections = 800         # pool grande por worker
    max_reqs_per_conn = 0 
    @task
    def hacer_inferencia(self):
        payload = {
            "Elevation": 2358,
            "Aspect": 8,
            "Slope": 5,
            "Horizontal_Distance_To_Hydrology": 170,
            "Vertical_Distance_To_Hydrology": 19,
            "Horizontal_Distance_To_Roadways": 1354,
            "Hillshade_9am": 214,
            "Hillshade_Noon": 230,
            "Hillshade_3pm": 153,
            "Horizontal_Distance_To_Fire_Points": 342,
            "Wilderness_Area": "Cache",
            "Soil_Type": "C2717"
        }
        # Enviar una petición POST al endpoint /predict
        # response = self.client.get("/models", json=payload)
        response = self.client.post("/predict", json=payload)
        # Opcional: validación de respuesta
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
