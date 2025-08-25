# 🐧 MLOps - Talleres Clasificador de Pingüinos

Este repositorio contiene el código usado para entrenar y desplegar un modelo de predicción de clases de pingüinos basado en el dataset disponible en: [palmerpenguins](https://pypi.org/project/palmerpenguins/). El proyecto abarca desde la preparación de datos y el entrenamiento de modelos hasta el desplieque de una API REST para realizar predicciones.

---

## Arquitectura de Servicios

- Servicio de Predicción: API REST con FastAPI para inferencia de modelos

- Servicio de Entrenamiento: JupyterLab para desarrollo y entrenamiento de modelos

- Volúmenes Compartidos: Sistema de archivos compartido para modelos entrenados

## Entrenamiento de Modelos

- Notebook interactivo (TrainModels.ipynb) para entrenamiento y experimentación

- Soporte para múltiples tipos de modelos con selección dinámica

- Persistencia automática de modelos entrenados

## Mejoras en la API

- Endpoint para listado de modelos disponibles

- Sistema de normalización de datos de entrada

- Manejo de errores mejorado

- Documentación interactiva automática

## Creación del modelo

- El dataset se carga usando el método `load_penguins` expuesto en la librería del proyecto `palmerpenguins`.  
- Se convierten columnas categóricas a numéricas usando One-shot Encoding.
- Se hace una escala para mantener la desviación estandar por debajo de 1.
- Se eliminan características no representativas para los modelos(year).
---

## Características principales

- 🐧 ETL completo para preparación de datos  
- 🤖 Entrenamiento de 3 modelos de ML diferentes  
- 🚀 API REST con FastAPI para predicciones  
- 📦 Dockerización con compose para despliegue multi-servicio
- 🔄 Sistema dinámico de selección de modelos  
- 📊 JupyterLab integrado para experimentación
- 🔍 Interfaz de documentación automática

---

## Instalación y configuración

Clonar el repositorio:

```bash
git clone https://github.com/JonatanGallo/MLOps---Talleres.git
cd penguins-taller-2
```

## Ejecución con Docker Compose

Construcción de la imagen Docker:

```bash
docker-compose up
```

## Servicios desplegados: 

- API de Predicción: http://localhost:8989
- JupyterLab: http://localhost:8888 (token: secret)

## Entrenamiento de modelos
Acceder a JupyterLab en http://localhost:8888 y abrir el notebook TrainModels.ipynb:
```python
# Ejemplo de entrenamiento desde el notebook
from etl import get_data
from model import Model
from models import ModelType

X, y = get_data()
model = Model(ModelType.RANDOM_FOREST)
model.train(X, y)
model.save('/models/model_random_forest.pkl')
```
---

## Uso de la API


### 1. Acceder a la interfaz de documentación

Abrir en el navegador:  
[http://localhost:8989/docs](http://localhost:8989/docs)

---

## Uso de la API

### Listar modelos disponibles

```bash
curl http://localhost:8989/models
```

Respuesta:

```json
{
  "available_models": [
    "random_forest",
    "svm",
    "neural_network"
  ]
}
```
### Cargar un modelo específico

```bash
curl http://localhost:8989/load_model/random_forest
```

### Predecir con selección dinámica de modelo (POST)

```bash
curl -X POST "http://localhost:8989/predict/random_forest" \
-H "Content-Type: application/json" \
-d '{
  "island": "Biscoe",
  "bill_length_mm": 50.0,
  "bill_depth_mm": 16.3,
  "flipper_length_mm": 230,
  "body_mass_g": 5700,
  "sex": "male",
  "year": 2007
}'
```

Respuesta:

```json
{
  "model_used": "random_forest",
  "prediction": "Gentoo"
}
```

---

## Estructura del proyecto

```
penguins-taller-2/
├── api/                         # Servicio de API de predicción
│   ├── main.py                  # Aplicación FastAPI principal
│   ├── model.py
│   ├── pyproject.toml
│   ├── skaler.pkl
│   ├── .python-version
│   ├── penguins.py             # Definición de especies de pingüinos
│   ├── uv.lock
│   ├── ModelService.py          # Servicio para manejar modelos
│   ├── requirements.txt         # Dependencias del API
│   ├── Dockerfile              # Dockerfile para el servicio de API
│   └── dto/                        # Objetos de transferencia de datos
│       └── model_prediction_request.py
│       └── normalized_request.py
├── training-app/               # Servicio de entrenamiento
│   ├── TrainModels.ipynb       # Notebook de entrenamiento
│   ├── etl.py                  # Extracción, transformación y carga
│   ├── model.py                # Definición de modelos de ML
│   ├── models.py               # Tipos de modelos disponibles
│   ├── pyproject.toml          # Configuración de dependencias
│   ├── uv.lock
│   ├── .gitignore
│   ├── README.md
│   ├── skaler.pkl
│   ├── .python-version
│   └── Dockerfile              # Dockerfile para el servicio de entrenamiento
├── models/                     # Modelos entrenados (volumen compartido)
│   ├── model_neural_network.pkl
│   ├── model_random_forest.pkl
│   └── model_svm.pkl 
└── README.md                   
```

---

## Modelos implementados

- Random Forest - Clasificador de bosques aleatorios  
- SVM - Máquinas de Soporte Vectorial  
- Neural Network - Red neuronal multicapa  

---


## Especies de pingüinos soportadas

| Especie    | Valor numérico | Descripción             |
|------------|----------------|-------------------------|
| Adelie     | 0              | Pingüinos Adelia        |
| Chinstrap  | 1              | Pingüinos de barbijo    |
| Gentoo     | 2              | Pingüinos papúa         |

---
## Desarrollo

Para desarrollo y debugging:

- Ejecutar docker-compose up para iniciar todos los servicios
- Acceder a JupyterLab en http://localhost:8888
- Modificar notebooks o código en el directorio training-app
- Los cambios se reflejarán automáticamente en el contenedor
- Los modelos entrenados se guardan en el directorio models/ compartido

---
## Notas de la versión

- Los modelos se persisten en el volumen compartido models/
- JupyterLab incluye el token de autenticación secret
- La API se recarga automáticamente durante el desarrollo
- Los modelos están disponibles inmediatamente después del entrenamiento
---

## Demo

[Video de demostración] https://livejaverianaedu-my.sharepoint.com/:v:/g/personal/torrespjc_javeriana_edu_co/ESykJVbzALhHnnBm-mcHQeUB_Btx7Po4SFejXkjhKh9QmA

---
