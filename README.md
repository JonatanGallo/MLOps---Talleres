# MLOps---TalleresClasificador de pinguinos
Este repositorio contiene el código usado para entrenar y desplegar un modelo de predicción de clases de pinguinos basado en el dataset disponible en: https://pypi.org/project/palmerpenguins/

Creación del modelo
El dataset se carga usando el método load_penguins expuesto en la libreria del proyecto palmerpenguins.
Se convierten columnas categóricas a numéricas usando One-Hot EncodingCódigo de los talleres del curso de MLOps de la PUJ-MINTA-MISA

Características principales
🐧 ETL completo para preparación de datos

🤖 Entrenamiento de 4 modelos de ML diferentes

🚀 API REST con FastAPI para predicciones

📦 Contenedorización lista para despliegue

🔄 Sistema dinámico de selección de modelos

Instalación y configuración
Clonar el repositorio:

bash
git clone https://github.com/tu-usuario/penguins-taller-1.git
cd penguins-taller-1

Instalar dependencias:

bash
pip install -r requirements.txt
Ejecución del proyecto
1. Entrenamiento de modelos
bash
python Train.py
Los modelos entrenados se guardarán en la carpeta models/.

2. Iniciar la API
bash
uvicorn server:app --reload --port 8989
3. Acceder a la interfaz de documentación
Abrir en el navegador: http://localhost:8989/docs

Uso de la API
Listar modelos disponibles
bash
curl http://localhost:8989/models
Respuesta:

json
{
  "available_models": [
    "random_forest",
    "svm",
    "neural_network",
    "linear_regression"
  ]
}
Predecir con un modelo específico (GET)
bash
curl "http://localhost:8989/predict_random_forest?bill_length_mm=39.2&bill_depth_mm=18.8&flipper_length_mm=196&body_mass_g=4000&island_Dream=0&island_Torgersen=1&sex_Male=1&sex_Unknown=0"

Respuesta:

json
{
  "model_used": "random_forest",
  "prediction": "Gentoo"
}
Predecir con selección dinámica de modelo (POST)
bash
curl -X POST "http://localhost:8989/predict/random_forest" \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.2,
    "bill_depth_mm": 18.8,
    "flipper_length_mm": 196,
    "body_mass_g": 4000,
    "island_Dream": 0,
    "island_Torgersen": 1,
    "sex_Male": 1,
    "sex_Unknown": 0
  }'
Respuesta:

json
{
  "model_used": "random_forest",
  "prediction": "Gentoo"
}
Estructura del proyecto
text
penguins-taller-1/
├── models/                   # Modelos entrenados (generados)
├── dto/                      # Objetos de transferencia de datos
│   └── model_prediction_request.py
├── etl.py                    # Extracción, transformación y carga
├── model.py                  # Definición de modelos de ML
├── models.py                 # Tipos de modelos disponibles
├── ModelService.py           # Servicio para manejar modelos
├── penguins.py               # Definición de especies de pingüinos
├── requirements.txt          # Dependencias
├── server.py                 # API con FastAPI
├── test_api.py               # Pruebas de la API
└── Train.py                  # Script de entrenamiento
Modelos implementados
Random Forest - Clasificador de bosques aleatorios

SVM - Máquinas de Soporte Vectorial

Neural Network - Red neuronal multicapa

Linear Regression - Regresión lineal (para propósitos comparativos)

Construcción de la imagen Docker
bash
docker build -t penguins-api .
Ejecución en Docker
bash
docker run -p 8989:8989 penguins-api
Especies de pingüinos soportadas
El sistema puede predecir las siguientes especies:

Especie 	Valor numérico	Descripción
Adelie	          0	        Pingüinos Adelia
Chinstrap	      1     	Pingüinos de barbijo
Gentoo	          2	        Pingüinos papúa

Contribución
Si deseas contribuir a este proyecto:

Haz un fork del repositorio

Crea una nueva rama (git checkout -b feature/nueva-funcionalidad)

Realiza tus cambios y haz commit (git commit -am 'Añade nueva funcionalidad')

Haz push a la rama (git push origin feature/nueva-funcionalidad)

Abre un Pull Request