# 🐧 MLOps - Talleres Clasificador de Pingüinos con MLflow

Este repositorio contiene el código usado para entrenar y desplegar un modelo de predicción de clases de pingüinos basado en el dataset disponible en: [palmerpenguins](https://pypi.org/project/palmerpenguins/). El proyecto abarca desde la preparación de datos y el entrenamiento de modelos hasta el despliegue de una API REST para realizar predicciones, utilizando MLflow para el seguimiento de experimentos y gestión del ciclo de vida de modelos.

---

## Arquitectura de Servicios

- **Servicio de Predicción**: API REST con FastAPI para inferencia de modelos registrados en MLflow
- **MLflow Tracking Server**: Servidor de seguimiento de experimentos y registro de modelos
- **MinIO**: Almacenamiento de artefactos S3-compatible para modelos y datos
- **Jupyter Lab**: Entorno de experimentación interactivo con integración MLflow
- **Base de Datos MySQL**: Almacenamiento de metadatos de MLflow y datos de entrenamiento
- **Servicio de Entrenamiento**: Aplicación Python para entrenamiento automatizado con logging a MLflow

### Componentes de MLflow

- **MLflow Tracking Server**: Servidor central para tracking de experimentos (puerto 8003)
- **MLflow Model Registry**: Registro centralizado de modelos con versionado y staging
- **MinIO S3 Storage**: Almacenamiento de artefactos de modelos y datasets (puerto 8000/8001)
- **MySQL Backend**: Base de datos para metadatos de experimentos y modelos
- **Jupyter Integration**: Notebooks con tracking automático de experimentos

## Entrenamiento de Modelos

### Workflow con MLflow Tracking

El entrenamiento de modelos utiliza MLflow para el seguimiento completo del ciclo de vida:

1. **Preparación de datos**: ETL con logging de datasets y métricas de calidad
2. **Entrenamiento de modelos**: Múltiples algoritmos con tracking automático de parámetros y métricas
3. **Registro de modelos**: Almacenamiento automático en MLflow Model Registry
4. **Versionado de modelos**: Control de versiones con staging (dev, staging, prod)

### Características del Entrenamiento

- **Tracking automático**: Todos los experimentos se registran automáticamente en MLflow
- **Soporte para múltiples tipos de modelos** (Random Forest, SVM, Neural Network)
- **Gestión de artefactos**: Modelos y datasets almacenados en MinIO S3
- **Comparación de experimentos** a través de la interfaz web de MLflow
- **Notebook interactivo** (Training.ipynb) con integración MLflow completa
- **Model Registry**: Registro centralizado con promoción de modelos entre stages

## Mejoras en la API

- **Integración con MLflow Model Registry**: Carga dinámica de modelos desde el registro
- **Endpoint para listado de modelos disponibles**: Consulta modelos registrados en MLflow
- **Sistema de normalización de datos de entrada**: Preprocesamiento automático
- **Manejo de errores mejorado**: Gestión robusta de excepciones
- **Documentación interactiva automática**: OpenAPI/Swagger integrado
- **Selección de modelo por stage**: Soporte para dev, staging y prod environments

## Creación del modelo

- El dataset se carga usando el método `load_penguins` expuesto en la librería del proyecto `palmerpenguins`.  
- Se convierten columnas categóricas a numéricas usando One-shot Encoding.
- Se hace una escala para mantener la desviación estandar por debajo de 1.
- Se eliminan características no representativas para los modelos(year).
---

## Características principales

- 🐧 ETL completo para preparación de datos con logging en MLflow
- 🤖 Entrenamiento de 3 modelos de ML diferentes con tracking automático
- 🚀 API REST con FastAPI para predicciones desde MLflow Model Registry
- 📦 Dockerización con compose para despliegue multi-servicio
- 🔄 Sistema dinámico de selección de modelos por stage (dev/staging/prod)
- 📊 JupyterLab integrado con MLflow tracking automático
- 🔍 Interfaz de documentación automática con OpenAPI
- 📈 **MLflow Tracking Server** para seguimiento completo de experimentos
- 🗄️ **MinIO S3 Storage** para almacenamiento de artefactos de modelos
- 📋 **MLflow Model Registry** para gestión del ciclo de vida de modelos
- 🔍 **Interfaz web MLflow** para comparación y análisis de experimentos

---

## Instalación y configuración

Clonar el repositorio:

```bash
git clone https://github.com/JonatanGallo/MLOps---Talleres.git
cd penguins-taller-4-mlflow
```

## Ejecución con Docker Compose

Construcción y ejecución de todos los servicios:

```bash
docker-compose up
```

## Servicios desplegados: 

- **API de Predicción**: http://localhost:8012
- **MLflow Tracking Server**: http://localhost:8003
- **MinIO Console**: http://localhost:8001 (usuario: admin, contraseña: supersecret)
- **MinIO API**: http://localhost:8000
- **Jupyter Lab**: http://localhost:8005 (token: cualquiera)
- **MySQL MLflow**: localhost:8004 (usuario: user, contraseña: password, base de datos: mlflow_db)
- **MySQL Training**: localhost:8006 (usuario: user, contraseña: password, base de datos: training)

## Entrenamiento de modelos

### Entrenamiento con MLflow Tracking

1. **Acceder a Jupyter Lab**: Abrir http://localhost:8005 en el navegador (token: cualquiera)
2. **Abrir notebook**: Navegar a `workspace/Training.ipynb`
3. **Ejecutar entrenamiento**: Ejecutar las celdas del notebook
4. **Monitorear en MLflow**: Ver experimentos en http://localhost:8003

### Entrenamiento Programático

Para entrenamiento desde código Python:

```python
# Ejemplo de entrenamiento con MLflow tracking
import mlflow
from etl import get_clean_data
from model import Model
from models import ModelType

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:8003")

# Entrenar modelo con tracking automático
X, y = get_clean_data()
with mlflow.start_run():
    model = Model(ModelType.RANDOM_FOREST)
    model.train(X, y)
    # El modelo se registra automáticamente en MLflow
```
---

## Uso de la API


### 1. Acceder a la interfaz de documentación

Abrir en el navegador:  
[http://localhost:8012/docs](http://localhost:8012/docs)

---

## Uso de la API

### Listar modelos disponibles (desde MLflow Registry)

```bash
curl http://localhost:8012/models
```

Respuesta:

```json
{
  "available_models": [
    "penguin_random_forest",
    "penguin_svm", 
    "penguin_neural_network"
  ]
}
```
### Cargar un modelo específico

```bash
curl http://localhost:8012/load_model/random_forest
```

### Predecir con modelo desde MLflow Registry (POST)

```bash
curl -X POST "http://localhost:8012/predict/penguin_random_forest" \
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
  "model_used": "penguin_random_forest",
  "prediction": "Gentoo"
}
```

### Predecir con modelo por stage

```bash
# Usar modelo en stage 'prod' (por defecto)
curl -X POST "http://localhost:8012/predict/penguin_random_forest" \
-H "Content-Type: application/json" \
-d '{...}'

# Para usar un stage específico, configurar la variable MODEL_STAGE
# en el docker-compose.yaml: MODEL_STAGE=staging
```

---

## Estructura del proyecto

```
penguins-taller-4-mlflow/
├── api/                         # Servicio de API de predicción
│   ├── main.py                  # Aplicación FastAPI con integración MLflow
│   ├── model.py                 # Clases de modelos ML
│   ├── pyproject.toml          # Configuración de dependencias
│   ├── scaler.pkl              # Scaler para normalización
│   ├── penguins.py             # Definición de especies de pingüinos
│   ├── uv.lock                 # Lock file de dependencias
│   ├── ModelService.py          # Servicio para manejar modelos
│   ├── Dockerfile              # Dockerfile para el servicio de API
│   └── dto/                    # Objetos de transferencia de datos
│       ├── model_prediction_request.py
│       └── normalized_request.py
├── training_app/               # Servicio de entrenamiento con MLflow
│   ├── Training.ipynb          # Notebook de entrenamiento con MLflow tracking
│   ├── etl.py                  # Extracción, transformación y carga
│   ├── model.py                # Definición de modelos de ML
│   ├── models.py               # Tipos de modelos disponibles
│   ├── train.py                # Script de entrenamiento con MLflow logging
│   ├── db.py                   # Conexión a base de datos
│   ├── pyproject.toml          # Configuración de dependencias
│   ├── uv.lock                 # Lock file de dependencias
│   ├── Dockerfile              # Dockerfile para Jupyter con MLflow
│   ├── requirements.txt        # Dependencias adicionales
│   ├── scaler.pkl              # Scaler para normalización
│   └── raw_data.csv            # Datos de entrenamiento
├── mlflow/                     # Configuración de MLflow
│   └── Dockerfile              # Dockerfile personalizado de MLflow
├── mlflowdb/                   # Directorio para base de datos MLflow
├── notebooks/                  # Notebooks adicionales
│   └── Untitled.ipynb         # Notebook de ejemplo
├── docker-compose.yaml         # Orquestación de servicios
└── README.md                   
```

---

## Workflow de MLflow

El flujo de trabajo con MLflow automatiza el seguimiento completo del ciclo de vida de modelos:

### Componentes del Workflow

1. **Experiment Tracking** 📊
   - Registro automático de parámetros, métricas y artefactos
   - Comparación de múltiples experimentos y runs
   - Visualización de métricas en tiempo real

2. **Model Registry** 📋
   - Registro centralizado de modelos entrenados
   - Versionado automático de modelos
   - Gestión de stages (dev, staging, prod)

3. **Artifact Storage** 💾
   - Almacenamiento de modelos en MinIO S3
   - Persistencia de datasets y scalers
   - Trazabilidad completa de artefactos

4. **Model Serving** 🚀
   - Carga dinámica de modelos desde el registry
   - Selección de modelos por stage
   - API REST para inferencia

### Flujo de Ejecución

```
Data Preparation → Model Training → MLflow Logging → Model Registry → API Serving
```

### Características del Tracking

- **Automatic Logging**: Parámetros, métricas y modelos se registran automáticamente
- **Experiment Comparison**: Interfaz web para comparar múltiples experimentos
- **Model Versioning**: Control de versiones automático en el registry
- **Stage Management**: Promoción de modelos entre dev, staging y prod

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

- **Ejecutar servicios**: `docker-compose up` para iniciar todos los servicios
- **Monitorear MLflow**: Acceder a http://localhost:8003 para ver experimentos y modelos
- **Jupyter Lab**: Usar http://localhost:8005 para experimentación interactiva
- **MinIO Console**: Acceder a http://localhost:8001 para gestionar artefactos
- **Ver experimentos**: Comparar runs y métricas en la interfaz web de MLflow
- **Modificar código**: Los cambios en `training_app/` se reflejan automáticamente
- **Modelos**: Los modelos se almacenan automáticamente en MinIO S3 y MLflow Registry
- **Base de datos**: Conectar a MySQL MLflow (8004) y Training (8006) para inspeccionar datos

---
## Notas de la versión

- **MLflow Tracking**: Seguimiento completo de experimentos con métricas y parámetros automáticos
- **Model Registry**: Registro centralizado con versionado y gestión de stages
- **MinIO S3**: Almacenamiento escalable de artefactos de modelos y datasets
- **API**: Integración directa con MLflow Model Registry para serving dinámico
- **Jupyter Integration**: Notebooks con tracking automático de experimentos MLflow
- **MySQL Backend**: Persistencia robusta de metadatos y datos de entrenamiento
- **Container Orchestration**: Docker Compose para despliegue multi-servicio

---

## Acceso a Servicios Web

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **API de Predicción** | http://10.43.100.99:8012 | - |
| **MLflow UI** | http://10.43.100.99:8003 | - |
| **Jupyter Lab** | http://10.43.100.99:8005 | Token: `cualquiera` |
| **MinIO Console** | http://10.43.100.99:8001 | admin / supersecret |
| **API Docs** | http://10.43.100.99:8012/docs | - |

---

## MLflow Model Registry

Para gestionar modelos en diferentes stages:

1. **Registrar modelo**: Los modelos se registran automáticamente durante el entrenamiento
2. **Promocionar modelo**: Usar la interfaz MLflow para mover modelos entre stages
3. **Usar en API**: La API carga automáticamente modelos del stage configurado (prod por defecto)
