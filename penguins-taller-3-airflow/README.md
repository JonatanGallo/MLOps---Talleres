# 🐧 MLOps - Talleres Clasificador de Pingüinos

Este repositorio contiene el código usado para entrenar y desplegar un modelo de predicción de clases de pingüinos basado en el dataset disponible en: [palmerpenguins](https://pypi.org/project/palmerpenguins/). El proyecto abarca desde la preparación de datos y el entrenamiento de modelos hasta el desplieque de una API REST para realizar predicciones.

---

## Arquitectura de Servicios

- **Servicio de Predicción**: API REST con FastAPI para inferencia de modelos
- **Orquestación de Workflows**: Apache Airflow para automatización y programación de tareas de ML
- **Servicio de Entrenamiento**: Aplicación Python para entrenamiento automatizado de modelos
- **Base de Datos**: MySQL para almacenamiento de datos de entrenamiento
- **Volúmenes Compartidos**: Sistema de archivos compartido para modelos entrenados y logs

### Componentes de Airflow

- **Airflow Webserver**: Interfaz web para monitoreo y gestión de DAGs (puerto 8080)
- **Airflow Scheduler**: Planificador de tareas que ejecuta los DAGs según su programación
- **Airflow Worker**: Ejecutor de tareas usando Celery
- **Airflow Triggerer**: Manejo de sensores y triggers asíncronos
- **PostgreSQL**: Base de datos de metadatos de Airflow
- **Redis**: Broker de mensajes para Celery

## Entrenamiento de Modelos

### Workflow Automatizado con Airflow

El entrenamiento de modelos se ejecuta mediante un DAG (Directed Acyclic Graph) de Airflow que automatiza todo el proceso:

1. **clear_raw_data**: Limpia datos anteriores de la base de datos
2. **store_raw_data**: Descarga y almacena datos frescos del dataset de pingüinos
3. **get_raw_data**: Extrae y procesa los datos para entrenamiento
4. **save_all_models**: Entrena y guarda todos los modelos (Random Forest, SVM, Neural Network)

### Características del Entrenamiento

- **Automatización completa**: El DAG se ejecuta según la programación definida
- **Soporte para múltiples tipos de modelos** con selección dinámica
- **Persistencia automática** de modelos entrenados en el volumen compartido
- **Monitoreo en tiempo real** a través de la interfaz web de Airflow
- **Notebook interactivo** (TrainModels.ipynb) disponible para experimentación manual

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
- ⚡ **Orquestación con Apache Airflow** para automatización de workflows
- 🗄️ **Base de datos MySQL** para persistencia de datos de entrenamiento
- 📈 **Monitoreo en tiempo real** de tareas de ML

---

## Instalación y configuración

Clonar el repositorio:

```bash
git clone https://github.com/JonatanGallo/MLOps---Talleres.git
cd penguins-taller-3-airflow
```

## Ejecución con Docker Compose

Construcción y ejecución de todos los servicios:

```bash
docker-compose up
```

## Servicios desplegados: 

- **API de Predicción**: http://localhost:8012
- **Airflow Webserver**: http://localhost:8080 (usuario: airflow, contraseña: airflow)
- **MySQL Database**: localhost:3306 (usuario: user, contraseña: password, base de datos: training)

## Entrenamiento de modelos

### Entrenamiento Automatizado con Airflow

1. **Acceder a Airflow**: Abrir http://localhost:8080 en el navegador
2. **Credenciales**: Usuario: `airflow`, Contraseña: `airflow`
3. **Ejecutar DAG**: Buscar el DAG `training_dag` y activarlo
4. **Monitorear**: Ver el progreso de las tareas en tiempo real

### Entrenamiento Manual (Opcional)

Para experimentación manual, se puede usar el notebook disponible:

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
[http://localhost:8012/docs](http://localhost:8012/docs)

---

## Uso de la API

### Listar modelos disponibles

```bash
curl http://localhost:8012/models
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
curl http://localhost:8012/load_model/random_forest
```

### Predecir con selección dinámica de modelo (POST)

```bash
curl -X POST "http://localhost:8012/predict/random_forest" \
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
penguins-taller-3-airflow/
├── api/                         # Servicio de API de predicción
│   ├── main.py                  # Aplicación FastAPI principal
│   ├── model.py
│   ├── pyproject.toml
│   ├── scaler.pkl
│   ├── penguins.py             # Definición de especies de pingüinos
│   ├── uv.lock
│   ├── ModelService.py          # Servicio para manejar modelos
│   ├── Dockerfile              # Dockerfile para el servicio de API
│   └── dto/                    # Objetos de transferencia de datos
│       ├── model_prediction_request.py
│       └── normalized_request.py
├── training-app/               # Servicio de entrenamiento
│   ├── TrainModels.ipynb       # Notebook de entrenamiento
│   ├── etl.py                  # Extracción, transformación y carga
│   ├── model.py                # Definición de modelos de ML
│   ├── models.py               # Tipos de modelos disponibles
│   ├── train.py                # Script de entrenamiento automatizado
│   ├── db.py                   # Conexión a base de datos
│   ├── pyproject.toml          # Configuración de dependencias
│   ├── uv.lock
│   ├── Dockerfile              # Dockerfile para el servicio de entrenamiento
│   └── raw_data.csv            # Datos de entrenamiento
├── dags/                       # DAGs de Airflow
│   ├── training.py             # DAG principal de entrenamiento
│   └── training_app/           # Módulos compartidos con el DAG
├── airflow/                    # Configuración de Airflow
│   ├── Dockerfile              # Dockerfile personalizado de Airflow
│   └── requirements.txt        # Dependencias adicionales
├── models/                     # Modelos entrenados (volumen compartido)
│   ├── model_neural_network.pkl
│   ├── model_random_forest.pkl
│   └── model_svm.pkl
├── logs/                       # Logs de Airflow
├── plugins/                    # Plugins personalizados de Airflow
├── docker-compose.yml          # Orquestación de servicios
└── README.md                   
```

---

## Workflow del DAG de Entrenamiento

El DAG `training_dag` automatiza el proceso completo de entrenamiento de modelos:

### Tareas del DAG

1. **clear_raw_data** 🗑️
   - Limpia datos anteriores de la base de datos MySQL
   - Prepara el entorno para nuevos datos

2. **store_raw_data** 📥
   - Descarga el dataset de pingüinos desde la fuente
   - Almacena los datos en la base de datos MySQL

3. **get_raw_data** 🔄
   - Extrae datos de la base de datos
   - Aplica transformaciones ETL (One-hot encoding, escalado)
   - Prepara los datos para entrenamiento

4. **save_all_models** 🤖
   - Entrena los 3 modelos: Random Forest, SVM, Neural Network
   - Guarda los modelos entrenados en el volumen compartido
   - Persiste el scaler para normalización

### Flujo de Ejecución

```
clear_raw_data → store_raw_data → get_raw_data → save_all_models
```

### Programación

- **Schedule**: `@once` (ejecución manual)
- **Start Date**: 2023-05-01
- **Dependencias**: Cada tarea depende de la anterior

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
- **Monitorear Airflow**: Acceder a http://localhost:8080 (usuario: airflow, contraseña: airflow)
- **Ejecutar DAGs**: Activar y ejecutar el DAG `training_dag` desde la interfaz web
- **Ver logs**: Monitorear el progreso de las tareas en tiempo real
- **Modificar código**: Los cambios en `training-app/` se reflejan automáticamente
- **Modelos**: Los modelos entrenados se guardan en el directorio `models/` compartido
- **Base de datos**: Conectar a MySQL en localhost:3306 para inspeccionar datos

---
## Notas de la versión

- **Modelos**: Se persisten en el volumen compartido `models/`
- **Airflow**: Incluye interfaz web completa para monitoreo y gestión de DAGs
- **Base de datos**: MySQL integrado para persistencia de datos de entrenamiento
- **API**: Se recarga automáticamente durante el desarrollo
- **Automatización**: Los modelos se entrenan automáticamente mediante el DAG de Airflow
- **Monitoreo**: Logs detallados disponibles en la interfaz de Airflow

---

## Diagrama de Arquitectura
![Alt text](./arquitecturaAirflow.drawio.svg)
