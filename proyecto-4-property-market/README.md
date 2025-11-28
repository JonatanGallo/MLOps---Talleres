# Proyecto 4 - Property Market MLOps

Sistema completo de Machine Learning Operations para predicción de precios en el mercado inmobiliario.

## Video de documentación del proyecto

https://youtu.be/rtFYMDGH0bE

## 📋 Descripción

Este proyecto implementa un pipeline completo de MLOps que incluye:
- **Entrenamiento de modelos** de machine learning para predecir precios de propiedades
- **API de predicción** con FastAPI
- **Orquestación** con Apache Airflow
- **Tracking de modelos** con MLflow
- **Almacenamiento** con MinIO (S3 compatible)
- **Interfaz web** con Gradio
- **Monitoreo** con Prometheus y Grafana

## 🏗️ Arquitectura

El proyecto está organizado en los siguientes componentes:

```
proyecto-4-property-market/
├── api/                    # API de predicción (FastAPI)
├── training_app/          # Aplicación de entrenamiento de modelos
├── webApp/                # Interfaz web con Gradio
├── dags/                  # DAGs de Airflow
├── airflow/               # Configuración de Airflow
├── mlflow/                # Configuración de MLflow
├── locust/                # Tests de carga
└── manifests/             # Manifiestos de Kubernetes
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker y Docker Compose
- Python 3.9+ (para desarrollo local)
- UV (gestor de paquetes Python)

### Ejecutar el proyecto completo

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd proyecto-4-property-market
   ```

2. **Iniciar todos los servicios con Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Acceder a los servicios**
   - **Airflow**: http://localhost:8080 (usuario: `airflow`, password: `airflow`)
   - **MLflow**: http://localhost:8003
   - **MinIO Console**: http://localhost:8001 (usuario: `admin`, password: `supersecret`)
   - **API de Predicción**: http://localhost:8012
   - **Gradio Web App**: http://localhost:8014
   - **Jupyter Notebook**: http://localhost:8006

## 📦 Componentes Principales

### Training App (`training_app/`)

Aplicación para entrenar modelos de machine learning. Incluye:
- ETL de datos (`etl.py`)
- Entrenamiento de modelos (`train.py`)
- Controlador de datos (`dataController.py`)
- Integración con MLflow para tracking

**Ejecutar entrenamiento localmente:**
```bash
cd training_app
uv sync
python train.py
```

### API de Predicción (`api/`)

API REST construida con FastAPI que expone endpoints para:
- `/predict` - Realizar predicciones
- `/models` - Listar modelos disponibles
- `/model` - Cargar modelo desde MLflow
- `/upload_preprocessor` - Subir preprocesador
- `/metrics` - Métricas de Prometheus

**Ejecutar API localmente:**
```bash
cd api
uv sync
uvicorn main:app --reload
```

### Airflow DAGs (`dags/`)

DAGs para orquestar el pipeline de ML:
- Entrenamiento automático de modelos
- Carga de datos
- Validación de modelos

### Web App (`webApp/`)

Interfaz web con Gradio para realizar predicciones de forma interactiva.

## 🔧 Configuración

### Variables de Entorno

Los servicios se configuran mediante variables de entorno en `docker-compose.yml`:

- `MLFLOW_TRACKING_URI`: URI del servidor MLflow
- `MODEL_NAME`: Nombre del modelo
- `MODEL_STAGE`: Etapa del modelo (prod, staging, etc.)
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`: Configuración de base de datos

### Bases de Datos

El proyecto utiliza dos bases de datos MySQL:
- **MySQL Training** (puerto 8005): Para datos de entrenamiento
- **MySQL MLflow** (puerto 8004): Para el tracking de MLflow

## 📊 Monitoreo

- **Prometheus**: Métricas del sistema en `/metrics`
- **Grafana**: Dashboards de visualización (ver `manifests/grafana/`)

## 🧪 Testing

### Tests de Carga con Locust

Locust se utiliza para realizar pruebas de carga y rendimiento en la API de predicción.

**Configuración:**
- **Master**: Coordina los workers y proporciona la interfaz web (puerto 8010)
- **Workers**: Ejecutan las pruebas de carga (escalable)
- **Target**: API de predicción en `http://10.43.100.102:8013`

**Ejecutar tests de carga:**

```bash
cd locust
docker-compose up
```

**Escalar workers:**
```bash
docker-compose up --scale locust-worker=4
```

**Acceder a la interfaz:**
- Locust UI: http://localhost:8010

**Características:**
- Simulación de usuarios concurrentes
- Tiempo de espera configurable entre tareas (1-3 segundos)
- Pool de conexiones optimizado (800 conexiones por worker)
- Monitoreo en tiempo real de métricas de rendimiento

## 🚀 CI/CD

### GitHub Actions

El proyecto utiliza GitHub Actions para automatizar el proceso de build y despliegue de la API.

**Workflow:** `.github/workflows/github-actions.yml`

**Proceso automatizado:**
1. **Trigger**: Se ejecuta en cada push a la rama `main` (ignora cambios en `manifests/**`)
2. **Build**: Construye la imagen Docker de la API
3. **Tag**: Genera un tag único con formato `YYYYMMDD-<commit-hash>`
4. **Push**: Sube la imagen a DockerHub (`jdromero9402/mlops_inference_market`)
5. **Actualización**: Actualiza el manifiesto de Kubernetes con la nueva imagen
6. **Commit**: Hace commit automático de los cambios (con `[skip ci]` para evitar loops)

**Secrets requeridos:**
- `DOCKERHUB_TOKEN`: Token de autenticación para DockerHub

**Ejemplo de tag generado:**
```
20251128-f4fc508
```

### Argo CD

Argo CD se utiliza para el despliegue continuo (CD) en Kubernetes mediante GitOps.

**Aplicación Argo CD:** `app-mlops-market.yaml`

**Configuración:**
- **Nombre**: `mlops-market-api`
- **Repositorio**: `https://github.com/JonatanGallo/MLOps---Talleres.git`
- **Rama**: `main`
- **Path**: `proyecto-4-property-market/manifests/api`
- **Namespace**: `default`

**Características:**
- **Sync automático**: Sincronización automática cuando hay cambios en el repositorio
- **Self-healing**: Argo CD detecta y corrige desviaciones del estado deseado
- **Prune**: Elimina recursos que ya no están en el repositorio

**Aplicar la aplicación:**
```bash
kubectl apply -f app-mlops-market.yaml
```

**Flujo completo CI/CD:**
1. Push a `main` → GitHub Actions construye y publica imagen
2. GitHub Actions actualiza el manifiesto de Kubernetes
3. Argo CD detecta el cambio en el repositorio
4. Argo CD despliega automáticamente la nueva versión en Kubernetes

## 📝 Notas

- Los modelos entrenados se guardan en la carpeta `models/`
- MLflow almacena los artefactos en MinIO
- Los logs de Airflow se almacenan en volúmenes Docker

## 🔗 Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Apache Airflow Documentation](https://airflow.apache.org/)
- [MLflow Documentation](https://mlflow.org/)
- [Gradio Documentation](https://www.gradio.app/)

