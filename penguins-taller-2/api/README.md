# Palmer Penguins API
Esta API permite predecir el tipo de pingüino (Adelie, Chinstrap, o Gentoo) basándose en sus características físicas, utilizando modelos de Machine Learning.

---
## Características
- Predicción de Tipo de Pingüino: Utiliza modelos de Machine Learning para clasificar pingüinos.
- Carga Dinámica de Modelos: Permite cargar diferentes modelos de ML en tiempo de ejecución.
- Normalización de Datos: Preprocesa automáticamente los datos de entrada para que coincidan con el formato de entrenamiento del modelo.
- Listado de Modelos Disponibles: Permite consultar qué modelos de predicción están disponibles.
## Tecnologías Utilizadas
- FastAPI: Framework web para construir la API.
- Pydantic: Para la validación y serialización de datos.
- Scikit-learn: Para la implementación de modelos de Machine Learning.
- Pandas: Para la manipulación de datos.
- Joblib: Para la carga y guardado de modelos y escaladores.
## Configuración del Entorno
Requisitos Previos
- Python 3.10 o superior.
- uvicorn para ejecutar la aplicación.

