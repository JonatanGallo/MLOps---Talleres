# Palmer Penguins Model Training App

Este directorio contiene la aplicación de entrenamiento de modelos para el dataset de Palmer Penguins. Aquí se encuentran los scripts y módulos necesarios para preparar los datos, entrenar diferentes modelos de machine learning y guardar los modelos entrenados.

## Estructura de archivos

- `etl.py`: Limpieza, transformación y escalado de los datos.
- `model.py`: Definición de la clase base para los modelos y lógica de entrenamiento.
- `train_model.py`: Script principal para entrenar y guardar los modelos.
- `TrainModels.ipynb`: Jupyter Notebook interactivo para exploración y entrenamiento de modelos.
- `models/`: Carpeta donde se guardan los modelos entrenados (`.pkl`).

---

## Cómo ejecutar el entrenamiento localmente

### Pre requisitos
  Docker
  UV

### 1. Usando el Jupyter Notebook

1. **Instala las dependencias**  
   uv sync
   ```

2. **Inicia Jupyter Notebook**  
   Desde este directorio, ejecuta:
   ```
   jupyter notebook
   ```
   Luego abre el archivo `TrainModels.ipynb` en tu navegador y sigue las celdas para explorar, limpiar datos y entrenar los modelos.

---

### 2. Usando el script Python

1. **Instala las dependencias**  
   ```
   pip install -r requirements.txt
   ```

2. **Ejecuta el entrenamiento**  
   ```
   python train_model.py
   ```
   Esto entrenará los modelos (Random Forest, SVM, Red Neuronal, Regresión Lineal) y los guardará en la carpeta `models/`.

---

## Cómo ejecutar el entrenamiento usando Docker

1. **Asegúrate de tener Docker instalado**  
   [Descargar Docker](https://www.docker.com/get-started/)

2. **Construye y ejecuta el contenedor de entrenamiento**  
   Desde el directorio raíz del proyecto, ejecuta:
   ```
   docker-compose run --rm model_builder
   ```
   Esto ejecutará el script `train_model.py` dentro de un contenedor, entrenando los modelos y guardándolos en la carpeta `models/` (mapeada como volumen).

---

## Notas

- El preprocesamiento de datos incluye imputación de valores nulos, codificación de variables categóricas y escalado de variables numéricas.
- El script guarda el scaler usado en el preprocesamiento como `scaler.pkl` para su uso posterior en la inferencia.
- Puedes modificar los hiperparámetros de los modelos en `train_model.py` o `model.py`.

---

## Referencias

- [palmerpenguins dataset](https://allisonhorst.github.io/palmerpenguins/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
