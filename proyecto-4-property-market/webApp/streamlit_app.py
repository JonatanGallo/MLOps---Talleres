# streamlit_app.py
import os
import json
import requests
import streamlit as st

# URL base de tu API de FastAPI.
# En local:  http://localhost:8000
# En Kubernetes/Docker: algo tipo http://property-api:8000
API_URL = os.getenv("PREDICT_API_URL") + "/predict"
PORT = int(os.getenv("WEBAPP_PORT"))


st.set_page_config(
    page_title="Property Price Inference",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Predicción de precio de propiedad")
st.caption("Interfaz de inferencia usando el modelo expuesto por FastAPI")

st.sidebar.header("Configuración")
st.sidebar.write(f"Endpoint API actual: `{API_URL}`")

tab_infer, tab_hist = st.tabs(["🔮 Inferencia", "📜 Historial y explicabilidad"])

# ---------------------------------------------------------------------
# TAB 1: INFERENCIA
# ---------------------------------------------------------------------
with tab_infer:
    st.subheader("Ingresar características de la propiedad")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            bed = st.number_input("Camas (bed)", min_value=0, step=1, value=3)
            bath = st.number_input("Baños (bath)", min_value=0.0, step=0.5, value=2.0)
            house_size = st.number_input(
                "Área de la casa (house_size, ft²)",
                min_value=0.0,
                step=10.0,
                value=1200.0,
            )
            acre_lot = st.number_input(
                "Tamaño del lote (acre_lot, acres)",
                min_value=0.0,
                step=0.1,
                value=0.1,
            )

        with col2:
            status = st.text_input("Estado (status)", value="for_sale")
            brokered_by = st.text_input("Agencia / corredor (brokered_by)")
            street = st.text_input("Dirección (street)")
            city = st.text_input("Ciudad (city)", value="Some City")
            state = st.text_input("Estado (state)", value="CA")
            zip_code = st.text_input("Código postal (zip_code)", value="00000")

            prev_sold_date = st.date_input(
                "Fecha de venta anterior (prev_sold_date)",
                value=None,
                format="YYYY-MM-DD",
            )

        submitted = st.form_submit_button("Predecir precio")

    if submitted:
        # Construir payload con el mismo esquema que ModelPredictionRequest
        payload = {
            "bath": bath,
            "prev_sold_date": prev_sold_date.isoformat()
            if prev_sold_date
            else None,
            "state": state or None,
            "status": status or None,
            "city": city or None,
            "zip_code": zip_code or None,
            "house_size": house_size,
            "bed": int(bed) if bed is not None else None,
            "brokered_by": brokered_by or None,
            "street": street or None,
            "acre_lot": acre_lot,
        }

        st.write("### Payload enviado a la API:")
        st.json(payload)

        try:
            with st.spinner("Llamando a la API de inferencia..."):
                resp = requests.post(
                    f"{API_URL}",
                    json=payload,
                    timeout=15,
                )

            if resp.status_code == 200:
                data = resp.json()
                pred = data.get("prediction")
                st.success("Predicción recibida correctamente ✅")
                st.metric("Precio estimado (unidades del modelo)", pred)
                st.write("Características normalizadas utilizadas (vista desde API):")
                st.json(data.get("features"))
            else:
                st.error(
                    f"Error {resp.status_code} al llamar a /predict: {resp.text}"
                )

        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API: {e}")

# ---------------------------------------------------------------------
# TAB 2: HISTORIAL Y EXPLICABILIDAD (ESQUELETO)
# ---------------------------------------------------------------------
with tab_hist:
    st.subheader("Historial de modelos y explicabilidad (MLflow + SHAP)")

    st.markdown(
        """
        Aquí puedes:
        - Ver las versiones de modelo registradas en MLflow y su *stage* (Production, Staging, etc.).
        - Comparar métricas entre versiones.
        - Mostrar interpretabilidad (por ejemplo con SHAP) para el modelo actual en producción.
        
        Lo siguiente es un ejemplo de cómo podrías hacerlo con MLflowClient; 
        deberás adaptarlo a tus nombres de experimento/modelo reales.
        """
    )

    if st.button("Cargar información de modelos desde MLflow"):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            MLFLOW_TRACKING_URI = os.getenv(
                "MLFLOW_TRACKING_URI", "http://10.43.100.99:8003"
            )
            MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-model")

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            st.write(f"MLflow URI: `{MLFLOW_TRACKING_URI}`")
            st.write(f"Model Registry: `{MODEL_NAME}`")

            # Listar versiones del modelo
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

            rows = []
            for mv in model_versions:
                rows.append(
                    {
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "status": mv.status,
                        "creation_timestamp": mv.creation_timestamp,
                    }
                )

            if rows:
                st.write("### Versiones de modelo en el Model Registry")
                st.dataframe(rows)
            else:
                st.info("No se encontraron versiones de modelo en el registro.")

        except Exception as e:
            st.error(f"Error consultando MLflow: {e}")

    st.markdown(
        """
        Para la parte de **SHAP**, una estrategia típica es:

        1. Cargar desde MLflow el modelo en producción (`models:/<MODEL_NAME>@prod`).
        2. Cargar o reconstruir el *preprocessor* (por ejemplo desde tu `preprocessor.pkl`).
        3. Tomar algunas observaciones de ejemplo (p. ej., últimas filas de CLEAN DATA).
        4. Calcular valores SHAP y graficarlos (summary plot, force plot, etc.) dentro de Streamlit.

        Esa lógica depende de cómo tengas montados los artefactos, por eso aquí te dejo solo el esqueleto.
        """
    )
