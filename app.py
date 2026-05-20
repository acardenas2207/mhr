# app.py

import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# TÍTULO
# ==========================================

st.title("🩺 Maternal Health Risk Predictor")

st.markdown("""
Esta aplicación utiliza modelos de Machine Learning para predecir el nivel de riesgo materno
basándose en información médica de pacientes embarazadas.

Seleccione un modelo, ingrese los datos solicitados y obtenga la predicción del riesgo.
""")

# ==========================================
# URLS DE MODELOS EN GITHUB
# ==========================================

LOGISTIC_MODEL_URL = "https://raw.githubusercontent.com/acardenas2207/mhr/main/modelos/logistic_regression_model.pkl"

TREE_MODEL_URL = "https://raw.githubusercontent.com/acardenas2207/mhr/main/modelos/decision_tree_model.pkl"

# ==========================================
# FUNCIÓN PARA CARGAR MODELOS
# ==========================================

@st.cache_resource
def load_model(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# ==========================================
# SELECCIÓN DEL MODELO
# ==========================================

model_option = st.selectbox(
    "Seleccione el modelo:",
    (
        "Logistic Regression",
        "Decision Tree"
    )
)

# ==========================================
# CARGA DEL MODELO
# ==========================================

if model_option == "Logistic Regression":
    model = load_model(LOGISTIC_MODEL_URL)
else:
    model = load_model(TREE_MODEL_URL)

# ==========================================
# INPUTS DEL USUARIO
# ==========================================

st.subheader("Ingrese los datos médicos")

Age = st.number_input("Edad", min_value=1, max_value=100, value=25)

SystolicBP = st.number_input("Presión Sistólica", min_value=50, max_value=250, value=120)

DiastolicBP = st.number_input("Presión Diastólica", min_value=30, max_value=200, value=80)

BS = st.number_input("Nivel de azúcar", min_value=0.0, max_value=30.0, value=6.5)

BodyTemp = st.number_input("Temperatura en F°", min_value=90.0, max_value=110.0, value=98.0)

HeartRate = st.number_input("Frecuencia Cardíaca", min_value=40, max_value=200, value=75)

# ==========================================
# DATAFRAME PARA PREDICCIÓN
# ==========================================

input_data = pd.DataFrame({
    'Age': [Age],
    'SystolicBP': [SystolicBP],
    'DiastolicBP': [DiastolicBP],
    'BS': [BS],
    'BodyTemp': [BodyTemp],
    'HeartRate': [HeartRate]
})

# ==========================================
# BOTÓN DE PREDICCIÓN
# ==========================================

if st.button("🔍 Obtener Predicción"):

    prediction = model.predict(input_data)[0]

    # Conversión de etiquetas numéricas
    risk_labels = {
        1: "low risk",
        2: "mid risk",
        3: "high risk"
    }

    predicted_risk = risk_labels.get(prediction, "Desconocido")

    # Mostrar resultado según nivel de riesgo
    if predicted_risk == "low risk":

        st.success(f"🟢 La paciente presenta un riesgo bajo.")
     
    elif predicted_risk == "mid risk":

        st.warning(f"🟡 La paciente presenta un riesgo medio.")
     
    elif predicted_risk == "high risk":

        st.error(f"🔴 La paciente presenta un riesgo alto.")
     
    else:

        st.info("No se pudo determinar el nivel de riesgo.")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.caption("Proyecto de Machine Learning - Maternal Health Risk")

# ==========================================
# INFORMACIÓN ADICIONAL
# ==========================================

st.markdown("### 📘 Notebook en Google Colab")

st.markdown(
    "[Abrir proyecto en Google Colab](https://colab.research.google.com/drive/1OtZFy43oVXSpVNhbgy3eoIWNfgnDo7rg?usp=sharing)"
)

st.markdown("""
**Author:** Julio Alberto Cárdenas Rincón  
**Cod ISIL:** 45862810
""")
st.markdown("---")
