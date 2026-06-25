import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Predicción de Calidad del Agua",
    page_icon="💧",
    layout="centered"
)

APP_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = APP_DIR / "predict.py"

DATAROBOT_API_KEY = st.secrets["DATAROBOT_API_KEY"]
DATAROBOT_DEPLOYMENT_ID = st.secrets["DATAROBOT_DEPLOYMENT_ID"]
DATAROBOT_HOST = st.secrets.get("DATAROBOT_HOST", "https://app.datarobot.com")


VARIABLES_MODELO = [
    "Aluminio",
    "Amonio",
    "Arsénico",
    "Bario",
    "Cadmio",
    "Cloraminas",
    "Cromo",
    "Cobre",
    "Fluoruro",
    "Bacteria",
    "Virus",
    "Plomo",
    "Nitratos",
    "Nitritos",
    "Mercurio",
    "Perclorato",
    "Radio",
    "Selenio",
    "Plata",
    "Uranio",
]


def ejecutar_prediccion_datarobot(df_entrada: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "entrada.csv"
        output_path = Path(temp_dir) / "salida.csv"

        df_entrada.to_csv(input_path, index=False, encoding="utf-8-sig")

        comando = [
            sys.executable,
            str(PREDICT_SCRIPT),
            str(input_path),
            str(output_path),
            DATAROBOT_DEPLOYMENT_ID,
            "--api_key",
            DATAROBOT_API_KEY,
            "--host",
            DATAROBOT_HOST,
            "--passthrough_columns_set",
            "--include_prediction_status",
        ]

        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=900
        )

        if resultado.returncode != 0:
            raise RuntimeError(resultado.stderr or resultado.stdout)

        if not output_path.exists():
            raise RuntimeError("DataRobot no generó el archivo de salida.")

        return pd.read_csv(output_path)


def mostrar_resultado(df_resultado: pd.DataFrame):
    st.subheader("Resultado")

    columnas_prediccion = [
        col for col in df_resultado.columns
        if "prediction" in col.lower()
        and "status" not in col.lower()
    ]

    if columnas_prediccion:
        valor = df_resultado[columnas_prediccion[0]].iloc[0]

        if str(valor) in ["1", "1.0", "True", "true"]:
            st.success("El modelo predice que el agua es segura para consumo humano.")
        elif str(valor) in ["0", "0.0", "False", "false"]:
            st.error("El modelo predice que el agua NO es segura para consumo humano.")
        else:
            st.metric("Predicción del modelo", valor)

    st.dataframe(df_resultado, use_container_width=True)

    csv = df_resultado.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Descargar resultado CSV",
        data=csv,
        file_name="resultado_calidad_agua.csv",
        mime="text/csv"
    )


st.title("💧 Predicción de Calidad del Agua")

st.write(
    "Modelo de inteligencia artificial conectado a DataRobot para evaluar "
    "si el agua es segura para consumo humano."
)

if not PREDICT_SCRIPT.exists():
    st.error("No encontré el archivo predict.py. Debe estar en la misma carpeta que app.py.")
    st.stop()


with st.form("formulario_agua"):
    st.subheader("Variables del agua")

    col1, col2 = st.columns(2)

    valores = {}

    for i, variable in enumerate(VARIABLES_MODELO):
        columna = col1 if i < 10 else col2

        with columna:
            valores[variable] = st.number_input(
                variable,
                value=0.01,
                min_value=0.0,
                format="%.6f"
            )

    enviar = st.form_submit_button("Evaluar calidad del agua")


if enviar:
    entrada = pd.DataFrame([valores], columns=VARIABLES_MODELO)

    st.write("Datos enviados al modelo:")
    st.dataframe(entrada, use_container_width=True)

    try:
        with st.spinner("Consultando DataRobot..."):
            resultado = ejecutar_prediccion_datarobot(entrada)

        mostrar_resultado(resultado)

    except Exception as error:
        st.error("Error procesando la predicción.")
        st.code(str(error))
