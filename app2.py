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


def ejecutar_prediccion_datarobot(df_entrada: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "entrada.csv"
        output_path = Path(temp_dir) / "salida.csv"

        df_entrada.to_csv(input_path, index=False, encoding="utf-8")

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

    st.dataframe(df_resultado, use_container_width=True)

    columnas_prediccion = [
        col for col in df_resultado.columns
        if "prediction" in col.lower()
        and "status" not in col.lower()
    ]

    if columnas_prediccion:
        valor = df_resultado[columnas_prediccion[0]].iloc[0]
        st.metric("Predicción del modelo", valor)

    columnas_estado = [
        col for col in df_resultado.columns
        if "prediction_status" in col.lower()
    ]

    if columnas_estado:
        st.caption(f"Estado: {df_resultado[columnas_estado[0]].iloc[0]}")

    csv = df_resultado.to_csv(index=False).encode("utf-8")

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

    with col1:
        aluminio = st.number_input("Aluminio", value=0.04, format="%.6f")
        amonio = st.number_input("Amonio", value=0.05, format="%.6f")
        arsenico = st.number_input("Arsénico", value=0.01, format="%.6f")
        bario = st.number_input("Bario", value=0.01, format="%.6f")
        cadmio = st.number_input("Cadmio", value=0.02, format="%.6f")
        cloraminas = st.number_input("Cloraminas", value=0.02, format="%.6f")
        cromo = st.number_input("Cromo", value=0.02, format="%.6f")
        cobre = st.number_input("Cobre", value=0.01, format="%.6f")
        fluoruro = st.number_input("Fluoruro", value=0.02, format="%.6f")
        bacteria = st.number_input("Bacteria", value=0.03, format="%.6f")

    with col2:
        virus = st.number_input("Virus", value=0.01, format="%.6f")
        plomo = st.number_input("Plomo", value=0.03, format="%.6f")
        nitratos = st.number_input("Nitratos", value=0.02, format="%.6f")
        nitritos = st.number_input("Nitritos", value=0.02, format="%.6f")
        mercurio = st.number_input("Mercurio", value=0.03, format="%.6f")
        perclorato = st.number_input("Perclorato", value=0.01, format="%.6f")
        radio = st.number_input("Radio", value=0.08, format="%.6f")
        selenio = st.number_input("Selenio", value=0.03, format="%.6f")
        plata = st.number_input("Plata", value=0.07, format="%.6f")
        uranio = st.number_input("Uranio", value=0.21, format="%.6f")

    enviar = st.form_submit_button("Evaluar calidad del agua")


if enviar:
    entrada = pd.DataFrame([
        {
            "aluminium": aluminio,
            "ammonia": amonio,
            "arsenic": arsenico,
            "barium": bario,
            "cadmium": cadmio,
            "chloramine": cloraminas,
            "chromium": cromo,
            "copper": cobre,
            "flouride": fluoruro,
            "bacteria": bacteria,
            "viruses": virus,
            "lead": plomo,
            "nitrates": nitratos,
            "nitrites": nitritos,
            "mercury": mercurio,
            "perchlorate": perclorato,
            "radium": radio,
            "selenium": selenio,
            "silver": plata,
            "uranium": uranio,
        }
    ])

    st.write("Datos enviados al modelo:")
    st.dataframe(entrada, use_container_width=True)

    try:
        with st.spinner("Consultando DataRobot..."):
            resultado = ejecutar_prediccion_datarobot(entrada)

        mostrar_resultado(resultado)

    except Exception as error:
        st.error("Error procesando la predicción.")
        st.code(str(error))
