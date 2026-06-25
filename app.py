import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Predicción de Calidad del Agua",
    layout="wide",
)

API_KEY = st.secrets["DATAROBOT_API_KEY"]
DEPLOYMENT_ID = st.secrets["DATAROBOT_DEPLOYMENT_ID"]
HOST = st.secrets["DATAROBOT_HOST"]

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #f4f9ff;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #0b3d91;
}

.subtitle {
    font-size: 18px;
    color: #4f6b95;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#1565c0,#42a5f5);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: 600;
}

.metric-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# VARIABLES
# ============================================================
variables = {
    "aluminium": "Aluminio",
    "ammonia": "Amonio",
    "arsenic": "Arsénico",
    "barium": "Bario",
    "cadmium": "Cadmio",
    "chloramine": "Cloraminas",
    "chromium": "Cromo",
    "copper": "Cobre",
    "flouride": "Fluoruro",
    "bacteria": "Bacteria",
    "viruses": "Virus",
    "lead": "Plomo",
    "nitrates": "Nitratos",
    "nitrites": "Nitritos",
    "mercury": "Mercurio",
    "perchlorate": "Perclorato",
    "radium": "Radio",
    "selenium": "Selenio",
    "silver": "Plata",
    "uranium": "Uranio"
}

# ============================================================
# API
# ============================================================
def predict_datarobot(payload):
    url = f"{HOST}/predApi/v1.0/deployments/{DEPLOYMENT_ID}/predictions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=[payload], headers=headers)

    if response.status_code != 200:
        return None, response.text

    return response.json(), None


def parse_prediction(response_json):
    """
    Ajustar si DataRobot devuelve otra estructura.
    """
    row = response_json["data"][0]

    prediction = row.get("prediction", None)

    positive_prob = 0.0
    if "predictionValues" in row:
        for pred in row["predictionValues"]:
            if str(pred["label"]) == "1":
                positive_prob = pred["value"]

    return prediction, positive_prob


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">Predicción de Calidad del Agua</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Modelo de inteligencia artificial conectado a DataRobot para evaluar si el agua es segura para consumo humano.</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# LAYOUT
# ============================================================
left, right = st.columns([1, 1.4])

# ============================================================
# INPUTS
# ============================================================
with left:
    st.markdown("### Variables del agua")

    user_data = {}

    for key, label in variables.items():
        user_data[key] = st.number_input(
            label,
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.6f"
        )

    predict = st.button("Realizar Predicción")

# ============================================================
# OUTPUT
# ============================================================
with right:
    st.markdown("### Resultado")

    if predict:
        with st.spinner("Consultando modelo..."):
            result, error = predict_datarobot(user_data)

        if error:
            st.error(error)
        else:
            try:
                prediction, positive_prob = parse_prediction(result)
                negative_prob = 1 - positive_prob

                if prediction == 1:
                    st.success("Agua segura para consumo")
                else:
                    st.error("Agua NO segura para consumo")

                st.metric(
                    "Probabilidad de agua segura",
                    f"{positive_prob * 100:.2f}%"
                )

                # Gauge
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=positive_prob * 100,
                    title={'text': "Confianza del modelo"},
                    gauge={
                        'axis': {'range': [0, 100]}
                    }
                ))
                st.plotly_chart(gauge, use_container_width=True)

                # Probabilidades
                prob_df = pd.DataFrame({
                    "Clase": ["Segura", "No Segura"],
                    "Probabilidad": [positive_prob, negative_prob]
                })

                bar1 = px.bar(
                    prob_df,
                    x="Clase",
                    y="Probabilidad",
                    text="Probabilidad",
                    title="Probabilidad por Clase"
                )
                st.plotly_chart(bar1, use_container_width=True)

                # Inputs chart
                inputs_df = pd.DataFrame({
                    "Variable": list(user_data.keys()),
                    "Valor": list(user_data.values())
                })

                bar2 = px.bar(
                    inputs_df,
                    x="Variable",
                    y="Valor",
                    title="Variables Ingresadas"
                )
                st.plotly_chart(bar2, use_container_width=True)

            except Exception as e:
                st.error(f"Error procesando respuesta del modelo: {e}")
