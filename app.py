import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================
# CONFIG DATAROBOT
# =========================
DATAROBOT_API_KEY = "TU_API_KEY_NUEVA"
DATAROBOT_DEPLOYMENT_ID = "6a3d39ed374bc6b8c9e18fc4"
DATAROBOT_HOST = "https://app.datarobot.com"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Predicción Calidad del Agua",
    page_icon="💧",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f3f8ff;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#1e88e5,#42a5f5);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# VARIABLES
# =========================
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

# =========================
# HEADER
# =========================
st.title("Predicción de Calidad del Agua")
st.markdown("Modelo de IA usando DataRobot para predecir si el agua es segura para consumo.")

# =========================
# LAYOUT
# =========================
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Variables de entrada")

    user_data = {}
    for key, label in variables.items():
        user_data[key] = st.number_input(label, min_value=0.0, value=0.0, format="%.6f")

    predict = st.button("Realizar Predicción")

# =========================
# API CALL
# =========================
def predict_datarobot(data):
    url = f"{DATAROBOT_HOST}/predApi/v1.0/deployments/{DATAROBOT_DEPLOYMENT_ID}/predictions"

    headers = {
        "Authorization": f"Bearer {DATAROBOT_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = [data]

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return None, response.text

    return response.json(), None

# =========================
# RESULT
# =========================
with right:
    if predict:
        result, error = predict_datarobot(user_data)

        if error:
            st.error(error)
        else:
            try:
                prediction = result["data"][0]["prediction"]

                positive_prob = 0.0
                prediction_values = result["data"][0].get("predictionValues", [])

                for item in prediction_values:
                    if str(item["label"]) == "1":
                        positive_prob = item["value"]

                negative_prob = 1 - positive_prob

                if prediction == 1:
                    st.success("Agua segura para consumo")
                else:
                    st.error("Agua NO segura para consumo")

                st.metric("Probabilidad de agua segura", f"{positive_prob*100:.2f}%")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=positive_prob * 100,
                    title={'text': "Confianza del modelo"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig, use_container_width=True)

                probs = pd.DataFrame({
                    "Clase": ["Segura", "No segura"],
                    "Probabilidad": [positive_prob, negative_prob]
                })

                fig2 = px.bar(
                    probs,
                    x="Clase",
                    y="Probabilidad",
                    text="Probabilidad"
                )
                st.plotly_chart(fig2, use_container_width=True)

                input_df = pd.DataFrame({
                    "Variable": list(user_data.keys()),
                    "Valor": list(user_data.values())
                })

                fig3 = px.bar(
                    input_df,
                    x="Variable",
                    y="Valor"
                )
                st.plotly_chart(fig3, use_container_width=True)

            except Exception as e:
                st.error(f"Error procesando respuesta: {e}")
