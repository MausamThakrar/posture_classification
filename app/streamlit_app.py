import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

@st.cache_resource
def load_artefacts():
    base = Path(__file__).resolve().parents[1]
    model = joblib.load(base / "best_posture_model.joblib")
    scaler = joblib.load(base / "scaler.joblib")
    encoder = joblib.load(base / "label_encoder.joblib")
    return model, scaler, encoder

def main():
    st.title("Posture Classification – Demo Dashboard")
    st.write(
        "This app uses a trained machine learning model to classify sitting "
        "posture based on simulated chair sensor inputs."
    )

    model, scaler, encoder = load_artefacts()

    st.sidebar.header("Sensor Inputs")
    st.sidebar.write("Adjust the sliders to simulate sensor readings.")

    sensor_front_left = st.sidebar.slider("Front Left", 0.0, 1.0, 0.8, 0.01)
    sensor_front_right = st.sidebar.slider("Front Right", 0.0, 1.0, 0.8, 0.01)
    sensor_back_left = st.sidebar.slider("Back Left", 0.0, 1.0, 0.9, 0.01)
    sensor_back_right = st.sidebar.slider("Back Right", 0.0, 1.0, 0.9, 0.01)
    sensor_middle = st.sidebar.slider("Middle", 0.0, 1.0, 0.7, 0.01)

    input_df = pd.DataFrame(
        [
            {
                "sensor_front_left": sensor_front_left,
                "sensor_front_right": sensor_front_right,
                "sensor_back_left": sensor_back_left,
                "sensor_back_right": sensor_back_right,
                "sensor_middle": sensor_middle,
            }
        ]
    )

    st.subheader("Current Sensor Values")
    st.dataframe(input_df)

    X_scaled = scaler.transform(input_df)

    y_pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[0]
    else:
        y_proba = None

    predicted_label = encoder.inverse_transform(y_pred)[0]

    st.subheader("Predicted Posture")
    st.markdown(f"### {predicted_label}")

    if y_proba is not None:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame([y_proba], columns=encoder.classes_).T
        prob_df.columns = ["Probability"]
        st.bar_chart(prob_df)

if __name__ == "__main__":
    main()
