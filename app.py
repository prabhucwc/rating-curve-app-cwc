import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.title("Stage–Discharge Rating Curve Generator")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if "Observed_Computed" in df.columns:
        df = df[df["Observed_Computed"] == "O"]

    df = df.dropna()

    H = df["Water_Level"].values
    Q_obs = df["Discharge"].values

    def rating_curve(H, a, h0, b):
        return a * (H - h0) ** b

    params, _ = curve_fit(rating_curve, H, Q_obs, maxfev=10000)
    Q_pred = rating_curve(H, *params)

    r2 = r2_score(Q_obs, Q_pred)
    rmse = np.sqrt(mean_squared_error(Q_obs, Q_pred))
    mae = mean_absolute_error(Q_obs, Q_pred)

    st.subheader("Generated Equation")
    st.write(f"Q = {params[0]:.3f} (H - {params[1]:.3f})^{params[2]:.3f}")

    st.subheader("Model Performance")
    st.write(f"R²: {r2:.4f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(H, Q_obs)
    ax.plot(H, Q_pred)
    ax.set_xlabel("Water Level")
    ax.set_ylabel("Discharge")
    st.pyplot(fig)
