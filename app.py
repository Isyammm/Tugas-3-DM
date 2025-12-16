import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =========================
# LOAD MODEL
# =========================
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# =========================
# UI STREAMLIT
# =========================
st.title("🚦 Prediksi Kondisi Lalu Lintas & Kecelakaan")
st.write("Aplikasi berbasis Machine Learning menggunakan K-Means & Logistic Regression")

st.sidebar.header("Input Data")

vehicle_count = st.sidebar.number_input("Jumlah Kendaraan", min_value=0, max_value=500, value=100)
avg_speed = st.sidebar.number_input("Kecepatan Rata-rata (km/h)", min_value=0.0, max_value=150.0, value=40.0)
visibility = st.sidebar.number_input("Jarak Pandang (meter)", min_value=0, max_value=10000, value=500)

# =========================
# DATAFRAME INPUT
# =========================
input_data = pd.DataFrame({
    'Vehicle_Count': [vehicle_count],
    'Avg_Speed(km/h)': [avg_speed],
    'Visibility(m)': [visibility]
})

# =========================
# NORMALISASI
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# PREDIKSI
# =========================
cluster = kmeans.predict(input_scaled)[0]
accident_pred = logreg.predict(input_scaled)[0]
accident_prob = logreg.predict_proba(input_scaled)[0][1]

# =========================
# OUTPUT
# =========================
st.subheader("📊 Hasil Prediksi")

st.write(f"**Cluster Lalu Lintas:** {cluster}")

if accident_pred == 1:
    st.error(f"⚠️ Potensi Kecelakaan: YA ({accident_prob*100:.2f}%)")
else:
    st.success(f"✅ Potensi Kecelakaan: TIDAK ({accident_prob*100:.2f}%)")

# =========================
# INTERPRETASI CLUSTER
# =========================
st.subheader("🧠 Interpretasi Cluster")

if cluster == 0:
    st.write("Lalu lintas relatif lancar dengan kecepatan stabil.")
elif cluster == 1:
    st.write("Lalu lintas padat, kecepatan rendah, berisiko kemacetan.")
else:
    st.write("Lalu lintas sepi namun kurang efisien.")

st.caption("Model dilatih menggunakan dataset transportasi sintetis.")
