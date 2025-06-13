import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Set page config
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")

# Judul
st.title("📈 Prediksi Harga Bitcoin Menggunakan LSTM")

# Ambil data historis
@st.cache_data
def load_data():
    df = yf.download('BTC-USD', period='100d')
    df.reset_index(inplace=True)
    return df

raw_data = load_data()

# Pastikan data tidak kosong
if raw_data.empty or raw_data['Close'].isnull().all():
    st.error("Data tidak tersedia saat ini. Silakan coba beberapa saat lagi.")
    st.stop()

# Tampilkan data
st.subheader("Data Historis (90 Hari Terakhir)")
st.dataframe(raw_data.tail(10))

# Ambil kolom 'Close'
close_prices = raw_data[['Close']]

# Load scaler dan model
try:
    scaler = joblib.load("scaler.pkl")  # pastikan file scaler.pkl tersedia
    model = load_model("model.h5")      # pastikan model LSTM disimpan di model.h5
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# Normalisasi data
scaled_data = scaler.transform(close_prices)

# Gunakan 60 data terakhir untuk prediksi
window_size = 60
if len(scaled_data) < window_size:
    st.error("Data historis tidak cukup untuk prediksi.")
    st.stop()

last_60 = scaled_data[-window_size:]
X_test = np.reshape(last_60, (1, window_size, 1))

# Prediksi
prediction_scaled = model.predict(X_test)
prediction = scaler.inverse_transform(prediction_scaled)[0][0]

# Harga terakhir
current_price = close_prices['Close'].iloc[-1]
prediction_date = raw_data['Date'].iloc[-1] + timedelta(days=1)

# Debug info
st.subheader("🔍 Debug Info")
st.write("Data Historis (tail):", close_prices.tail())
st.write(f"Harga terakhir (USD): {current_price}")
st.write(f"Prediksi besok (USD): {prediction}")

# Periksa prediksi aneh
if prediction > current_price * 1.5 or prediction < current_price * 0.5:
    st.warning("⚠️ Prediksi harga tampaknya terlalu tinggi atau terlalu rendah. Cek model dan scaler.")

# Visualisasi
st.subheader("📊 Visualisasi Harga")
fig = go.Figure()

# Garis harga historis
fig.add_trace(go.Scatter(
    x=raw_data['Date'],
    y=raw_data['Close'],
    mode='lines',
    name='Harga Historis',
    line=dict(color='royalblue')
))

# Titik prediksi
fig.add_trace(go.Scatter(
    x=[prediction_date],
    y=[prediction],
    mode='markers+text',
    name='Prediksi Harga',
    marker=dict(color='orange', size=12, symbol='star'),
    text=["Prediksi Besok"],
    textposition='top center'
))

fig.update_layout(
    title="Pergerakan Harga Bitcoin 90 Hari Terakhir & Prediksi Harga Besok",
    xaxis_title="Tanggal",
    yaxis_title="Harga (USD)",
    template='plotly_dark',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Ringkasan
st.subheader("📌 Ringkasan Prediksi")
st.markdown(f"""
- Harga Bitcoin terakhir: **${current_price:,.2f}**
- Prediksi harga besok: **${prediction:,.2f}**
- Tanggal prediksi: `{prediction_date.strftime('%Y-%m-%d')}`
""")
