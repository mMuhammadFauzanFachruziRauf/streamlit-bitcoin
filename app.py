import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --------------------- Fungsi Ambil Data ---------------------
@st.cache_data
def load_data():
    df = yf.download("BTC-USD", period="100d", interval="1d")
    df.reset_index(inplace=True)
    return df

# --------------------- Fungsi Training Model ---------------------
def train_models(df):
    df = df[['Date', 'Close']].dropna()
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

    X = df[['Date_ordinal']]
    y = df['Close']

    linear_model = LinearRegression()
    linear_model.fit(X, y)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    return linear_model, rf_model

# --------------------- Fungsi Prediksi ---------------------
def predict_tomorrow(df, model):
    last_date = pd.to_datetime(df['Date'].max())
    tomorrow_date = last_date + timedelta(days=1)
    tomorrow_ordinal = np.array([[tomorrow_date.toordinal()]])

    predicted_price = model.predict(tomorrow_ordinal)[0]
    return tomorrow_date, predicted_price

# --------------------- Fungsi Visualisasi ---------------------
def plot_data(df, tomorrow_date, predicted_price):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], label='Harga Historis', color='blue')
    ax.plot(tomorrow_date, predicted_price, 'orange', marker='*', markersize=15, label='Prediksi Harga')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (USD)")
    ax.set_title("Pergerakan Harga Bitcoin 90 Hari Terakhir & Prediksi Harga Besok")
    ax.legend()
    ax.grid(True)

    # Tambahkan teks label di titik prediksi
    ax.annotate('Prediksi Besok', xy=(tomorrow_date, predicted_price),
                xytext=(tomorrow_date, predicted_price + 500),
                arrowprops=dict(facecolor='orange', shrink=0.05),
                fontsize=10, color='orange')

    return fig

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📈 Visualisasi Harga")

# Sidebar pilihan model
st.sidebar.header("🧠 Opsi Prediksi")
model_option = st.sidebar.selectbox("Pilih Model untuk Prediksi:",
                                    ("Model Terbaik (Regresi Linear)", "Random Forest"))

# Tombol prediksi
predict = st.sidebar.button("🚀 Lakukan Prediksi Harga Besok")

# Load dan tampilkan data
raw_data = load_data()
history_df = raw_data.tail(90)[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
history_df = history_df.dropna(subset=['Close'])

# Validasi data kosong
if bool(history_df['Close'].isnull().any()):
    st.warning("Terdapat nilai kosong dalam data historis.")

# Tampilkan tabel data
st.subheader("📅 Data Harga Bitcoin 90 Hari Terakhir")
st.dataframe(history_df, use_container_width=True)

# Jalankan model prediksi saat tombol ditekan
if predict:
    linear_model, rf_model = train_models(history_df)

    if model_option == "Random Forest":
        model = rf_model
    else:
        model = linear_model

    pred_date, pred_price = predict_tomorrow(history_df, model)

    # Tampilkan hasil prediksi
    st.subheader("📊 Hasil Prediksi")
    st.markdown(f"📅 **Tanggal Prediksi:** `{pred_date.date()}`")
    st.markdown(f"💰 **Prediksi Harga (USD):** `{pred_price:,.2f}`")

    # Visualisasi
    fig = plot_data(history_df, pred_date, pred_price)
    st.pyplot(fig)
