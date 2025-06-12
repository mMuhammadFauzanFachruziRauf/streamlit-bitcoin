# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

# =============================================================================
# FUNGSI-FUNGSI UTAMA (dengan caching)
# =============================================================================

@st.cache_resource
def load_all_assets():
    """Memuat semua aset model dan scaler yang ada."""
    assets = {}
    try:
        # Model ML Terbaik
        assets['best_model'] = joblib.load('model/best_bitcoin_model.pkl')
        assets['feature_scaler'] = joblib.load('model/feature_scaler.pkl')
        assets['feature_columns'] = joblib.load('model/feature_columns.pkl')

        # Model LSTM
        assets['lstm_model'] = load_model('model/lstm_bitcoin_model.keras')
        assets['lstm_scaler'] = joblib.load('model/lstm_scaler.pkl')

        st.success("Model dan aset berhasil dimuat.")
        return assets
    except FileNotFoundError as e:
        st.error(
            f"File tidak ditemukan: {e.filename}. "
            "Pastikan file-file dari notebook Anda ada di dalam folder 'model/'."
        )
        return None

@st.cache_data(ttl=3600)
def load_data(ticker="BTC-USD"):
    """Mengambil data historis Bitcoin dari Yahoo Finance."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        st.warning("Gagal mengambil data dari yfinance.")
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data

def create_features(df):
    """Membuat fitur teknikal dari data harga mentah."""
    df_feat = df.copy()
    df_feat['MA_7'] = df_feat['Close'].rolling(window=7).mean()
    df_feat['MA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['MA_90'] = df_feat['Close'].rolling(window=90).mean()
    df_feat['Daily_Return'] = df_feat['Close'].pct_change()
    df_feat['Volatility_7'] = df_feat['Daily_Return'].rolling(window=7).std()

    for lag in [1, 2, 3, 7, 14]:
        df_feat[f'Close_lag_{lag}'] = df_feat['Close'].shift(lag)
        df_feat[f'Volume_lag_{lag}'] = df_feat['Volume'].shift(lag)

    df_feat.dropna(inplace=True)
    return df_feat

# =============================================================================
# TAMPILAN APLIKASI STREAMLIT
# =============================================================================

st.title("₿ Prediksi & Analisis Harga Bitcoin (BTC-USD)")
st.markdown("Aplikasi untuk memprediksi harga penutupan Bitcoin menggunakan model ML terbaik dan model LSTM.")

# Memuat semua aset model
assets = load_all_assets()

if assets:
    # Sidebar
    st.sidebar.header("Opsi Prediksi")
    model_options = {
        "Model Terbaik (Regresi Linear)": "best_model",
        "Model LSTM": "lstm_model"
    }
    selected_model_key = st.sidebar.selectbox(
        "Pilih Model untuk Prediksi:",
        options=model_options.keys()
    )

    if st.sidebar.button("🚀 Lakukan Prediksi Harga Besok"):
        with st.spinner("Mengambil dan memproses data terbaru..."):
            raw_data = load_data()
            if raw_data is not None:
                feature_data = create_features(raw_data)

                if not feature_data.empty:
                    model_name = model_options[selected_model_key]

                    if model_name == "lstm_model":
                        model = assets['lstm_model']
                        scaler = assets['lstm_scaler']
                        lookback = 60
                        latest_prices = raw_data['Close'].iloc[-lookback:].values.reshape(-1, 1)
                        latest_scaled = scaler.transform(latest_prices)
                        input_lstm = latest_scaled.reshape(1, lookback, 1)

                        prediction_scaled = model.predict(input_lstm)
                        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
                    else:
                        model = assets['best_model']
                        scaler = assets['feature_scaler']
                        feature_columns = assets['feature_columns']

                        latest_input_df = feature_data[feature_columns].iloc[-1:]
                        input_scaled = scaler.transform(latest_input_df)
                        prediction = model.predict(input_scaled)[0]

                    current_price = raw_data['Close'].iloc[-1]
                    price_change = prediction - current_price
                    pct_change = (price_change / current_price) * 100

                    st.success("Prediksi berhasil dibuat!")
                    st.divider()

                    # Tampilkan hasil
                    st.subheader("Hasil Prediksi untuk Besok")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Harga Saat Ini", f"${current_price:,.2f}")
                    col2.metric("Prediksi Harga Besok", f"${prediction:,.2f}", f"{price_change:,.2f} ({pct_change:.2f}%)")
                    col3.info(f"Model yang digunakan: **{selected_model_key}**")

                    # Visualisasi
                    st.subheader("Visualisasi Harga")
                    history_df = raw_data.tail(90)
                    prediction_date = history_df.index[-1] + timedelta(days=1)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], mode='lines', name='Harga Historis'))
                    fig.add_trace(go.Scatter(x=[prediction_date], y=[prediction], mode='markers', name='Harga Prediksi', marker=dict(color='orange', size=12, symbol='star')))
                    fig.update_layout(title='Pergerakan Harga Bitcoin: 90 Hari Terakhir & Prediksi Besok', xaxis_title='Tanggal', yaxis_title='Harga (USD)')

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Tidak cukup data untuk membuat fitur.")
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: Ini bukan nasihat keuangan.")
else:
    st.error("Gagal memuat aset model. Mohon periksa kembali isi folder `model/` Anda.")
