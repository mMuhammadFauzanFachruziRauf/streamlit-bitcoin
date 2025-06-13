# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import requests
import time

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

# =============================================================================
# FUNGSI-FUNGSI UTAMA (dengan caching untuk efisiensi)
# =============================================================================

@st.cache_resource
def load_all_assets():
    """Memuat semua aset model dan scaler yang sudah dilatih."""
    assets = {}
    model_dir = 'model/'
    try:
        # Pastikan path ini benar di repositori Anda
        assets['best_model'] = joblib.load(f'{model_dir}best_bitcoin_model.pkl')
        assets['feature_scaler'] = joblib.load(f'{model_dir}feature_scaler.pkl')
        assets['feature_columns'] = joblib.load(f'{model_dir}feature_columns.pkl')
        assets['lstm_model'] = load_model(f'{model_dir}lstm_bitcoin_model.keras')
        assets['lstm_scaler'] = joblib.load(f'{model_dir}lstm_scaler.pkl')
        st.success("Model dan semua aset berhasil dimuat.")
        return assets
    except FileNotFoundError as e:
        st.error(
            f"File tidak ditemukan: {e.filename}. "
            f"Pastikan folder 'model' dan semua isinya ada di repositori GitHub Anda "
            f"dan path-nya sudah benar."
        )
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat aset model: {e}")
        return None


# --- PERUBAHAN UTAMA DI FUNGSI INI ---
@st.cache_data(ttl=3600) # Cache data selama 1 jam
def load_data(ticker="BTC-USD"):
    """
    Mengambil data historis Bitcoin terbaru dengan cara yang lebih tangguh.
    Fungsi ini menggunakan session dengan user-agent dan logika retry.
    """
    st.info("Mengambil data terbaru dari server yfinance...")

    # Membuat session untuk membuat permintaan terlihat seperti dari browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    # Logika retry: coba 3 kali jika gagal
    for i in range(3):
        try:
            # Gunakan yf.Ticker untuk memanfaatkan session yang sudah dibuat
            btc_ticker = yf.Ticker(ticker, session=session)
            # Ambil data 200 hari terakhir untuk memastikan perhitungan MA cukup
            data = btc_ticker.history(period="200d", auto_adjust=True)

            if not data.empty:
                st.success("Data berhasil diambil dari yfinance.")
                # Mengganti nama kolom agar konsisten (yf terkadang mengubahnya)
                data.rename(columns={
                    'Open': 'Open', 'High': 'High', 'Low': 'Low',
                    'Close': 'Close', 'Volume': 'Volume'
                }, inplace=True)
                return data
            
        except Exception as e:
            st.warning(f"Percobaan {i+1} gagal: {e}. Mencoba lagi dalam 3 detik...")
            time.sleep(3)

    st.error("Gagal total mengambil data dari yfinance setelah beberapa kali percobaan. Server yfinance mungkin sedang sibuk atau membatasi akses. Silakan coba lagi nanti.")
    return None


def create_features(df):
    """Membuat fitur teknikal yang konsisten dengan saat pelatihan."""
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
# TAMPILAN DAN LOGIKA APLIKASI STREAMLIT
# =============================================================================

st.title("₿ Prediksi & Analisis Harga Bitcoin (BTC-USD)")
st.markdown("Aplikasi interaktif untuk memprediksi harga penutupan Bitcoin esok hari.")

assets = load_all_assets()

if assets:
    st.sidebar.header("Opsi Prediksi")
    model_options = {
        "Model Terbaik (Regresi Linear)": "best_model",
        "Model LSTM": "lstm_model"
    }
    selected_model_display = st.sidebar.selectbox(
        "Pilih Model untuk Prediksi:",
        options=list(model_options.keys())
    )
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi Harga Besok"):
        raw_data = load_data()

        if raw_data is not None and len(raw_data) > 90:
            with st.spinner("Membuat fitur dan melakukan prediksi..."):
                feature_data = create_features(raw_data.copy())

                if not feature_data.empty:
                    prediction = 0.0
                    
                    if selected_model_code == "lstm_model":
                        model = assets['lstm_model']
                        scaler = assets['lstm_scaler']
                        lookback = 60
                        # Pastikan data yang diambil cukup untuk lookback
                        if len(raw_data) >= lookback:
                            latest_prices = raw_data['Close'].iloc[-lookback:].values.reshape(-1, 1)
                            latest_scaled = scaler.transform(latest_prices)
                            input_lstm = latest_scaled.reshape(1, lookback, 1)
                            prediction_scaled = model.predict(input_lstm)
                            prediction = scaler.inverse_transform(prediction_scaled)[0][0]
                        else:
                            st.warning(f"Data tidak cukup untuk lookback LSTM ({len(raw_data)}/{lookback} baris tersedia).")
                            st.stop()
                    else:
                        model = assets['best_model']
                        scaler = assets['feature_scaler']
                        feature_columns = assets['feature_columns']
                        latest_input_df = feature_data[feature_columns].iloc[-1:]
                        input_scaled = scaler.transform(latest_input_df)
                        prediction = model.predict(input_scaled)[0]
                    
                    st.success("Prediksi berhasil dibuat!")
                    
                    hist
