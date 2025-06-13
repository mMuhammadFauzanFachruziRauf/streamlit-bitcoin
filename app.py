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
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

# =============================================================================
# LOAD ASET MODEL
# =============================================================================
@st.cache_resource
def load_all_assets():
    assets = {}
    try:
        assets['best_model'] = joblib.load('model/best_bitcoin_model.pkl')
        assets['feature_scaler'] = joblib.load('model/feature_scaler.pkl')
        assets['feature_columns'] = joblib.load('model/feature_columns.pkl')
        assets['lstm_model'] = load_model('model/lstm_bitcoin_model.keras')
        assets['lstm_scaler'] = joblib.load('model/lstm_scaler.pkl')
        st.success("Model dan semua aset berhasil dimuat.")
        return assets
    except Exception as e:
        st.error(f"Gagal memuat aset model: {e}")
        return None

# =============================================================================
# AMBIL DATA BITCOIN
# =============================================================================
@st.cache_data(ttl=3600)
def load_data(ticker="BTC-USD"):
    try:
        data = yf.download(tickers=ticker, period="200d", auto_adjust=True, progress=False)
        if data.empty:
            st.warning("Data yfinance kosong.")
            return None
        data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low',
                             'Close': 'Close', 'Volume': 'Volume'}, inplace=True, errors='ignore')
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# =============================================================================
# BUAT FITUR DARI DATA
# =============================================================================
def create_features(df):
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
# HALAMAN UTAMA
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
    selected_model_display = st.sidebar.selectbox("Pilih Model:", list(model_options.keys()))
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi Harga Besok"):
        raw_data = load_data()
        if raw_data is not None and len(raw_data) > 90:
            feature_data = create_features(raw_data.copy())

            if not feature_data.empty:
                prediction = 0.0
                if selected_model_code == "lstm_model":
                    model = assets['lstm_model']
                    scaler = assets['lstm_scaler']
                    lookback = 60
                    if len(raw_data) >= lookback:
                        latest_prices = raw_data['Close'].iloc[-lookback:].values.reshape(-1, 1)
                        latest_scaled = scaler.transform(latest_prices)
                        input_lstm = np.reshape(latest_scaled, (1, lookback, 1))
                        prediction_scaled = model.predict(input_lstm)
                        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
                    else:
                        st.warning("Data historis kurang dari 60 hari untuk LSTM.")
                        st.stop()
                else:
                    model = assets['best_model']
                    scaler = assets['feature_scaler']
                    feature_columns = assets['feature_columns']
                    input_scaled = scaler.transform(feature_data[feature_columns].iloc[-1:])
                    prediction = model.predict(input_scaled)[0]

                current_price = raw_data['Close'].iloc[-1]
                price_change = prediction - current_price
                pct_change = (price_change / current_price) * 100

                st.success("✅ Prediksi berhasil dibuat!")
                st.divider()

                # ====================
                # TAMPILKAN INFORMASI
                # ====================
                st.subheader("Hasil Prediksi")
                col1, col2, col3 = st.columns(3)
                col1.metric("Harga Saat Ini", f"${current_price:,.2f}")
                col2.metric("Prediksi Harga Besok", f"${prediction:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
                col3.info(f"Model: **{selected_model_display}**")

                # ====================
                # TAMPILKAN GRAFIK
                # ====================
                st.subheader("Visualisasi Harga")
                history_df = raw_data.tail(90)
                prediction_date = history_df.index[-1] + timedelta(days=1)

                # Cek jika data flat
                if history_df['Close'].nunique() == 1:
                    st.warning("Grafik terlihat datar karena semua harga historis sama.")
                    history_df['Close'] += np.random.normal(0, 0.01, size=len(history_df))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df.index, y=history_df['Close'], mode='lines', name='Harga Historis'
                ))
                fig.add_trace(go.Scatter(
                    x=[prediction_date], y=[prediction], mode='markers', name='Harga Prediksi',
                    marker=dict(color='orange', size=12, symbol='star')
                ))
                fig.update_layout(
                    title='Pergerakan Harga Bitcoin: 90 Hari Terakhir & Prediksi Besok',
                    xaxis_title='Tanggal', yaxis_title='Harga (USD)', template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data fitur kosong setelah preprocessing.")
        else:
            st.warning("Data historis tidak mencukupi (min 90 hari).")
else:
    st.error("❌ Model belum berhasil dimuat. Cek folder 'model/'.")
