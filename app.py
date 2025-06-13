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
# FUNGSI-FUNGSI UTAMA (dengan caching untuk efisiensi)
# =============================================================================

@st.cache_resource
def load_all_assets():
    """Memuat semua aset model dan scaler yang sudah dilatih."""
    assets = {}
    model_dir = 'model/'
    try:
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
            f"Pastikan Anda sudah menjalankan 'training_script.py' terlebih dahulu."
        )
        return None

@st.cache_data(ttl=3600)
def load_data(ticker="BTC-USD"):
    """Mengambil data historis Bitcoin terbaru menggunakan parameter 'period'."""
    st.info("Mengambil data terbaru dari server yfinance...")
    data = yf.download(
        tickers=ticker,
        period="200d",  # Data 200 hari terakhir agar MA dan lag bisa dihitung
        progress=False
    )
    if data.empty:
        st.warning("Gagal mengambil data dari yfinance. Coba lagi nanti.")
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

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
                feature_data = create_features(raw_data)

                if not feature_data.empty:
                    prediction = 0.0
                    
                    if selected_model_code == "lstm_model":
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
                    
                    st.success("Prediksi berhasil dibuat!")
                    
                    history_df = raw_data.tail(90)
                    prediction_date = history_df.index[-1] + timedelta(days=1)

                    st.subheader("Informasi Data")
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"Tanggal historis terakhir: **{history_df.index[-1].strftime('%d %b %Y')}**")
                    with col_info2:
                        st.info(f"Tanggal yang sedang diprediksi: **{prediction_date.strftime('%d %b %Y')}**")
                    
                    st.divider()

                    current_price = raw_data['Close'].iloc[-1]
                    price_change = prediction - current_price
                    pct_change = (price_change / current_price) * 100

                    st.subheader("Hasil Prediksi untuk Esok Hari")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Harga Terakhir (Saat Ini)", f"${current_price:,.2f}")
                    col2.metric("Prediksi Harga Besok", f"${prediction:,.2f}", f"${price_change:,.2f} ({pct_change:.2f}%)")
                    col3.info(f"Model: **{selected_model_display}**")
                    
                    st.subheader("Visualisasi Harga")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], mode='lines', name='Harga Historis'))
                    fig.add_trace(go.Scatter(
                        x=[prediction_date], y=[prediction], mode='markers', name='Harga Prediksi',
                        marker=dict(color='orange', size=12, symbol='star'),
                        hovertemplate=f"<b>Prediksi untuk {prediction_date.strftime('%d %b %Y')}</b><br>Harga: ${prediction:,.2f}<extra></extra>"
                    ))
                    fig.update_layout(
                        title='Pergerakan Harga Bitcoin: 90 Hari Terakhir & Prediksi Besok',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga (USD)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Tidak cukup data untuk membuat fitur.")
        else:
            st.warning("Tidak cukup data historis dari yfinance.")
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: Ini bukan nasihat keuangan.")
else:
    st.error("Aplikasi tidak dapat berjalan karena aset model gagal dimuat.")
