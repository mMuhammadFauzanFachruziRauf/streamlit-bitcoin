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

@st.cache_data(ttl=3600)
def load_data(ticker="BTC-USD"):
    """Mengambil data historis Bitcoin terbaru."""
    message_container = st.empty()
    message_container.info("Mengambil data terbaru dari server yfinance...")
    try:
        data = yf.download(
            tickers=ticker,
            period="200d",
            auto_adjust=True,
            progress=False
        )
        if data.empty:
            message_container.error(f"Tidak ada data yang diterima dari yfinance untuk ticker {ticker}.")
            return None
        
        # --- PERBAIKAN: Standarisasi nama kolom menjadi huruf kecil ---
        data.columns = [col.lower() for col in data.columns]
        
        message_container.success("Data berhasil diambil dari yfinance.")
        return data
    except Exception as e:
        message_container.error(f"Gagal mengambil data dari yfinance. Kesalahan: {e}")
        return None

def create_features(df):
    """Membuat fitur teknikal yang konsisten dengan saat pelatihan."""
    df_feat = df.copy()
    # --- PERBAIKAN: Menggunakan nama kolom huruf kecil ---
    df_feat['MA_7'] = df_feat['close'].rolling(window=7).mean()
    df_feat['MA_30'] = df_feat['close'].rolling(window=30).mean()
    df_feat['MA_90'] = df_feat['close'].rolling(window=90).mean()
    df_feat['Daily_Return'] = df_feat['close'].pct_change()
    df_feat['Volatility_7'] = df_feat['Daily_Return'].rolling(window=7).std()
    for lag in [1, 2, 3, 7, 14]:
        df_feat[f'Close_lag_{lag}'] = df_feat['close'].shift(lag)
        df_feat[f'Volume_lag_{lag}'] = df_feat['volume'].shift(lag)
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

    if st.sidebar.button("🚀 Lakukan Prediksi Harga Besok", key="predict_button"):
        with st.spinner("Mengambil data dan melakukan prediksi..."):
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
                            # --- PERBAIKAN: Menggunakan nama kolom huruf kecil ---
                            latest_prices = raw_data['close'].iloc[-lookback:].values.reshape(-1, 1)
                            latest_scaled = scaler.transform(latest_prices)
                            input_lstm = np.reshape(latest_scaled, (1, lookback, 1))
                            prediction_scaled = model.predict(input_lstm)
                            prediction = scaler.inverse_transform(prediction_scaled)[0][0]
                        else:
                            st.warning(f"Data tidak cukup untuk lookback LSTM ({len(raw_data)}/{lookback} baris tersedia).")
                            st.stop()
                    else: # 'best_model'
                        model = assets['best_model']
                        scaler = assets['feature_scaler']
                        feature_columns = assets['feature_columns']
                        latest_input_df = feature_data[feature_columns].iloc[-1:]
                        input_scaled = scaler.transform(latest_input_df)
                        prediction = model.predict(input_scaled)[0]
                    
                    st.session_state['prediction_made'] = True
                    st.session_state['raw_data'] = raw_data
                    st.session_state['prediction'] = prediction
                    st.session_state['selected_model_display'] = selected_model_display
                
                else:
                    st.warning("Tidak cukup data untuk membuat fitur setelah proses pembersihan.")
                    st.session_state['prediction_made'] = False
            
            elif raw_data is not None:
                 st.warning(f"Tidak cukup data historis (diperlukan > 90 hari, didapatkan {len(raw_data)} hari).")
                 st.session_state['prediction_made'] = False

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini dibuat untuk tujuan edukasi dan bukan merupakan nasihat keuangan. Selalu lakukan riset Anda sendiri (DYOR).")

if st.session_state.get('prediction_made', False):
    raw_data = st.session_state['raw_data']
    prediction = st.session_state['prediction']
    selected_model_display = st.session_state['selected_model_display']
    
    history_df = raw_data.tail(90)
    prediction_date = history_df.index[-1].to_pydatetime() + timedelta(days=1)

    st.subheader("Informasi Data")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"Tanggal historis terakhir: **{history_df.index[-1].strftime('%d %b %Y')}**")
    with col_info2:
        st.info(f"Tanggal yang sedang diprediksi: **{prediction_date.strftime('%d %b %Y')}**")
    
    st.divider()

    # --- PERBAIKAN: Menggunakan nama kolom huruf kecil ---
    close_data = raw_data['close']
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    current_price = close_data.iloc[-1]

    if pd.isna(current_price) or not np.isscalar(current_price):
        st.error(f"Gagal memproses harga terakhir yang valid. Nilai: '{current_price}'.")
        st.stop()
    if pd.isna(prediction) or not np.isscalar(prediction):
        st.error(f"Model menghasilkan prediksi yang tidak valid. Nilai: '{prediction}'.")
        st.stop()

    price_change = prediction - current_price
    pct_change = (price_change / current_price) * 100

    st.subheader("Hasil Prediksi untuk Esok Hari")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir (Saat Ini)", f"${current_price:,.2f}")
    col2.metric("Prediksi Harga Besok", f"${prediction:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    col3.info(f"Model: **{selected_model_display}**")
    
    st.subheader("Visualisasi Harga")
    fig = go.Figure()
    
    # --- PERBAIKAN: Menggunakan nama kolom huruf kecil ---
    valid_history = history_df.dropna(subset=['close'])
    
    fig.add_trace(go.Scatter(x=valid_history.index, y=valid_history['close'], mode='lines', name='Harga Historis'))
    fig.add_trace(go.Scatter(
        x=[prediction_date], y=[prediction], mode='markers', name='Harga Prediksi',
        marker=dict(color='orange', size=12, symbol='star', line=dict(width=1, color='darkorange')),
        hovertemplate=f"<b>Prediksi untuk {prediction_date.strftime('%d %b %Y')}</b><br>Harga: ${prediction:,.2f}<extra></extra>"
    ))
    
    # --- PERBAIKAN: Menggunakan nama kolom huruf kecil ---
    all_prices = list(valid_history['close']) + [prediction]
    finite_prices = [p for p in all_prices if np.isfinite(p)]
    
    if finite_prices:
        min_price = min(finite_prices)
        max_price = max(finite_prices)
        padding = (max_price - min_price) * 0.10
        yaxis_range = [min_price - padding, max_price + padding]
        
        fig.update_layout(
            title='Pergerakan Harga Bitcoin: 90 Hari Terakhir & Prediksi Besok',
            xaxis_title='Tanggal',
            yaxis_title='Harga (USD)',
            template='plotly_dark',
            yaxis_range=yaxis_range
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada data harga yang valid untuk divisualisasikan pada grafik.")

if 'prediction_made' not in st.session_state:
    st.info("Pilih model di sidebar kiri dan klik tombol 'Lakukan Prediksi' untuk memulai.")

if not assets:
    st.error("Aplikasi tidak dapat berjalan karena aset model gagal dimuat.")

