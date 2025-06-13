import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from curl_cffi.requests import Session

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

# =============================================================================
# DEFINISI KONSTANTA
# =============================================================================
MODEL_DIR = 'model/'
LSTM_LOOKBACK = 60

# =============================================================================
# FUNGSI-FUNGSI UTAMA
# =============================================================================

@st.cache_resource
def load_all_assets():
    """Memuat semua aset model dan scaler dari disk."""
    try:
        assets = {
            'best_model': joblib.load(f'{MODEL_DIR}best_bitcoin_model.pkl'),
            'feature_scaler': joblib.load(f'{MODEL_DIR}feature_scaler.pkl'),
            'feature_columns': joblib.load(f'{MODEL_DIR}feature_columns.pkl'),
            'lstm_model': load_model(f'{MODEL_DIR}lstm_bitcoin_model.keras'),
            'lstm_scaler': joblib.load(f'{MODEL_DIR}lstm_scaler.pkl')
        }
        return assets
    except Exception as e:
        st.error(f"Error fatal saat memuat aset model: {e}. Aplikasi tidak bisa berjalan.")
        return None

# =============================================================================
# Pola Pengambilan Data yang Kuat (Pemisahan Cache dan UI)
# =============================================================================

@st.cache_data(ttl=3600)
def _get_data_from_yfinance(ticker="BTC-USD"):
    """Fungsi MURNI yang di-cache: Hanya download data, tanpa interaksi UI."""
    try:
        session = Session(impersonate="chrome110")
        data = yf.download(tickers=ticker, period="250d", auto_adjust=True, progress=False, session=session)
        return data
    except Exception as e:
        return e

def load_data_with_ui(ticker="BTC-USD"):
    """Fungsi yang dipanggil oleh aplikasi: Menangani UI dan memanggil fungsi cache."""
    st.info("Memeriksa data harga Bitcoin terbaru...")
    data = _get_data_from_yfinance(ticker)
    
    if isinstance(data, pd.DataFrame):
        if data.empty:
            st.error(f"Gagal mengambil data untuk {ticker} (tidak ada data yang dikembalikan).")
            return None
        st.success("Data berhasil dimuat.")
        return data
    elif isinstance(data, Exception):
        st.error(f"Terjadi error saat mengambil data: {data}")
        return None
    return None # Fallback jika data bukan DataFrame atau Exception

def create_features(df):
    df_feat = df.copy()
    df_feat['MA_7'] = df_feat['Close'].rolling(window=7).mean()
    df_feat['MA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['Daily_Return'] = df_feat['Close'].pct_change()
    df_feat.dropna(inplace=True)
    return df_feat

def run_prediction(assets, raw_data, model_code):
    if model_code == "lstm_model":
        # Logika LSTM
        model, scaler = assets['lstm_model'], assets['lstm_scaler']
        if len(raw_data) < LSTM_LOOKBACK: return None
        latest_prices = raw_data['Close'].iloc[-LSTM_LOOKBACK:].values.reshape(-1, 1)
        latest_scaled = scaler.transform(latest_prices)
        input_lstm = np.reshape(latest_scaled, (1, LSTM_LOOKBACK, 1))
        prediction_scaled = model.predict(input_lstm)
        return float(scaler.inverse_transform(prediction_scaled)[0][0])
    else:
        # Logika Regresi
        model, scaler, feature_columns = assets['best_model'], assets['feature_scaler'], assets['feature_columns']
        feature_data = create_features(raw_data.copy())
        if feature_data.empty: return None
        latest_input_df = feature_data.iloc[-1:]
        input_scaled = scaler.transform(latest_input_df[feature_columns])
        return float(model.predict(input_scaled)[0])

def display_prediction_results(prediction_result, raw_data, model_display_name):
    history_df = raw_data.tail(90)
    prediction_date = history_df.index[-1].to_pydatetime() + timedelta(days=1)
    current_price = float(history_df['Close'].values[-1])
    
    price_change = prediction_result - current_price
    pct_change = (price_change / current_price) * 100 if current_price != 0 else 0

    st.subheader("Hasil Prediksi untuk Esok Hari")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir", f"${current_price:,.2f}", f"per {history_df.index[-1].strftime('%d %b %Y')}")
    col2.metric("Prediksi Harga Besok", f"${prediction_result:,.2f}", f"${price_change:+.2f} ({pct_change:+.2f}%)")
    col3.metric("Model Digunakan", model_display_name)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Harga Historis', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=[prediction_date], y=[prediction_result], name='Prediksi Harga', mode='markers', marker=dict(color='orange', size=12, symbol='star')))
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAMPILAN STREAMLIT UTAMA
# =============================================================================
def main():
    st.title("₿ Prediksi & Analisis Harga Bitcoin")
    
    assets = load_all_assets()
    if not assets:
        st.stop()

    st.sidebar.header("Opsi Prediksi")
    model_options = {"Model Regresi": "best_model", "Model LSTM": "lstm_model"}
    selected_model_display = st.sidebar.selectbox("Pilih Model:", options=list(model_options.keys()))
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi", type="primary"):
        with st.spinner("Memproses prediksi..."):
            raw_data_pred = load_data_with_ui()
            if raw_data_pred is not None:
                prediction = run_prediction(assets, raw_data_pred, selected_model_code)
                if prediction is not None:
                    st.session_state['prediction_result'] = prediction
                    st.session_state['raw_data'] = raw_data_pred
                    st.session_state['model_name'] = selected_model_display
                else:
                    st.error("Gagal membuat prediksi. Data tidak mencukupi.")
                    st.session_state.clear()
            else:
                st.error("Gagal prediksi karena data tidak dapat diambil.")
                st.session_state.clear()

    st.sidebar.info("Aplikasi ini bukan merupakan nasihat keuangan.")
    st.divider()

    if 'prediction_result' in st.session_state:
        display_prediction_results(
            st.session_state['prediction_result'],
            st.session_state['raw_data'],
            st.session_state['model_name']
        )
    else:
        st.info("Pilih model di sidebar & klik 'Lakukan Prediksi' untuk memulai.")
        raw_data_hist = load_data_with_ui()
        
        # Pengecekan paling kuat dan defensif
        if isinstance(raw_data_hist, pd.DataFrame) and not raw_data_hist.empty:
            st.subheader("Pergerakan Harga Bitcoin (90 Hari Terakhir)")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=raw_data_hist.index.tail(90), y=raw_data_hist['Close'].tail(90)))
            fig_hist.update_layout(title='Grafik Harga Historis BTC-USD')
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Tidak dapat menampilkan grafik karena data awal gagal dimuat.")

if __name__ == '__main__':
    main()
