import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import date, timedelta
from curl_cffi.requests import Session

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="₿", layout="wide")

# =============================================================================
# FUNGSI HELPERS (Tidak perlu diubah)
# =============================================================================

@st.cache_resource
def load_all_assets():
    """Memuat semua aset model dan scaler dari disk."""
    try:
        assets = {
            'best_model': joblib.load('model/best_bitcoin_model.pkl'),
            'feature_scaler': joblib.load('model/feature_scaler.pkl'),
            'feature_columns': joblib.load('model/feature_columns.pkl'),
            'lstm_model': load_model('model/lstm_bitcoin_model.keras'),
            'lstm_scaler': joblib.load('model/lstm_scaler.pkl')
        }
        return assets
    except Exception as e:
        st.error(f"Error fatal saat memuat aset model: {e}. Aplikasi tidak bisa berjalan.")
        return None

@st.cache_data(ttl=3600)
def _get_data_from_yfinance(ticker="BTC-USD"):
    """Fungsi MURNI yang di-cache: Hanya download data dan perbaiki kolom."""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        session = Session(impersonate="chrome110")
        data = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, session=session)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return e

def load_data_with_ui(ticker="BTC-USD"):
    """Fungsi yang dipanggil aplikasi: Menangani UI dan memanggil fungsi cache."""
    data = _get_data_from_yfinance(ticker)
    if isinstance(data, pd.DataFrame) and not data.empty:
        return data
    elif isinstance(data, Exception):
        st.error(f"Terjadi error saat mengambil data: {data}")
        return None
    else:
        st.error(f"Gagal mengambil data untuk {ticker} (tidak ada data yang dikembalikan).")
        return None
        
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

def run_prediction(assets, raw_data, model_code):
    LSTM_LOOKBACK = 60
    if model_code == "lstm_model":
        model, scaler = assets['lstm_model'], assets['lstm_scaler']
        if len(raw_data) < LSTM_LOOKBACK: return None
        latest_prices = raw_data['Close'].iloc[-LSTM_LOOKBACK:].values.reshape(-1, 1)
        latest_scaled = scaler.transform(latest_prices)
        input_lstm = np.reshape(latest_scaled, (1, LSTM_LOOKBACK, 1))
        prediction_scaled = model.predict(input_lstm)
        return float(scaler.inverse_transform(prediction_scaled)[0][0])
    else: # "best_model"
        model, scaler, feature_columns = assets['best_model'], assets['feature_scaler'], assets['feature_columns']
        feature_data = create_features(raw_data.copy())
        if feature_data.empty: return None
        latest_input_df = feature_data.iloc[-1:]
        input_scaled = scaler.transform(latest_input_df[feature_columns])
        return float(model.predict(input_scaled)[0])

# =============================================================================
# FUNGSI TAMPILAN TERPADU
# =============================================================================
def display_content():
    """Satu fungsi untuk menampilkan semua konten berdasarkan session_state."""
    
    # Ambil data dari session_state
    raw_data = st.session_state.get('raw_data')
    prediction_result = st.session_state.get('prediction_result')
    model_display_name = st.session_state.get('model_name')
    
    # Tampilkan Metrik hanya jika ada hasil prediksi
    if prediction_result is not None and model_display_name is not None:
        st.subheader("Hasil Prediksi untuk Esok Hari")
        history_df_metric = raw_data.tail(90)
        current_price = float(history_df_metric['Close'].values[-1])
        price_change = prediction_result - current_price
        pct_change = (price_change / current_price) * 100 if current_price != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Terakhir", f"${current_price:,.2f}", f"per {history_df_metric.index[-1].strftime('%d %b %Y')}")
        
        color = "#00c04b" if price_change >= 0 else "#ff4b4b"
        delta_string = f"${price_change:+.2f} ({pct_change:+.2f}%)"
        col2.markdown(f"""
        <div style="font-size: 0.875rem; color: rgba(250, 250, 250, 0.6);">Prediksi Harga Besok</div>
        <div style="font-size: 1.75rem; font-weight: 600;">{f"${prediction_result:,.2f}"}</div>
        <div style="color: {color}; font-weight: 600;">{delta_string}</div>
        """, unsafe_allow_html=True)
        col3.metric("Model Digunakan", model_display_name)
    
    # Tampilkan Grafik (selalu, selama data ada)
    st.subheader("Visualisasi Harga")
    history_df_chart = raw_data.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df_chart.index, y=history_df_chart['Close'], name='Harga Historis', mode='lines', line=dict(color='royalblue')))
    
    if prediction_result is not None:
        prediction_date = history_df_chart.index[-1].to_pydatetime() + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[prediction_date], y=[prediction_result], name='Prediksi Harga', mode='markers', marker=dict(color='orange', size=12, symbol='star')))
        
    fig.update_layout(title='Pergerakan Harga Bitcoin & Prediksi', xaxis_title='Tanggal', yaxis_title='Harga (USD)')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# ALUR UTAMA APLIKASI
# =============================================================================
def main():
    st.title("₿ Prediksi & Analisis Harga Bitcoin")
    
    # Muat aset-aset penting
    assets = load_all_assets()
    if not assets:
        st.stop()

    # Muat data historis ke dalam state jika belum ada
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = load_data_with_ui()

    # --- Sidebar ---
    st.sidebar.header("Opsi Prediksi")
    model_options = {"Model Regresi": "best_model", "Model LSTM": "lstm_model"}
    selected_model_display = st.sidebar.selectbox("Pilih Model:", options=list(model_options.keys()))
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi", type="primary"):
        if st.session_state['raw_data'] is not None:
            with st.spinner("Memproses prediksi..."):
                prediction = run_prediction(assets, st.session_state['raw_data'], selected_model_code)
                if prediction is not None:
                    st.session_state['prediction_result'] = prediction
                    st.session_state['model_name'] = selected_model_display
                else:
                    st.error("Gagal membuat prediksi. Data tidak mencukupi untuk membuat fitur.")
                    # Jangan hapus raw_data, hapus saja hasil prediksi sebelumnya
                    st.session_state.pop('prediction_result', None)
                    st.session_state.pop('model_name', None)
        else:
            st.error("Gagal prediksi karena data historis tidak dapat dimuat.")

    st.sidebar.info("Aplikasi ini bukan merupakan nasihat keuangan.")
    st.divider()

    # --- Tampilan Konten Utama ---
    if st.session_state.get('raw_data') is not None:
        display_content()
    else:
        st.warning("Gagal memuat data awal. Silakan coba muat ulang halaman.")

if __name__ == '__main__':
    main()
