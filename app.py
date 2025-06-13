import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import date, timedelta
from curl_cffi.requests import Session
import io # <-- Tambahkan import ini untuk debugging

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

@st.cache_data(ttl=3600)
def _get_data_from_yfinance(ticker="BTC-USD"):
    """Fungsi MURNI yang di-cache: Hanya download data, tanpa interaksi UI."""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        session = Session(impersonate="chrome110")
        data = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, session=session)
        return data
    except Exception as e:
        return e

def load_data_with_ui(ticker="BTC-USD"):
    """Fungsi yang dipanggil aplikasi: Menangani UI dan memanggil fungsi cache."""
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
    return None

def create_features(df):
    """Membuat semua fitur teknikal yang dibutuhkan oleh model."""
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
    """Menjalankan proses prediksi dan mengembalikan float."""
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
# DIUBAH: Fungsi display_prediction_results dengan mode debug
# =============================================================================
def display_prediction_results(prediction_result, raw_data, model_display_name):
    """Menampilkan hasil prediksi."""
    
    # =================================================================
    # DEBUGGING SECTION
    # =================================================================
    st.subheader("🕵️‍♂️ DEBUG MODE: Analisis Data")
    st.warning("Tolong screenshot seluruh bagian ini (dari 'DEBUG MODE' sampai bawah) dan kirimkan.")
    
    st.write("---")
    st.write("#### 1. Informasi Umum `raw_data`")
    buffer = io.StringIO()
    raw_data.info(buf=buffer)
    info_string = buffer.getvalue()
    st.text(info_string)

    st.write("#### 2. Lima Baris Terakhir dari `raw_data`")
    st.dataframe(raw_data.tail())

    history_df = raw_data.tail(90)
    st.write("#### 3. Informasi `history_df` (Data yang akan digambar)")
    st.write(f"Jumlah baris dalam `history_df`: **{len(history_df)}**")
    st.write("Lima baris pertama dari `history_df`:")
    st.dataframe(history_df.head())
    
    current_price = float(history_df['Close'].values[-1])
    st.write("#### 4. Nilai Variabel Kunci")
    st.code(f"""
    Tipe data 'prediction_result' : {type(prediction_result)}
    Nilai 'prediction_result'   : {prediction_result}

    Tipe data 'current_price'     : {type(current_price)}
    Nilai 'current_price'         : {current_price}
    """)
    st.write("---")
    # =================================================================
    # END OF DEBUGGING SECTION
    # =================================================================
    
    # Kode asli untuk menampilkan hasil
    prediction_date = history_df.index[-1].to_pydatetime() + timedelta(days=1)
    price_change = prediction_result - current_price
    pct_change = (price_change / current_price) * 100 if current_price != 0 else 0

    st.subheader("Hasil Prediksi untuk Esok Hari")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir", f"${current_price:,.2f}", f"per {history_df.index[-1].strftime('%d %b %Y')}")
    col2.metric("Prediksi Harga Besok", f"${prediction_result:,.2f}", f"${price_change:+.2f} ({pct_change:+.2f}%)")
    col3.metric("Model Digunakan", model_display_name)
    
    st.subheader("Visualisasi Harga")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['Close'], name='Harga Historis', mode='lines', line=dict(color='royalblue')))
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
                    st.error("Gagal membuat prediksi. Data tidak mencukupi untuk membuat fitur.")
                    st.session_state.clear()
            else:
                st.error("Gagal prediksi karena data tidak dapat diambil.")
                st.session_state.clear()

    st.sidebar.info("Aplikasi ini bukan merupakan nasihat keuangan.")
    st.divider()

    if 'prediction_result' in st.session_state and st.session_state.get('prediction_result') is not None:
        display_prediction_results(
            st.session_state['prediction_result'],
            st.session_state['raw_data'],
            st.session_state['model_name']
        )
    else:
        # Bagian ini tidak perlu diubah
        st.info("Pilih model di sidebar & klik 'Lakukan Prediksi' untuk memulai.")

if __name__ == '__main__':
    main()
