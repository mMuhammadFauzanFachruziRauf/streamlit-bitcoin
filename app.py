import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# DIUBAH: Import disederhanakan sesuai versi terbaru curl_cffi
from curl_cffi.requests import Session

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

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
    assets = {}
    try:
        assets['best_model'] = joblib.load(f'{MODEL_DIR}best_bitcoin_model.pkl')
        assets['feature_scaler'] = joblib.load(f'{MODEL_DIR}feature_scaler.pkl')
        assets['feature_columns'] = joblib.load(f'{MODEL_DIR}feature_columns.pkl')
        assets['lstm_model'] = load_model(f'{MODEL_DIR}lstm_bitcoin_model.keras')
        assets['lstm_scaler'] = joblib.load(f'{MODEL_DIR}lstm_scaler.pkl')
        return assets
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e.filename}. Pastikan folder '{MODEL_DIR}' ada.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat aset model: {e}")
        return None

# =============================================================================
# DIUBAH: Fungsi load_data disederhanakan dengan API curl_cffi terbaru
# =============================================================================
@st.cache_data(ttl=3600)
def load_data(ticker="BTC-USD"):
    """Mengambil data historis dari Yahoo Finance menggunakan sesi curl_cffi."""
    st.info("Mengambil data terbaru dari yfinance...")
    try:
        # Membuat session langsung dari curl_cffi dengan impersonate
        # Ini cara yang lebih modern dan sederhana
        session = Session(impersonate="chrome110")

        data = yf.download(
            tickers=ticker,
            period="250d",
            auto_adjust=True,
            progress=False,
            session=session # Gunakan session ini
        )

        if data.empty:
            st.error(f"Tidak ada data yang diterima dari yfinance untuk {ticker}.")
            return None
        
        st.success("Data berhasil diambil dari yfinance.")
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data dari yfinance. Kesalahan: {e}")
        return None

def create_features(df):
    """Membuat fitur-fitur teknikal dari data harga."""
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
    """Menjalankan proses prediksi berdasarkan model yang dipilih."""
    if model_code == "lstm_model":
        model = assets['lstm_model']
        scaler = assets['lstm_scaler']
        if len(raw_data) < LSTM_LOOKBACK:
            st.warning(f"Data tidak cukup untuk LSTM ({len(raw_data)}/{LSTM_LOOKBACK} hari).")
            return None
        
        latest_prices = raw_data['Close'].iloc[-LSTM_LOOKBACK:].values.reshape(-1, 1)
        latest_scaled = scaler.transform(latest_prices)
        input_lstm = np.reshape(latest_scaled, (1, LSTM_LOOKBACK, 1))
        prediction_scaled = model.predict(input_lstm)
        return scaler.inverse_transform(prediction_scaled)[0][0]
    else: # "best_model"
        model = assets['best_model']
        scaler = assets['feature_scaler']
        feature_columns = assets['feature_columns']
        
        feature_data = create_features(raw_data.copy())
        if feature_data.empty:
            st.warning("Tidak cukup data untuk membuat fitur (diperlukan > 90 hari).")
            return None
            
        latest_input_df = feature_data[feature_columns].iloc[-1:]
        input_scaled = scaler.transform(latest_input_df)
        return model.predict(input_scaled)[0]

def display_prediction_results(prediction_result, raw_data, model_display_name):
    """Menampilkan metrik dan grafik hasil prediksi."""
    history_df = raw_data.tail(90)
    prediction_date = history_df.index[-1].to_pydatetime() + timedelta(days=1)
    
    current_price = history_df['Close'].iloc[-1]
    prediction = prediction_result
    
    price_change = prediction - current_price
    pct_change = (price_change / current_price) * 100 if current_price != 0 else 0

    st.subheader("Hasil Prediksi untuk Esok Hari")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir", f"${current_price:,.2f}", f"per {history_df.index[-1].strftime('%d %b %Y')}")
    col2.metric("Prediksi Harga Besok", f"${prediction:,.2f}", f"${price_change:+.2f} ({pct_change:+.2f}%)")
    col3.metric("Model Digunakan", model_display_name)
    
    st.subheader("Visualisasi Harga")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df['Close'], mode='lines', name='Harga Historis',
        line=dict(color='royalblue', width=2), fill='tozeroy', fillcolor='rgba(65,105,225,0.1)',
        hovertemplate='Tanggal: %{x|%d %b %Y}<br>Harga: $%{y:,.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[prediction_date], y=[prediction], mode='markers', name='Prediksi Harga',
        marker=dict(color='orange', size=14, symbol='star', line=dict(width=2, color='darkorange')),
        hovertemplate=f"<b>Prediksi {prediction_date.strftime('%d %b %Y')}</b><br>Harga: ${prediction:,.2f}<extra></extra>"
    ))
    
    fig.add_annotation(
        x=prediction_date, y=prediction, text="Prediksi Besok", showarrow=True, arrowhead=2,
        arrowsize=1, arrowwidth=2, arrowcolor="orange", ax=0, ay=-60,
        bgcolor="rgba(255,165,0,0.8)", font=dict(color="black")
    )

    fig.update_layout(
        title=dict(
            text='📈 Pergerakan Harga Bitcoin 90 Hari Terakhir & Prediksi Harga Besok',
            x=0.5, xanchor='center', font=dict(size=20)
        ),
        xaxis_title='Tanggal', yaxis_title='Harga (USD)', template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAMPILAN STREAMLIT UTAMA
# =============================================================================
def main():
    st.title("₿ Prediksi & Analisis Harga Bitcoin (BTC-USD)")
    st.markdown("Aplikasi interaktif untuk memprediksi harga penutupan Bitcoin.")

    assets = load_all_assets()
    if not assets:
        st.stop()

    st.sidebar.header("Opsi Prediksi")
    model_options = {
        "Model Terbaik (Regresi Linear)": "best_model",
        "Model Jangka Panjang (LSTM)": "lstm_model"
    }
    selected_model_display = st.sidebar.selectbox(
        "Pilih Model:",
        options=list(model_options.keys())
    )
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi", type="primary"):
        with st.spinner("Mengambil data & melakukan prediksi..."):
            raw_data_pred = load_data()
            if raw_data_pred is not None:
                prediction = run_prediction(assets, raw_data_pred, selected_model_code)
                st.session_state['prediction_result'] = prediction
                st.session_state['raw_data'] = raw_data_pred
                st.session_state['model_name'] = selected_model_display
            else:
                st.error("Gagal prediksi karena data tidak dapat diambil.")
                st.session_state.pop('prediction_result', None)

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini bukan merupakan nasihat keuangan.")

    st.divider()

    if 'prediction_result' in st.session_state and st.session_state['prediction_result'] is not None:
        st.success("Prediksi berhasil dibuat!")
        display_prediction_results(
            st.session_state['prediction_result'],
            st.session_state['raw_data'],
            st.session_state['model_name']
        )
    else:
        st.info("Pilih model di sidebar & klik tombol 'Lakukan Prediksi' untuk memulai.")
        with st.spinner("Memuat data historis..."):
            raw_data_hist = load_data()
            if raw_data_hist is not None:
                st.subheader("Pergerakan Harga Bitcoin (90 Hari Terakhir)")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=raw_data_hist.index.tail(90), y=raw_data_hist['Close'].tail(90),
                    mode='lines', name='Harga Historis', line=dict(color='royalblue', width=2),
                    fill='tozeroy', fillcolor='rgba(65,105,225,0.1)'
                ))
                fig_hist.update_layout(
                    title='Grafik Harga Historis BTC-USD',
                    xaxis_title='Tanggal', yaxis_title='Harga (USD)', template='plotly_white'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Tidak dapat menampilkan grafik karena gagal memuat data.")

if __name__ == '__main__':
    main()
