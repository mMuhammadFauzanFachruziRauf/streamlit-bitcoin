import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # <- Pastikan impor ini ada
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
    """Memuat semua aset yang diperlukan seperti model dan scaler."""
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
        st.error(f"File tidak ditemukan: {e.filename}. Pastikan folder 'model' dan semua isinya ada.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat aset model: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data(ticker="BTC-USD"):
    """Mengambil data historis dari Yahoo Finance."""
    st.info("Mengambil data terbaru dari yfinance...")
    try:
        data = yf.download(
            tickers=ticker,
            period="200d",  # Ambil 200 hari agar MA 90 hari bisa dihitung dengan baik
            auto_adjust=True,
            progress=False
        )
        if data.empty:
            st.error(f"Tidak ada data dari yfinance untuk ticker {ticker}.")
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

# =============================================================================
# TAMPILAN STREAMLIT
# =============================================================================

st.title("₿ Prediksi & Analisis Harga Bitcoin (BTC-USD)")
st.markdown("Aplikasi interaktif untuk memprediksi harga penutupan Bitcoin esok hari menggunakan model Machine Learning.")

assets = load_all_assets()

if assets:
    st.sidebar.header("Opsi Prediksi")
    model_options = {
        "Model Terbaik (Regresi Linear)": "best_model",
        "Model LSTM (Jaringan Saraf Tiruan)": "lstm_model"
    }
    selected_model_display = st.sidebar.selectbox(
        "Pilih Model untuk Prediksi:",
        options=list(model_options.keys())
    )
    selected_model_code = model_options[selected_model_display]

    if st.sidebar.button("🚀 Lakukan Prediksi Harga Besok"):
        raw_data = load_data()

        if raw_data is not None and len(raw_data) > 90:
            with st.spinner("Membuat fitur dan melakukan prediksi... Harap tunggu sebentar."):
                feature_data = create_features(raw_data.copy())

                if not feature_data.empty:
                    prediction = 0.0
                    
                    # Logika Prediksi berdasarkan model yang dipilih
                    if selected_model_code == "lstm_model":
                        model = assets['lstm_model']
                        scaler = assets['lstm_scaler']
                        lookback = 60 # Sesuai dengan input shape model LSTM
                        if len(raw_data) >= lookback:
                            latest_prices = raw_data['Close'].iloc[-lookback:].values.reshape(-1, 1)
                            latest_scaled = scaler.transform(latest_prices)
                            input_lstm = np.reshape(latest_scaled, (1, lookback, 1))
                            prediction_scaled = model.predict(input_lstm)
                            prediction = scaler.inverse_transform(prediction_scaled)[0][0]
                        else:
                            st.warning(f"Data tidak cukup untuk LSTM ({len(raw_data)}/{lookback} data).")
                            st.stop()
                    
                    else: # Model Regresi
                        model = assets['best_model']
                        scaler = assets['feature_scaler']
                        feature_columns = assets['feature_columns']
                        latest_input_df = feature_data[feature_columns].iloc[-1:]
                        input_scaled = scaler.transform(latest_input_df)
                        prediction = model.predict(input_scaled)[0]
                    
                    st.success("Prediksi berhasil dibuat!")
                    
                    # Persiapan data untuk ditampilkan
                    history_df = raw_data.tail(90)
                    prediction_date = history_df.index[-1].to_pydatetime() + timedelta(days=1)

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
                    with col3:
                         st.info(f"Model: **{selected_model_display}**")
                    
                    # =============================================================================
                    # VISUALISASI HARGA (VERSI ELEGAN)
                    # =============================================================================
                    st.subheader("Visualisasi Analisis Teknikal & Prediksi")

                    # Kita butuh fitur MA untuk grafik, jadi kita hitung dari data mentah
                    chart_features = create_features(raw_data.copy())

                    # Inisialisasi gambar dengan 2 baris: 1 untuk harga, 1 untuk volume
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.75, 0.25]
                    )

                    # --- BARIS 1: GRAFIK HARGA ---
                    fig.add_trace(go.Candlestick(
                        x=history_df.index,
                        open=history_df['Open'], high=history_df['High'],
                        low=history_df['Low'], close=history_df['Close'],
                        name='Harga (OHLC)',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=chart_features.index, y=chart_features['MA_7'],
                        mode='lines', name='MA 7 Hari',
                        line=dict(color='orange', width=1.5)
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=chart_features.index, y=chart_features['MA_30'],
                        mode='lines', name='MA 30 Hari',
                        line=dict(color='dodgerblue', width=1.5)
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=[prediction_date], y=[prediction],
                        mode='markers', name='Prediksi',
                        marker=dict(
                            color='#FFD700', size=18, symbol='star',
                            line=dict(width=2, color='darkorange')
                        ),
                        hovertemplate=f"<b>Prediksi:</b><br>${prediction:,.2f}<extra></extra>"
                    ), row=1, col=1)

                    fig.add_annotation(
                        x=prediction_date, y=prediction,
                        text=f"<b>Prediksi:</b><br>${prediction:,.2f}",
                        showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2,
                        arrowcolor="darkorange", ax=70, ay=-50,
                        bordercolor="darkorange", borderwidth=2, borderpad=4,
                        bgcolor="rgba(255, 255, 224, 0.9)",
                        font=dict(size=12, color="black"),
                        row=1, col=1
                    )

                    # --- BARIS 2: GRAFIK VOLUME ---
                    fig.add_trace(go.Bar(
                        x=history_df.index, y=history_df['Volume'],
                        name='Volume', marker_color='rgba(150, 150, 150, 0.6)'
                    ), row=2, col=1)

                    # --- PENGATURAN LAYOUT GLOBAL ---
                    fig.update_layout(
                        title=dict(
                            text='📈 <b>Analisis & Prediksi Harga Bitcoin (BTC-USD)</b>',
                            y=0.95, x=0.5, xanchor='center', yanchor='top',
                            font=dict(size=22, family="Arial, sans-serif")
                        ),
                        xaxis_rangeslider_visible=False,
                        template='plotly_white',
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1
                        ),
                        margin=dict(l=50, r=50, t=100, b=20),
                        hovermode='x unified',
                        yaxis1_title="Harga (USD)",
                        yaxis2_title="Volume",
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        yaxis2_showgrid=False,
                    )

                    fig.update_xaxes(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1bln", step="month", stepmode="backward"),
                                dict(count=3, label="3bln", step="month", stepmode="backward"),
                                dict(step="all", label="Semua")
                            ]),
                            y=1.1,
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("Tidak cukup data setelah pembuatan fitur untuk melakukan prediksi.")
        
        elif raw_data is not None:
            st.warning(f"Tidak cukup data historis (diperlukan > 90 hari, tersedia {len(raw_data)} hari).")

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini dibuat untuk tujuan edukasi dan demonstrasi, bukan merupakan nasihat keuangan.")
else:
    st.error("Aplikasi tidak dapat dijalankan karena aset model gagal dimuat. Pastikan folder 'model' ada dan berisi semua file yang diperlukan.")
