import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --------------------- Ambil data dari Yahoo Finance ---------------------
@st.cache_data
def load_data():
    df = yf.download("BTC-USD", period="100d", interval="1d")
    df.reset_index(inplace=True)
    return df

# --------------------- Training model ---------------------
def train_models(df):
    df = df[['Date', 'Close']].dropna()
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

    X = df[['Date_ordinal']]
    y = df['Close']

    linear_model = LinearRegression().fit(X, y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

    return linear_model, rf_model

# --------------------- Prediksi harga besok ---------------------
def predict_tomorrow(df, model):
    last_date = pd.to_datetime(df['Date'].max())
    tomorrow_date = last_date + timedelta(days=1)
    tomorrow_ordinal = np.array([[tomorrow_date.toordinal()]])
    predicted_price = model.predict(tomorrow_ordinal)[0]
    return tomorrow_date, predicted_price

# --------------------- Visualisasi Plotly ---------------------
def plot_price(df, pred_date, pred_price):
    fig = go.Figure()

    # Harga historis
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], mode='lines', name='Harga Historis',
        line=dict(color='blue')
    ))

    # Titik prediksi
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[pred_price], mode='markers+text', name='Prediksi Besok',
        marker=dict(color='orange', size=12, symbol='star'),
        text=["Prediksi"], textposition="top center"
    ))

    fig.update_layout(
        title="Pergerakan Harga Bitcoin & Prediksi Harga Besok",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    return fig

# --------------------- UI Streamlit ---------------------
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📈 Visualisasi Harga Bitcoin")

# Sidebar
st.sidebar.header("🧠 Opsi Prediksi")
model_option = st.sidebar.selectbox("Pilih Model untuk Prediksi:",
                                    ("Model Terbaik (Regresi Linear)", "Random Forest"))
predict_button = st.sidebar.button("🚀 Lakukan Prediksi Harga Besok")

# Load data
df = load_data()
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

st.subheader("📊 Data Harga Bitcoin (90 Hari Terakhir)")
st.dataframe(df.tail(90), use_container_width=True)

# Jika tombol ditekan
if predict_button:
    linear_model, rf_model = train_models(df)
    model = linear_model if model_option == "Model Terbaik (Regresi Linear)" else rf_model
    pred_date, pred_price = predict_tomorrow(df, model)

    st.subheader("📍 Prediksi Harga Besok")
    st.markdown(f"📅 Tanggal: `{pred_date.date()}`")
    st.markdown(f"💰 Prediksi Harga: `${pred_price:,.2f}`")

    # Tampilkan grafik
    fig = plot_price(df.tail(90), pred_date, pred_price)
    st.plotly_chart(fig, use_container_width=True)
