import streamlit as st  # For building the web app interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting charts
from textblob import TextBlob  # For sentiment analysis
from sklearn.linear_model import LinearRegression  # For simple forecasting
from datetime import datetime, timedelta, date  # For date handling
import yfinance as yf  # For fetching financial data
import streamlit.components.v1 as components  # For custom HTML components
from pycoingecko import CoinGeckoAPI  # For crypto data

# Initialize CoinGecko API
cg = CoinGeckoAPI()

def get_crypto_data(coin_id, days=30):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock/crypto analysis + 7-day AI forecast + News Sentiment + Watchlist*")

# === LOGIN ===
if "user" not in st.session_state:
    st.session_state.user = None
if st.session_state.user is None:
    with st.form("login"):
        name = st.text_input("Your Name")
        submitted = st.form_submit_button("Login")
        if submitted and name:
            st.session_state.user = name
            st.success(f"Welcome, {name}!")
            st.rerun()
else:
    st.success(f"Logged in as **{st.session_state.user}**")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

# === WATCHLIST ===
if "watchlist" not in st.session_state:
    st.session_state.watchlist = [("Stock", "NVDA"), ("Stock", "AAPL")]

# Asset type select for input
asset_type = st.selectbox('Asset Type', ['Stock', 'Crypto'])

if asset_type == 'Stock':
    asset_id = st.text_input("Enter Stock Ticker (e.g., NVDA)", "NVDA").upper().strip()
else:
    asset_id = st.text_input("Enter Crypto ID (e.g., bitcoin)", "bitcoin").lower().strip()

if st.button("Add to Watchlist"):
    if asset_id and (asset_type, asset_id) not in st.session_state.watchlist:
        st.session_state.watchlist.append((asset_type, asset_id))
        st.success(f"{asset_id} ({asset_type}) added!")

# === ANALYZE ===
st.header("Analyze Asset")
if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        try:
            if asset_type == 'Stock':
                data = yf.download(asset_id, period="3mo")
                if data.empty or len(data) < 7:
                    raise ValueError("Invalid ticker or insufficient data.")
                prices = data['Close'].values
                dates = data.index
            else:  # Crypto
                df = get_crypto_data(asset_id, days=90)  # ~3 months
                if df.empty or len(df) < 7:
                    raise ValueError("Invalid coin ID or insufficient data.")
                prices = df['price'].values
                dates = df['timestamp']

            days = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
            pred = model.predict(future)
            future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, prices, label="Actual Price", color='blue')
            ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--')
            ax.set_title(f"{asset_id} ({asset_type})
st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
