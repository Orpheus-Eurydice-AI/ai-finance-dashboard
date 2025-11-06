import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import requests
from textblob import TextBlob

# Page config
st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis + 7-day AI forecast + News Sentiment*")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)", "NVDA").upper().strip()
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.button("Analyze"):
    with st.spinner("Fetching data & news..."):
        # === STOCK DATA ===
        data = yf.Ticker(ticker).history(period=period)
        if data.empty or len(data) < 7:
            st.error("Invalid ticker. Try **NVDA**, **AAPL**, or **TSLA**.")
        else:
            prices = data['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices)+7).reshape(-1, 1)
            pred = model.predict(future)
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)

            # === NEWS SENTIMENT (FREE MOCK DATA) ===
            headlines = [
                f"{ticker} surges on strong
