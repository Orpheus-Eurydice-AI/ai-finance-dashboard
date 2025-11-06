import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")

st.markdown("**Real-time stock analysis + 7-day AI price prediction**")

ticker = st.text_input("Enter Stock Ticker (e.g. NVDA, AAPL)", "NVDA").upper()
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"])

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            st.error("Invalid ticker! Try NVDA, AAPL, TSLA...")
        else:
            prices = data['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            
            # Train AI
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices)+7).reshape(-1, 1)
            pred = model.predict(future)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(data.index, prices, label="Actual Price", color='blue', linewidth=2)
            ax.plot(pd.date_range(start=data.index[-1], periods=8)[1:], pred, 
                    label="AI 7-Day Forecast", color='red', linestyle='--', linewidth=3)
            ax.set_title(f"{ticker} Stock + AI Prediction")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"**AI Predicts {ticker} in 7 days: ${pred[-1]:.2f}** (from ${prices[-1]:.2f} today)")

st.caption("Built by Jack Evans | Moorpark College AS-T Business Admin | Nov 2025")
