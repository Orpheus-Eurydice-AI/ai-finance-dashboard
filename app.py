import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Page config
st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis with 7-day AI price forecast*")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)", "NVDA").upper().strip()
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        data = yf.Ticker(ticker).history(period=period)
        if data.empty or len(data) < 7:
            st.error("Invalid ticker or insufficient data. Try **NVDA**, **AAPL**, or **TSLA**.")
        else:
            prices = data['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            
            # Train AI
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices)+7).reshape(-1, 1)
            pred = model.predict(future)
            
            # Plot
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(data.index, prices, label="Actual Price", color='#1f77b4', linewidth=2.5)
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
            ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--', linewidth=3)
            ax.set_title(f"{ticker} Stock Price + AI Prediction", fontsize=16, fontweight='bold')
            ax.set_ylabel("Price ($)", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Prediction result
            current = prices[-1]
            forecast = pred[-1]
            change = forecast - current
            pct = (change / current) * 100
            if change > 0:
                st.success(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** (from ${current:.2f}) → **+${change:.2f} (+{pct:.1f}%)**")
            else:
                st.warning(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** (from ${current:.2f}) → **${change:.2f} ({pct:.1f}%)**")

st.caption("Built by Jack Evans | Moorpark College AS-T Business Admin | Nov 2025")
