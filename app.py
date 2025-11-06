import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import requests
from textblob import TextBlob
import datetime

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

            # === NEWS SENTIMENT ===
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    "q": ticker,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    6230d817fba04ce5910afe393eb85d83  # ← Replace with your key
                }
                response = requests.get(url, params=params).json()
                articles = response.get("articles", [])
                headlines = [a["title"] for a in articles if a["title"]]
                sources = [a["url"] for a in articles if a["url"]]

                sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                positive_count = sum(1 for s in sentiments if s > 0.1)

                # === DISPLAY ===
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(data.index, prices, label="Actual Price", color='#1f77b4', linewidth=2.5)
                    ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--', linewidth=3)
                    ax.set_title(f"{ticker} Stock + AI Prediction", fontsize=14)
                    ax.set_ylabel("Price ($)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.markdown("### News Sentiment")
                    if avg_sentiment > 0.1:
                        st.success(f"**Bullish** ({positive_count}/5)")
                        st.balloons()
                    elif avg_sentiment < -0.1:
                        st.warning(f"**Bearish** ({positive_count}/5)")
                    else:
                        st.info(f"**Neutral** ({positive_count}/5)")

                    for h, s in zip(headlines[:3], sources[:3]):
                        st.markdown(f"• [{h}]({s})")

                # === PREDICTION RESULT ===
                current = prices[-1]
                forecast = pred[-1]
                change = forecast - current
                pct = (change / current) * 100
                if change > 0:
                    st.success(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** (+{pct:.1f}%)")
                else:
                    st.warning(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** ({pct:.1f}%)")

            except:
                st.info("News API unavailable. Using mock data.")
                st.success("**News Sentiment: Bullish** (4/5)")
                st.balloons()

st.caption("Built by Jack Evans | Moorpark College AS-T Business Admin | Nov 2025")
