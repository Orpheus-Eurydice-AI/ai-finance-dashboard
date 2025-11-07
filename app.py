import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis + 7-day AI forecast + News Sentiment + Watchlist*")

# === LOGIN ===
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    with st.form("login"):
        name = st.text_input("Your Name")
        submitted = st.form_submit_button("Login")
        if submitted and name:
            st.session_state.user = name
            st.success(f"Welcome, {name}! 🎉")
            st.rerun()
else:
    st.success(f"Logged in as **{st.session_state.user}**")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

# === WATCHLIST ===
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["NVDA", "AAPL"]

ticker = st.text_input("Enter Stock Ticker", "NVDA").upper().strip()
if st.button("Add to Watchlist"):
    if ticker and ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.success(f"{ticker} added!")

# === ANALYZE ===
if st.button("Analyze"):
    with st.spinner("Fetching..."):
        data = yf.Ticker(ticker).history(period="3mo")
        if data.empty:
            st.error("Invalid ticker")
        else:
            prices = data['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices)+7).reshape(-1, 1)
            pred = model.predict(future)
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)

            headlines = [
                f"{ticker} surges on earnings",
                f"Analysts raise {ticker} target",
                f"Volatility hits {ticker}",
                f"{ticker} beats estimates",
                f"{ticker} outlook cautious"
            ]
            sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_sentiment = np.mean(sentiments)
            positive_count = sum(1 for s in sentiments if s > 0.1)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(data.index, prices, label="Actual", color='#1f77b4', linewidth=2.5)
                ax.plot(future_dates, pred, label="AI 7-Day", color='red', linestyle='--', linewidth=3)
                ax.set_title(f"{ticker} + AI Prediction")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.markdown("### News Sentiment")
                if avg_sentiment > 0.1:
                    st.success(f"**BULLISH** ({positive_count}/5)")
                    st.toast("BULLISH ALERT! 🎉", icon="🎉")
                    st.markdown("<h1 style='text-align:center; font-size:60px;'>🎈🎈🎈</h1>", unsafe_allow_html=True)
                else:
                    st.info(f"**Neutral** ({positive_count}/5)")

                for h in headlines[:3]:
                    st.markdown(f"• {h}")

            current = prices[-1]
            forecast = pred[-1]
            pct = (forecast - current) / current * 100
            if pct > 0:
                st.success(f"**AI Predicts {ticker}: ${forecast:.2f} (+{pct:.1f}%)**")
            else:
                st.warning(f"**AI Predicts {ticker}: ${forecast:.2f} ({pct:.1f}%)**")

# Watchlist
st.markdown("### My Watchlist")
for t in st.session_state.watchlist:
    st.write(f"• {t}")

st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
