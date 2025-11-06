import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from textblob import TextBlob

# Page config
st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis + 7-day AI forecast + News Sentiment*")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)", "NVDA").upper().strip()
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
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

            # === MOCK NEWS ===
            headlines = [
                f"{ticker} surges on strong earnings",
                f"Analysts raise price target for {ticker}",
                f"Market volatility impacts {ticker}",
                f"{ticker} beats revenue expectations",
                f"Investors cautious on {ticker} outlook"
            ]
            sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_sentiment = np.mean(sentiments)
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
               # === INSIDE col2 (replace old sentiment block) ===
st.markdown("### News Sentiment")
if avg_sentiment > 0.1:
    st.success(f"**BULLISH** ({positive_count}/5) 🎈")
    st.toast("BULLISH ALERT! 🎉", icon="🎉")
    # Force Chrome balloons
    st.markdown(
        """
        <script>
        const balloons = () => {
            for(let i=0; i<30; i++){
                const b = document.createElement('div');
                b.innerText = '🎈';
                b.style.position = 'fixed';
                b.style.left = Math.random()*100 + 'vw';
                b.style.bottom = '-10vh';
                b.style.fontSize = '30px';
                b.style.zIndex = '9999';
                b.style.animation = 'float 4s ease-in-out forwards';
                document.body.appendChild(b);
                setTimeout(() => b.remove(), 4000);
            }
        };
        balloons();
        </script>
        <style>
        @keyframes float {
            to { transform: translateY(-120vh) rotate(360deg); opacity: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.info(f"**Neutral** ({positive_count}/5)")

            # === PREDICTION ===
            current = prices[-1]
            forecast = pred[-1]
            change = forecast - current
            pct = (change / current) * 100
            if change > 0:
                st.success(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** (+{pct:.1f}%)")
            else:
                st.warning(f"**AI Predicts {ticker} in 7 days: ${forecast:.2f}** ({pct:.1f}%)")

st.caption("Built by Jack Evans | Moorpark College AS-T Business Admin | Nov 2025")
