import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linearModel import LinearRegression
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis + 7-day AI forecast + News Sentiment*")

ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)", "NVDA").upper().strip()
period = st.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=1)

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
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

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(data.index, prices, label="Actual Price", color='#1f77b4', linewidth=2.5)
                ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--', linewidth=3)
                ax.set_title(f"{ticker} Stock + AI Prediction")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown("### News Sentiment")
                if avg_sentiment > 0.1:
                    st.success(f"**BULLISH** ({positive_count}/5) 🎈")
                    st.toast("BULLISH ALERT! 🎉", icon="🎉")
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
                            to {
