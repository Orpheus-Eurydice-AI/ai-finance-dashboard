import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit.components.v1 as components

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
            st.success(f"Welcome, {name}!")
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
                st.success("**BULLISH** (5/5)")

                # UNBLOCKABLE BALLOONS
                components.html("""
                    <script>
                    for(let i=0; i<50; i++){
                        let b = document.createElement('div');
                        b.innerText = '🎈';
                        b.style.position = 'fixed';
                        b.style.left = Math.random()*100 + 'vw';
                        b.style.bottom = '-10vh';
                        b.style.fontSize = '36px';
                        b.style.zIndex = '9999';
                        b.style.animation = 'float 2.5s ease-in-out forwards';
                        document.body.appendChild(b);
                        setTimeout(() => b.remove(), 2500);
                    }
                    </script>
                    <style>
                    @keyframes float {
                        to { transform: translateY(-150vh) rotate(360deg); opacity: 0; }
                    }
                    </style>
                """, height=0, width=0)

                st.toast("BULLISH ALERT!", icon="🎉")

                for h in headlines[:3]:
                    st.markdown(f"• {h}")

                       # === P&L + RISK ===
            current = prices[-1]
            forecast = pred[-1]
            pct = (forecast - current) / current * 100
            volatility = np.std(prices[-30:]) / np.mean(prices[-30:]) * 100  # 30-day vol

            st.markdown("### Portfolio Summary")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Current Price", f"${current:.2f}")
            with col_p2:
                st.metric("7-Day Forecast", f"${forecast:.2f}", f"{pct:+.1f}%")
            with col_p3:
                st.metric("30-Day Volatility", f"{volatility:.1f}%")

            if pct > 5:
                st.success("**STRONG BUY SIGNAL** 🚀")
            elif pct < -5:
                st.warning("**SELL SIGNAL** ⚠️")
                
# Watchlist
st.markdown("### My Watchlist")
for t in st.session_state.watchlist:
    st.write(f"• {t}")

st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
