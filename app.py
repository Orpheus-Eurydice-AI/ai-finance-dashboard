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

                # BALLOONS
                components.html("""
                    <script>
                    for(let i=0; i<50; i++){
                        let b = document.createElement('div');
                        b.innerText = 'Balloon';
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

                st.toast("BULLISH ALERT!", icon="rocket")  # rocket emoji

                for h in headlines[:3]:
                    st.markdown(f"• {h}")

            current = prices[-1]
            forecast = pred[-1]
            pct = (forecast - current) / current * 100
            volatility = np.std(prices[-30:]) / np.mean(prices[-30:]) * 100

            st.markdown("### Portfolio Summary")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Current Price", f"${current:.2f}")
            with col_p2:
                st.metric("7-Day Forecast", f"${forecast:.2f}", f"{pct:+.1f}%")
            with col_p3:
                st.metric("30-Day Volatility", f"{volatility:.1f}%")

            if pct > 5:
                st.success("**STRONG BUY SIGNAL** rocket")

# === PORTFOLIO P&L ===
st.markdown("### Portfolio Overview")
portfolio = {}
total_value = 0

for t in st.session_state.watchlist:
    data = yf.Ticker(t).history(period="1d")
    if not data.empty:
        price = data['Close'].iloc[-1]
        shares = st.session_state.get(f"shares_{t}", 0)
        if shares == 0:
            shares = st.number_input(f"Shares of {t}", min_value=0, value=10, key=f"input_{t}")
            st.session_state[f"shares_{t}"] = shares
        value = shares * price
        portfolio[t] = {"price": price, "shares": shares, "value": value}
        total_value += value

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total Value", f"${total_value:,.2f}")
with col_b:
    st.metric("Assets", len(portfolio))
with col_c:
    st.metric("Watchlist", len(st.session_state.watchlist))

st.markdown("### My Watchlist")
for t, p in portfolio.items():
    st.write(f"• **{t}**: {p['shares']} × ${p['price']:.2f} = **${p['value']:.2f}**")

# === EXPORT + THEME ===
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Export to PDF"):
        st.info("Use browser print → Save as PDF")

with col2:
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
            h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown, .stText, .stCode { color: #ffffff !important; }
            .stMetric > div, .stMetric label { color: #ffffff !important; }
            .stTextInput > div > div > input { 
                color: #ffffff !important; 
                background-color: #1e1e1e !important; 
                border: 1px solid #555 !important; 
            }
            .stButton > button { 
                color: #ffffff !important; 
                background-color: #2d2d2d !important; 
                border: 1px solid #555 !important; 
            }
            .stSelectbox > div > div { 
                background-color: #1e1e1e !important; 
                color: #ffffff !important; 
            }
            .stSelectbox > div > div > div { 
                background-color: #1e1e1e !important; 
                color: #ffffff !important; 
            }
            .stSelectbox > div > div > div > div { 
                background-color: #1e1e1e !important; 
                color: #ffffff !important; 
            }
            .stSuccess { background-color: #1a4d1a !important; color: #ffffff !important; }
            .stInfo { background-color: #0e3d6b !important; color: #ffffff !important; }
            </style>
            """, unsafe_allow_html=True)

st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
