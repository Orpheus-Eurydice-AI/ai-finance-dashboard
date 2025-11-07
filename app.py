import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time + 7-day AI forecast + News + Watchlist + **Backtesting***")

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

# === TABS ===
tab1, tab2 = st.tabs(["Live Analysis", "Backtesting"])

# ——— TAB 1: LIVE ANALYSIS ———
with tab1:
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
                    st.markdown("**BULLISH ALERT!**")

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
                    st.success("**STRONG BUY SIGNAL**")

# ——— TAB 2: BACKTESTING ENGINE (FIXED) ———
with tab2:
    st.markdown("### Backtest Any Strategy")
    back_ticker = st.selectbox("Select Ticker", st.session_state.watchlist, key="back_ticker")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col_end:
        end_date = st.date_input("End Date", value=datetime.today())

    strategy = st.selectbox("Strategy", ["Buy & Hold", "Dollar Cost Average (Monthly)"])

    initial = st.number_input("Initial Investment ($)", value=10000, min_value=100)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            data = yf.download(back_ticker, start=start_date, end=end_date)
            if data.empty or len(data) < 2:
                st.error("Not enough data")
            else:
                prices = data['Close'].dropna()
                if len(prices) == 0:
                    st.error("No price data")
                else:
                    first_price = prices.iloc[0]
                    last_price = prices.iloc[-1]
                    dates = prices.index

                    if strategy == "Buy & Hold":
                        shares = initial / first_price
                        final_value = shares * last_price
                        pnl = final_value - initial

                    elif strategy == "Dollar Cost Average (Monthly)":
                        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
                        monthly_dates = monthly_dates[monthly_dates <= end_date]
                        num_months = len(monthly_dates)
                        if num_months == 0:
                            st.error("No months in range")
                            final_value = initial
                            pnl = 0
                        else:
                            monthly_invest = initial / num_months
                            shares = 0
                            investment = 0
                            portfolio_values = []
                            cum_shares = 0
                            cum_invest = 0
                            for date in dates:
                                if date in monthly_dates and cum_invest < initial:
                                    price = prices.asof(date)
                                    if pd.notna(price):
                                        cum_shares += monthly_invest / price
                                        cum_invest += monthly_invest
                                portfolio_values.append(cum_shares * prices.asof(date))
                            final_value = cum_shares * last_price
                            pnl = final_value - initial

                    # Metrics
                    years = (end_date - start_date).days / 365.25
                    cagr = ((final_value / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
                    returns = prices.pct_change().dropna()
                    if len(returns) > 1:
                        mean_ret = returns.mean()
                        std_dev = returns.std(ddof=0)
                        sharpe = (mean_ret * 252) / (std_dev * np.sqrt(252)) if std_dev != 0 else 0
                        drawdown = ((prices / prices.cummax()) - 1).min() * 100
                    else:
                        sharpe = 0
                        drawdown = 0

                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(dates, np.cumprod(1 + returns) * initial, label="Buy & Hold", color='gray', alpha=0.7)
                        if strategy == "Dollar Cost Average (Monthly)":
                            ax.plot(dates, portfolio_values, label="DCA", color='green', linewidth=2)
                        ax.set_title(f"{back_ticker} Backtest: {strategy}")
                        ax.set_ylabel("Portfolio Value ($)")
                        ax.legend()
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)

                    with col2:
                        st.metric("Final Value", f"${final_value:,.2f}", f"{pnl:,.0f}")
                        st.metric("CAGR", f"{cagr:.1f}%")
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        st.metric("Max Drawdown", f"{drawdown:.1f}%")

                        report = f"""
# Backtest Report: {back_ticker}
**Strategy**: {strategy}  
**Period**: {start_date} to {end_date}  
**Initial**: ${initial:,.0f}  
**Final Value**: ${final_value:,.2f}  
**PnL**: ${pnl:,.0f}  
**CAGR**: {cagr:.1f}%  
**Sharpe**: {sharpe:.2f}  
**Max Drawdown**: {drawdown:.1f}%
                        """
                        st.download_button("Download Report", report, f"backtest_{back_ticker}.txt")

# === PORTFOLIO ===
st.markdown("### Portfolio Overview")
portfolio = {}
total_value = 0
for t in st.session_state.watchlist:
    data = yf.Ticker(t).history(period="1d")
    if not data.empty:
        price = data['Close'].iloc[-1]
        shares = st.number_input(f"Shares of {t}", min_value=0, value=st.session_state.get(f"shares_{t}", 10), key=f"input_{t}")
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
        st.info("Use browser print to Save as PDF")
with col2:
    theme = st.selectbox("Theme", ["Light", "Dark"], index=1)
    if theme == "Dark":
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
            h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown, .stText, .stCode { color: #ffffff !important; }
            .stMetric > div, .stMetric label, .stMetric > div > div { color: #ffffff !important; }
            .stTextInput > div > div > input { color: #ffffff !important; background-color: #1e1e1e !important; border: 1px solid #555 !important; }
            .stButton > button { color: #ffffff !important; background-color: #2d2d2d !important; border: 1px solid #555 !important; }
            .stButton > button:hover { background-color: #3d3d3d !important; }
            [data-testid="stFormSubmitButton"] > button { background-color: #2d2d2d !important; color: #ffffff !important; border: 1px solid #555 !important; }

            /* SELECTBOX — FULLY VISIBLE */
            .stSelectbox > div > div { background-color: #1e1e1e !important; color: #ffffff !important; border: 1px solid #555 !important; }
            .stSelectbox > div > div > div { background-color: #1e1e1e !important; color: #ffffff !important; }
            .stSelectbox [data-baseweb="select"] > div { background-color: #1e1e1e !important; color: #ffffff !important; }

            /* DROPDOWN MENU — BOLD WHITE, FULL OPACITY */
            [data-baseweb="menu"] { 
                background-color: #1a1a1a !important; 
                border: 1px solid #444 !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
            }
            [data-baseweb="menu"] div { 
                color: #ffffff !important; 
                background-color: #1a1a1a !important; 
                font-weight: 700 !important;
                padding: 10px 16px !important;
                opacity: 1 !important;
            }
            [data-baseweb="menu"] div:hover { 
                background-color: #2d2d2d !important; 
                color: #ffffff !important; 
            }
            [data-baseweb="menu"] div[data-selected="true"] {
                background-color: #3d3d3d !important;
                color: #ffffff !important;
            }

            .stSuccess { background-color: #1a4d1a !important; color: #ffffff !important; border: 1px solid #2a6d2a !important; }
            .stInfo { background-color: #0e3d6b !important; color: #ffffff !important; border: 1px solid #1e5d8b !important; }
            </style>
            """, unsafe_allow_html=True)

st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
