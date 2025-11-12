import streamlit as st  # For building the web app interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting charts
from textblob import TextBlob  # For sentiment analysis
from sklearn.linear_model import LinearRegression  # For simple forecasting
from datetime import datetime, timedelta, date  # For date handling
import yfinance as yf  # For fetching financial data
import streamlit.components.v1 as components  # For custom HTML components
from pycoingecko import CoinGeckoAPI  # For crypto data

# Initialize CoinGecko API
cg = CoinGeckoAPI()

def get_crypto_data(coin_id, days=30):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock/crypto analysis + 7-day AI forecast + News Sentiment + Watchlist*")
st.info("For crypto, use valid IDs like 'bitcoin', 'ethereum' (lowercase). Find IDs on coingecko.com.")

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
    st.session_state.watchlist = [("Stock", "NVDA"), ("Stock", "AAPL")]

# Asset type select for input
asset_type = st.selectbox('Asset Type', ['Stock', 'Crypto'])

if asset_type == 'Stock':
    asset_id = st.text_input("Enter Stock Ticker (e.g., NVDA)", "NVDA").upper().strip()
else:
    asset_id = st.text_input("Enter Crypto ID (e.g., bitcoin)", "bitcoin").lower().strip()

if st.button("Add to Watchlist"):
    if asset_id and (asset_type, asset_id) not in st.session_state.watchlist:
        st.session_state.watchlist.append((asset_type, asset_id))
        st.success(f"{asset_id} ({asset_type}) added!")

# === ANALYZE ===
st.header("Analyze Asset")
if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        try:
            if asset_type == 'Stock':
                data = yf.download(asset_id, period="3mo")
                if data.empty or len(data) < 7:
                    raise ValueError("Invalid ticker or insufficient data.")
                prices = data['Close'].values
                dates = data.index
            else:  # Crypto
                df = get_crypto_data(asset_id, days=90)  # ~3 months
                if df.empty or len(df) < 7:
                    raise ValueError("Invalid coin ID or insufficient data.")
                prices = df['price'].values
                dates = df['timestamp']

            days = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(days, prices)
            future = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
            pred = model.predict(future)
            future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, prices, label="Actual Price", color='blue')
            ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--')
            ax.set_title(f"{asset_id} ({asset_type}) Price + AI Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            current = float(prices[-1])
            forecast = float(pred[-1])
            change = forecast - current
            pct = (change / current) * 100 if current != 0 else 0.0
            pct = float(np.nan_to_num(pct))
            recent_prices = prices[-30:]
            volatility = float(np.nanstd(recent_prices) / np.nanmean(recent_prices) * 100) if len(recent_prices) > 0 else 0.0
            volatility = float(np.nan_to_num(volatility))
            st.markdown("### Forecast Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current:.2f}")
            with col2:
                st.metric("7-Day Forecast", f"${forecast:.2f}", f"{pct:+.1f}%")
            with col3:
                st.metric("30-Day Volatility", f"{volatility:.1f}%")
            # News Sentiment (stock only for now)
            if asset_type == 'Stock':
                news = yf.Ticker(asset_id).news
                headlines = [article.get('title', '') for article in news if 'title' in article]
                if headlines:
                    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
                    avg_sentiment = np.mean(sentiments)
                    positive_count = sum(1 for s in sentiments if s > 0.1)
                    st.markdown("### News Sentiment")
                    if avg_sentiment > 0.1:
                        st.success(f"Bullish ({positive_count}/{len(sentiments)})")
                        # Balloons
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
                    elif avg_sentiment < -0.1:
                        st.warning(f"Bearish ({positive_count}/{len(sentiments)})")
                    else:
                        st.info(f"Neutral ({positive_count}/{len(sentiments)})")
                    for h in headlines[:3]:
                        st.markdown(f"• {h}")
                else:
                    st.info("No recent news available.")
            else:
                st.info("News sentiment not available for crypto yet.")
            if pct > 5:
                st.success("STRONG BUY SIGNAL")
        except Exception as e:
            st.error(f"Error fetching data: {type(e).__name__}: {str(e)}. Check ID or network.")

# === BACKTESTING ===
st.header("Backtesting: Sentiment-Based Strategy (Stocks Only)")
if asset_type == 'Crypto':
    st.info("Backtesting available for stocks only.")
else:
    start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    end_date = st.date_input("End Date", value=date(2024, 12, 31))
    if st.button("Run Backtest"):
        try:
            data = yf.download(asset_id, start=start_date, end=end_date)
            if data.empty:
                st.error("No data available for the selected dates. Adjust the range or check the ticker.")
            else:
                # Ensure single-level columns
                data.columns = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else data.columns
                df = data[['Close']].rename(columns={'Close': 'close'})
                df = df.reset_index()  # Single-level with 'Date' as column
                # Fetch news
                news = yf.Ticker(asset_id).news
                daily_sent = {}
                for n in news:
                    if 'providerPublishTime' in n:
                        pub_date = datetime.fromtimestamp(n['providerPublishTime']).date()
                        polarity = TextBlob(n.get('title', '')).sentiment.polarity
                        if pub_date in daily_sent:
                            daily_sent[pub_date].append(polarity)
                        else:
                            daily_sent[pub_date] = [polarity]
                # Average sentiments
                for d in list(daily_sent.keys()):
                    daily_sent[d] = sum(daily_sent[d]) / len(daily_sent[d])
                df['sentiment'] = df['Date'].apply(lambda x: daily_sent.get(x.date(), 0))
                df['ret'] = df['close'].pct_change()
                df['strategy_ret'] = df['ret'] * (df['sentiment'].shift(1) > 0)
                df['strategy_cum'] = (1 + df['strategy_ret']).cumprod().fillna(1)
                df['buy_hold_cum'] = (1 + df['ret']).cumprod().fillna(1)
                # Rename for intuition
                df = df.rename(columns={
                    'close': 'Closing Price',
                    'sentiment': 'Daily Sentiment Score',
                    'strategy_cum': 'Strategy Cumulative Return',
                    'buy_hold_cum': 'Buy & Hold Cumulative Return'
                })
                # Display table with config
                st.subheader("Backtest Summary Table")
                column_config = {
                    "Closing Price": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Daily closing price of the stock"
                    ),
                    "Daily Sentiment Score": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Average sentiment polarity from news headlines (-1 bearish to +1 bullish)"
                    ),
                    "Strategy Cumulative Return": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Cumulative return from sentiment-based strategy"
                    ),
                    "Buy & Hold Cumulative Return": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Cumulative return from simple buy & hold"
                    )
                }
                st.dataframe(
                    df[['Date', 'Closing Price', 'Daily Sentiment Score', 'Strategy Cumulative Return', 'Buy & Hold Cumulative Return']].set_index('Date'),
                    column_config=column_config,
                    use_container_width=True
                )
                # Chart with renamed columns (ensure single-level)
                st.subheader("Cumulative Returns Chart")
                chart_df = df.set_index('Date')[['Strategy Cumulative Return', 'Buy & Hold Cumulative Return']]
                chart_df.columns = pd.Index(chart_df.columns)  # Force single-level
                st.line_chart(chart_df)
        except Exception as e:
            st.error(f"Error running backtest: {type(e).__name__}: {str(e)}. Check dates or network.")

# === PORTFOLIO P&L ===
st.markdown("### Portfolio Overview")
portfolio = {}
total_value = 0
for a_type, a_id in st.session_state.watchlist:
    try:
        if a_type == 'Stock':
            data = yf.Ticker(a_id).history(period="1d")
            if not data.empty:
                price = data['Close'].iloc[-1]
        else:  # Crypto
            price_data = cg.get_price(a_id, vs_currencies='usd')
            price = price_data[a_id]['usd']
        shares = st.number_input(f"Shares of {a_id} ({a_type})", min_value=0, value=st.session_state.get(f"shares_{a_type}_{a_id}", 10), key=f"input_{a_type}_{a_id}")
        st.session_state[f"shares_{a_type}_{a_id}"] = shares
        value = shares * price
        portfolio[(a_type, a_id)] = {"price": price, "shares": shares, "value": value}
        total_value += value
    except Exception as e:
        st.warning(f"Could not fetch price for {a_id} ({a_type}): {type(e).__name__}: {str(e)}. Skipping.")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total Value", f"${total_value:,.2f}")
with col_b:
    st.metric("Assets", len(portfolio))
with col_c:
    st.metric("Watchlist", len(st.session_state.watchlist))
st.markdown("### My Watchlist")
for (a_type, a_id), p in portfolio.items():
    st.write(f"• **{a_id} ({a_type})**: {p['shares']} × ${p['price']:.2f} = **${p['value']:.2f}**")

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
            .stButton > button:hover { background-color: #3d3d2d !important; }
            [data-testid="stFormSubmitButton"] > button { background-color: #2d2d2d !important; color: #ffffff !important; border: 1px solid #555 !important; }
            /* SELECTBOX — LIGHT BG + DARK TEXT */
            .stSelectbox > div > div { background-color: #f0f0f0 !important; color: #0e1117 !important; border: 1px solid #ddd !important; }
            .stSelectbox > div > div > div { background-color: #f0f0f0 !important; color: #0e1117 !important; }
            .stSelectbox [data-baseweb="select"] > div { background-color: #f0f0f0 !important; color: #0e1117 !important; }
            /* DROPDOWN MENU — DARK TEXT on LIGHT BG + HIGH CONTRAST */
            [data-baseweb="menu"] {
                background-color: #f0f0f0 !important;
                border: 1px solid #ddd !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            }
            [data-baseweb="menu"] div {
                color: #0e1117 !important;
                background-color: #f0f0f0 !important;
                font-weight: 600 !important;
                padding: 10px 16px !important;
            }
            [data-baseweb="menu"] div:hover {
                background-color: #e0e0e0 !important;
                color: #0e1117 !important;
            }
            [data-baseweb="menu"] div[data-selected="true"] {
                background-color: #d0d0d0 !important;
                color: #0e1117 !important;
            }
            .stSuccess { background-color: #1a4d1a !important; color: #ffffff !important; border: 1px solid #2a6d2a !important; }
            .stInfo { background-color: #0e3d6b !important; color: #ffffff !important; border: 1px solid #1e5d8b !important; }
            </style>
            """, unsafe_allow_html=True)

st.caption("Jack Evans | Moorpark College AS-T | Nov 2025")
