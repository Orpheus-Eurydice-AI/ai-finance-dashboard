import streamlit as st  # For building the web app interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting charts
from textblob import TextBlob  # For sentiment analysis
from sklearn.linear_model import LinearRegression  # For simple forecasting
from datetime import datetime, timedelta, date  # For date handling
from polygon import RESTClient  # For fetching financial data via Polygon API
import streamlit.components.v1 as components  # For custom HTML components
import time  # For retry delays

st.set_page_config(page_title="Jack Evans AI Finance", layout="centered")
st.title("AI Finance Dashboard – Jack Evans")
st.markdown("*Real-time stock analysis + 7-day AI forecast + News Sentiment + Watchlist*")

# Cached API fetch functions to avoid rate limits
@st.cache_data(ttl=300)  # Cache for 5 min
def cached_get_aggs(client, ticker, multiplier, timespan, from_date, to_date):
    return client.get_aggs(ticker, multiplier, timespan, from_date, to_date)

@st.cache_data(ttl=300)  # Cache for 5 min
def cached_list_ticker_news(client, ticker, **kwargs):
    return list(client.list_ticker_news(ticker, **kwargs))

# Function with retry for 429 errors
def api_call_with_retry(func, *args, max_retries=3, delay=15, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if '429' in str(e):
                st.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Max retries exceeded due to rate limits.")

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
    with st.spinner("Fetching data..."):
        try:
            client = RESTClient(api_key=st.secrets["POLYGON_API_KEY"])
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Approx 3 months
            aggs = api_call_with_retry(cached_get_aggs, client, ticker, 1, "day", start_date.date().isoformat(), end_date.date().isoformat())
            if not aggs or len(aggs) < 7:
                st.error("Invalid ticker or insufficient data. Try **NVDA**, **AAPL**, or **TSLA**.")
            else:
                data = pd.DataFrame(aggs)
                data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date
                data.set_index('date', inplace=True)
                data = data[['close']]
                prices = data['close'].values
                days = np.arange(len(prices)).reshape(-1, 1)
                model = LinearRegression()
                model.fit(days, prices)
                future = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
                pred = model.predict(future)
                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, prices, label="Actual Price", color='blue')
                ax.plot(future_dates, pred, label="AI 7-Day Forecast", color='red', linestyle='--')
                ax.set_title(f"{ticker} Stock Price + AI Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                current = prices[-1]
                forecast = pred[-1]
                change = forecast - current
                pct = (change / current) * 100 if current != 0 else 0
                volatility = np.std(prices[-30:]) / np.mean(prices[-30:]) * 100 if len(prices) > 30 else 0
                st.markdown("### Forecast Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current:.2f}")
                with col2:
                    st.metric("7-Day Forecast", f"${forecast:.2f}", f"{pct:+.1f}%")
                with col3:
                    st.metric("30-Day Volatility", f"{volatility:.1f}%")
                # News Sentiment (Real via Polygon)
                news_list = api_call_with_retry(cached_list_ticker_news, client, ticker=ticker, limit=5)
                headlines = [n.title for n in news_list]
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
                if pct > 5:
                    st.success("STRONG BUY SIGNAL")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}. Check your API key or network.")

# === BACKTESTING ===
st.header("Backtesting: Sentiment-Based Strategy")
start_date = st.date_input("Start Date", value=date(2024, 1, 1))
end_date = st.date_input("End Date", value=date(2024, 12, 31))
if st.button("Run Backtest"):
    try:
        client = RESTClient(api_key=st.secrets["POLYGON_API_KEY"])
        # Fetch historical daily price data
        aggs = api_call_with_retry(cached_get_aggs, client, ticker, 1, "day", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.set_index('date', inplace=True)
        df = df[['close']]
        # Fetch historical news
        news_list = api_call_with_retry(cached_list_ticker_news, client, ticker=ticker, published_utc_gte=start_date.strftime("%Y-%m-%d"), published_utc_lte=end_date.strftime("%Y-%m-%d"), limit=1000, order="asc")
        # Compute daily average sentiment from news titles
        daily_sent = {}
        for n in news_list:
            pub_date = datetime.fromisoformat(n.published_utc.rstrip('Z')).date()
            title = n.title
            polarity = TextBlob(title).sentiment.polarity
            if pub_date in daily_sent:
                daily_sent[pub_date].append(polarity)
            else:
                daily_sent[pub_date] = [polarity]
        # Average sentiments per day
        for d in list(daily_sent.keys()):
            daily_sent[d] = sum(daily_sent[d]) / len(daily_sent[d])
        # Add sentiment to price DataFrame
        df['sentiment'] = pd.Series(daily_sent)
        df['sentiment'] = df['sentiment'].fillna(0)  # Neutral if no news
        # Compute daily returns
        df['ret'] = df['close'].pct_change()
        # Strategy returns: Multiply daily return by 1 if prev day's sentiment > 0, else 0
        df['strategy_ret'] = df['ret'] * (df['sentiment'].shift(1) > 0)
        # Cumulative returns
        df['strategy_cum'] = (1 + df['strategy_ret']).cumprod().fillna(1)
        df['buy_hold_cum'] = (1 + df['ret']).cumprod().fillna(1)
        # Display results
        st.subheader("Backtest Summary Table")
        st.dataframe(df[['close', 'sentiment', 'strategy_cum', 'buy_hold_cum']])
        st.subheader("Cumulative Returns Chart")
        st.line_chart(df[['strategy_cum', 'buy_hold_cum']])
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}. Check your API key, dates, or network.")

# === PORTFOLIO P&L ===
st.markdown("### Portfolio Overview")
portfolio = {}
total_value = 0
client = RESTClient(api_key=st.secrets["POLYGON_API_KEY"])  # Initialize client here for portfolio
for t in st.session_state.watchlist:
    try:
        # Fetch last day's close price using Polygon with retry
        today = datetime.now().date().isoformat()
        aggs = api_call_with_retry(cached_get_aggs, client, t, 1, "day", today, today)
        if aggs:
            price = aggs[0].close
            shares = st.number_input(f"Shares of {t}", min_value=0, value=st.session_state.get(f"shares_{t}", 10), key=f"input_{t}")
            st.session_state[f"shares_{t}"] = shares
            value = shares * price
            portfolio[t] = {"price": price, "shares": shares, "value": value}
            total_value += value
    except Exception as e:
        st.warning(f"Could not fetch price for {t}: {str(e)}. Skipping.")
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
