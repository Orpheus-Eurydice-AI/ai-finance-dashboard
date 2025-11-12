AI Finance Dashboard 📈
An interactive web app for real-time stock analysis, AI-driven forecasting, and portfolio management. Built from scratch in November 2025 as my first AI project — from zero coding experience to a deployed app with ML predictions in 2 weeks.
What It Does

Pulls real-time stock data from Yahoo Finance (e.g., NVDA, AAPL)
Displays historical charts (line, candlestick) with technical indicators (SMA, RSI)
Uses AI models (LSTM via Keras) for 7-day price forecasts and news sentiment analysis
Manages portfolios: Add stocks, track value/returns, assess risk (VaR)
Interactive: Enter tickers → instant visuals with Plotly + export reports (PDF/CSV)

Demo
Live Streamlit Version
Tech Stack

Python 3 + yfinance (data) + Plotly (charts) + scikit-learn/Keras (AI)
Streamlit (web app) + GitHub/Streamlit Cloud (deployment)

Setup (Run Locally)

Clone: git clone https://github.com/[your-username]/ai-finance-dashboard.git
Install: pip install streamlit yfinance plotly scikit-learn tensorflow
Run: streamlit run app.py

Roadmap

Week 3: Add crypto tracking (e.g., BTC via CoinGecko)
Week 4: Integrate advanced NLP for sentiment + user auth
Contribute? Open a PR!

Built by Jack Evans — Aspiring AI Entrepreneur | Moorpark College AS-T Business Admin
