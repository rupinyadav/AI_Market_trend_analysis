import yfinance as yf
import streamlit as st
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Reliance Market Trend Analysis",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📈 AI-based Market Trend Analysis: Reliance Industries")
st.write(
    "This dashboard presents historical trends, volatility patterns, "
    "and ARIMA-based forecasts for Reliance stock."
)

# -----------------------------
# LOAD DATA
# -----------------------------
#@st.cache_data(ttl=86400)  # cache for 1 day
def load_latest_data():
    df = yf.download("RELIANCE.NS", start="2018-01-01", progress=False)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df

df = load_latest_data()

# Moving averages
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

# Returns & volatility
df["Daily_Return"] = df["Close"].pct_change()
df["Volatility"] = df["Daily_Return"].rolling(20).std()

# Trend label
df["Trend"] = "Downtrend"
df.loc[df["MA20"] > df["MA50"], "Trend"] = "Uptrend"

# -----------------------------
# KEY METRICS
# -----------------------------
st.subheader("📊 Key Market Metrics")

latest_price = float(df["Close"].iloc[-1])
latest_trend = df["Trend"].iloc[-1]
latest_vol = float(df["Volatility"].iloc[-1])

col1, col2, col3 = st.columns(3)
col1.metric("Latest Close Price", f"{latest_price:.2f}")
col2.metric("Current Trend", latest_trend)
col3.metric("Recent Volatility", f"{latest_vol:.4f}")

# -----------------------------
# PRICE + MOVING AVERAGES
# -----------------------------
st.subheader("📉 Price Trend with Moving Averages")
st.image("data/eda_price_ma.png", use_column_width=True)

# -----------------------------
# VOLATILITY
# -----------------------------
st.subheader("📊 Volatility Over Time")
st.image("data/eda_volatility.png", use_column_width=True)

# -----------------------------
# DAILY RETURNS
# -----------------------------
st.subheader("📈 Daily Returns")
st.image("data/eda_daily_returns.png", use_column_width=True)

# -----------------------------
# OUTLIERS
# -----------------------------
st.subheader("🚨 Outlier Identification")
st.image("data/eda_outliers.png", use_column_width=True)

# -----------------------------
# FORECAST
# -----------------------------
st.subheader("🔮 ARIMA Price Forecast (Next 30 Trading Days)")
st.image("data/arima_forecast.png", use_column_width=True)

st.caption(
    "Forecast generated using an ARIMA time-series model trained on "
    "historical closing prices."
)
