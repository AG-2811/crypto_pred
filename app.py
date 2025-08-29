import time
import math
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from binance.client import Client
import os
import pickle
import random

with open('crypto_pred.pkl','rb') as f:
     HAS_XGB = pickle.load(f)

api_key = "vBKombEymiA64O7LjsqEIACYJTsCrLojshIVjoMy9ibh63ePcTcaz6Ga5i229JzV"
api_secret = "r7LZZ37YlshUXdQoUkUoaMmsfIKAo1NT9esV7a2yWxcAZz5lp8it8txWvVVTBGdM"
client = Client(api_key, api_secret,testnet=True)

# Get 100 days of daily BTCUSDT data
candles = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, limit=103)
df = pd.DataFrame(candles, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["close"] = df["close"].astype(float)

# Train-test split index
split = int(len(df) * 0.8)

# Scale ONLY on training set to avoid leakage
scaler = MinMaxScaler()
scaler.fit(df[["close"]].iloc[:split])
df["close_scaled"] = scaler.transform(df[["close"]])

# Create lag features + technical features
df["lag1"] = df["close_scaled"].shift(1)
df["lag2"] = df["close_scaled"].shift(2)
df["lag3"] = df["close_scaled"].shift(3)
df["return"] = df["close"].pct_change()
df["rolling_mean_3"] = df["close_scaled"].rolling(3).mean()

# Drop NaN rows caused by shifting/rolling
df = df.dropna().reset_index(drop=True)
st.set_page_config(layout="wide")
st.title("📈 Bitcoin Price Prediction")
# st.markdown("---")
st.caption("⚠️ Richer Risk")
SYMBOL_DEFAULT = "BTCUSDT"
with st.sidebar:
    st.sidebar.title("📊 Crypto Dashboard")
    st.sidebar.image('https://i.makeagif.com/media/6-09-2020/3ccDb4.gif')
    option = st.sidebar.radio("Select view:", ["Live Price & Prediction", "About"])
    st.header("Settings for Prediction")
    symbol = st.text_input("Symbol", value=SYMBOL_DEFAULT, help="Use Binance symbols like BTCUSDT, ETHUSDT")
    interval = st.selectbox("Interval", ["1 week", "1 month", "3 months"], index=0)
    lookback_days = st.slider("Lookback (days)", 1, 100, 20)
    auto_refresh_sec = st.slider("Auto-refresh seconds (spot price)", 5, 120, 1)
    train_btn = st.button("Train / Refresh Model", type="primary")

if option == "Live Price & Prediction":   
    st_autorefresh_count = st.experimental_rerun if False else None  # placeholder to avoid lints
    st.markdown("---")
    st.image('https://cdnb.artstation.com/p/assets/images/images/009/358/213/original/tony-twaine-comp-2-2.gif?1518528958')
    col1, col2, col3 = st.columns([3,3,1])
    with col1:
        st.subheader("Real-Time Spot Price")
        try:
            from binance.client import Client
            import os
            
            # Create a client (you can create an API key in Binance account)
            api_key = "zZAJfb9fnVSD56Z6WWavnm1tcsYucAmcFYRk4LSX3Z0Cai2Wlqt31C9Kyv3JTG0y"
            api_secret = "Z86f6sjcJpUxCwgKijcjkL1Tm9uZXh8myubOER1eqFtdqVCdLZbxt1gIs0T1onKc"
            client = Client(api_key, api_secret,testnet=True)
            
            # Get latest BTC/USDT price
            price = client.get_symbol_ticker(symbol="BTCUSDT")
            price = float(price['price'])
            # print(f"BTC Price: {price['price']} USD")
            st.metric(label=f"{symbol} spot", value=f"{price:,.2f}")
            st.caption("Updates when you press 'R' to rerun, or on page interaction. Use the slider to control frequency and re-run.")
            st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        except Exception as e:
            st.error(f"Failed to fetch spot price: {e}")
    
    
    with col2:
        st.subheader("Historical Candles")
        if train_btn:
            st.cache_data.clear()
        st.spinner("Fetching historical data...")
            # hist = fetch_history(symbol, interval, lookback_days)
        if df is None or df.empty:
            st.warning("No data returned.")
        else:
            fig = go.Figure(data=[go.Candlestick(x=df.iloc[-lookback_days:]['timestamp'],
                                                 open=df['open'],
                                                 high=df['high'],
                                                 low=df['low'],
                                                 close=df['close'])])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Model Used")
        st.write("XGBoost" if HAS_XGB else "RandomForest")
        st.write(f"Interval: {interval}")
        st.write(f"Lookback: {lookback_days}d")
    
    def evaluate(final_value, y_pred):
        mae = mean_absolute_error(final_value, y_pred)
        rmse = math.sqrt(mean_squared_error(final_value, y_pred))
        return mae, rmse
    
    st.markdown("---")
    st.subheader("Train & Evaluate")
    col = ["lag1", "lag2", "lag3", "return", "rolling_mean_3"]
    
    features = df[["lag1", "lag2", "lag3", "return", "rolling_mean_3"]]
    y_pred_scaled = HAS_XGB.predict(features)
    
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    # st.write(f'Prediction:{y_pred}')
    mae = ((mean_absolute_error(df['close'], y_pred))/price)*100
    rmse = ((math.sqrt(mean_squared_error(df['close'], y_pred)))/price)*100
    met1, met2 = st.columns(2)
    met1.metric("MAE (return)", f"{mae:.6f}")
    met2.metric("RMSE (return)", f"{rmse:.6f}")
    fig2 = go.Figure()
    if interval == '1 week':
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(7), y=df['close'], mode="lines", name="Actual Close"))
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(7), y=y_pred, mode="lines", name="Predicted (walk-forward)"))
    elif interval == '1 month':
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(30), y=df['close'], mode="lines", name="Actual Close"))
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(30), y=y_pred, mode="lines", name="Predicted (walk-forward)"))
    elif interval == '3 months':
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(90), y=df['close'], mode="lines", name="Actual Close"))
        fig2.add_trace(go.Scatter(x=df['timestamp'].tail(90), y=y_pred, mode="lines", name="Predicted (walk-forward)"))
    else:
        st.warning("Not enough data after feature creation. Increase lookback.")
    fig2.update_layout(height=420, margin=dict(l=30, r=30, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    
    # met3, met4 = st.columns(2)
    # met3.metric("BTC Actual Price",f"${df['close'].tail(1):}")
    # met3.metric("BTC Predicted Price", f"${y_pred[-1]:,.2f}")
    
    st.markdown("### Forecast Next Steps")
    def future_pred():
        last_row = df.iloc[-1]
        
        # Features must match training features order
        X_next = np.array([[
            last_row["lag1"],
            last_row["lag2"],
            last_row["lag3"],
            last_row["return"],
            last_row["rolling_mean_3"]
        ]])
        
        # --- Step 2: predict (still in scaled form)
        y_next_scaled = HAS_XGB.predict(X_next)
        
        # --- Step 3: inverse transform back to real BTC price
        y_next = scaler.inverse_transform(y_next_scaled.reshape(-1, 1)).ravel()[0]
        return y_next
    
    #For Tommorow pred
    
    pred = future_pred()
    st.write(f"###### Predicted tommorow BTC price: :green[${pred:,.2f}🚀]")
if option == "About": 
    st.header("ℹ️ About This Dashboard")
    st.write("""
    This dashboard shows **real-time BTC/USDT price** using Binance API.  
    Features:  
    - Live price metric  
    - Real-time Price (updated every second)  
    - Sidebar navigation for easy access
    """)




