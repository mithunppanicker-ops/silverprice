import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Silver Price Predictor", page_icon="🪙", layout="wide")

@st.cache_data
def load_data():
    ticker = 'SI=F' # Silver Futures
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

st.title("🪙 Silver Price Predictor (5-Year Historical & 1-Year Forecast)")
st.markdown("This application extracts the last 5 years of silver prices and uses an XGBoost machine learning model to predict prices for the next year, optimizing for the lowest Root Mean Squared Error (RMSE).")

with st.spinner("Fetching 5 years of historical silver data from Yahoo Finance..."):
    df = load_data()

st.subheader("Historical Data (Last 5 Years)")
st.dataframe(df.tail())

# Preprocessing for Time Series as Regression problem (Predicting next day based on past N days)
df_model = df[['Date', 'Close']].copy()
# yfinance might return MultiIndex columns for 'Close' if multiple tickers are fetched, 
# although we just fetched one, let's ensure it's a 1D Series.
if isinstance(df_model.columns, pd.MultiIndex):
    df_model.columns = df_model.columns.droplevel() # Drop 'Ticker' level if it exists

df_model = df[['Date']].copy()
df_model['Close'] = df['Close']

df_model.set_index('Date', inplace=True)
# Fill missing business days
df_model = df_model.resample('D').ffill()
df_model.reset_index(inplace=True)

# Feature Engineering
df_model['day'] = df_model['Date'].dt.day
df_model['month'] = df_model['Date'].dt.month
df_model['year'] = df_model['Date'].dt.year
df_model['dayofweek'] = df_model['Date'].dt.dayofweek

# Lag features
for i in range(1, 8):
    df_model[f'lag_{i}'] = df_model['Close'].shift(i)
    
df_model.dropna(inplace=True)

# Prepare train/test
X = df_model.drop(['Date', 'Close'], axis=1)
y = df_model['Close']

# Train Test split (Sequential for time series)
split_idx = int(len(df_model) * 0.95) # 95% train, 5% test for RMSE calculation
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

with st.spinner("Training XGBoost prediction model minimized for RMSE..."):
    # Train Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

st.success(f"Model trained successfully! Holdout Test RMSE: **${rmse:.2f}**")

# Future Prediction (1 Year - 365 Days)
st.subheader("1-Year Price Forecast (Next 365 Days)")
future_dates = [df_model['Date'].iloc[-1] + timedelta(days=x) for x in range(1, 366)]
future_df = pd.DataFrame({'Date': future_dates})

# We need to iteratively predict since we rely on lag features
last_known_data = df_model.iloc[-1].copy()
predictions = []

current_features = {
    'day': future_dates[0].day,
    'month': future_dates[0].month,
    'year': future_dates[0].year,
    'dayofweek': future_dates[0].dayofweek,
}
for i in range(1, 8):
    current_features[f'lag_{i}'] = last_known_data[f'lag_{i-1}'] if i > 1 else last_known_data['Close']

with st.spinner("Generating 365-day forecast..."):
    for date in future_dates:
        # Predict current day
        # Ensure column order matches X_train precisely
        pred_features = pd.DataFrame([current_features], columns=X_train.columns).astype(float)
        pred_close = model.predict(pred_features)[0]
        predictions.append(pred_close)
        
        # Update features for next iteration
        next_date = date + timedelta(days=1)
        next_features = {
            'day': next_date.day,
            'month': next_date.month,
            'year': next_date.year,
            'dayofweek': next_date.dayofweek,
        }
        
        # Shift lags
        for i in range(7, 1, -1):
            next_features[f'lag_{i}'] = current_features[f'lag_{i-1}']
        next_features['lag_1'] = pred_close
        
        current_features = next_features

future_df['Predicted_Close'] = predictions

# Visualization
fig = go.Figure()

# Historical Trace
fig.add_trace(go.Scatter(
    x=df_model['Date'], 
    y=df_model['Close'],
    mode='lines',
    name='Historical Close Price',
    line=dict(color='blue')
))

# Future Trace
fig.add_trace(go.Scatter(
    x=future_df['Date'], 
    y=future_df['Predicted_Close'],
    mode='lines',
    name='Predicted Close Price',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title='Silver Price History & Forecast',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_white',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

st.info("💡 Note: Commodity prices are highly volatile and influenced by unpredictable macroeconomic factors. This model relies purely on historical price patterns (auto-regression).")
