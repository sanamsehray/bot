import pandas as pd
from binance.client import Client
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os
# from dotenv import load_dotenv

# Load API keys from environment variables (ensure .env file contains your keys)
# load_dotenv()
api_key = 'nCGMS7LnW15hpORxSNnvS7Hqa7SDJ94wduZKuSjl8aLWQYO7JZ9MghtJ9QOB9qMi.'
api_secret = 'WGjW8dSRjgzvT4215TfHJB6ZLLmZuINHfdTwq8TR95fnilgAHufcAqYVao4yMiwj'
client = Client(api_key, api_secret)

# Fetch historical data from Binance
symbol = 'XRPUSDT'
interval = Client.KLINE_INTERVAL_1HOUR  # 1 hour candles
limit = 1000  # Number of data points to fetch (can change as needed)

klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

# Convert data into a DataFrame
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'num_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])

# Convert timestamps to datetime
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# Use only relevant columns: Open Time, Close Price
df = df[['open_time', 'close']]

# Set the datetime as index
df.set_index('open_time', inplace=True)

# Convert close price to numeric
df['close'] = pd.to_numeric(df['close'])

# Display the first few rows
print(df.head())

# Prophet Forecasting
df_prophet = df[['close']].reset_index()
df_prophet.columns = ['ds', 'y']  # 'ds' is the date, 'y' is the target variable (price)

# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Make future predictions
future = prophet_model.make_future_dataframe(periods=24)  # Predict the next 24 hours (hourly frequency)
forecast = prophet_model.predict(future)

# Plot the forecast
prophet_model.plot(forecast)
plt.title('Prophet Forecast for XRP/USDT')

# Normalize data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])

# Prepare data for LSTM (create sequences)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Use the last 60 hours to predict the next hour
X, y = create_dataset(scaled_data, time_step)

# Reshape X to be 3D for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict using the trained model
predicted_price = lstm_model.predict(X_test)

# Inverse transform to get the actual values
predicted_price = scaler.inverse_transform(predicted_price)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Real Price')
plt.plot(df.index[-len(predicted_price):], predicted_price, color='red', label='Predicted Price')
plt.title('XRP/USDT Price Prediction (LSTM Model)')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the LSTM model
rmse = np.sqrt(mean_squared_error(y_test, predicted_price))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Get current price from Binance
ticker = client.get_symbol_ticker(symbol="XRPUSDT")
print(f"Current XRP/USDT price: {ticker['price']}")

# ... (Your previous code up until training the LSTM model)

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict using the trained model
predicted_price = lstm_model.predict(X_test)

# Inverse transform to get the actual values
predicted_price = scaler.inverse_transform(predicted_price)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Real Price')
plt.plot(df.index[-len(predicted_price):], predicted_price, color='red', label='Predicted Price')
plt.title('XRP/USDT Price Prediction (LSTM Model)')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the LSTM model
rmse = np.sqrt(mean_squared_error(y_test, predicted_price))
# print(f"Root Mean Squared Error (RMSE): {rmse}")

# Get current price from Binance
ticker = client.get_symbol_ticker
# ... (Your previous code up until training the LSTM model)

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict using the trained model
predicted_price = lstm_model.predict(X_test)

# Inverse transform to get the actual values
predicted_price = scaler.inverse_transform(predicted_price)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Real Price')
plt.plot(df.index[-len(predicted_price):], predicted_price, color='red', label='Predicted Price')
plt.title('XRP/USDT Price Prediction (LSTM Model)')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the LSTM model
rmse = np.sqrt(mean_squared_error(y_test, predicted_price))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Get current price from Binance
ticker = client.get_symbol_ticker(symbol="XRPUSDT")
print(f"Current XRP/USDT price: {ticker['price']}")

# Now let's predict the next hour's price and set a price target

# Get the last 60 hours of data to predict the next hour
last_data = scaled_data[-time_step:].reshape(1, time_step, 1)  # Use the last time_step (60 hours)

# Predict the next hour's price using the trained LSTM model
predicted_price_next_hour = lstm_model.predict(last_data)

# Inverse transform to get the actual price value
predicted_price_next_hour = scaler.inverse_transform(predicted_price_next_hour)

# Calculate a target price (for example, 2% increase based on the predicted price)
target_percentage = 0.02  # 2% increase
price_target = predicted_price_next_hour * (1 + target_percentage)

# Output the prediction and target price
print(f"Predicted price for the next hour: {predicted_price_next_hour[0][0]:.4f} USDT")
print(f"Price target (2% increase): {price_target[0][0]:.4f} USDT")

