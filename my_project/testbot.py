import ccxt
import pandas as pd
import time
import numpy as np

# Setup Binance API
api_key = 'nCGMS7LnW15hpORxSNnvS7Hqa7SDJ94wduZKuSjl8aLWQYO7JZ9MghtJ9QOB9qMi'
api_secret = 'WGjW8dSRjgzvT4215TfHJB6ZLLmZuINHfdTwq8TR95fnilgAHufcAqYVao4yMiwj'

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True  # To avoid hitting rate limits
})

# Define constants
symbol = 'XRP/USDT'
timeframe = '5m'
limit = 500  # Max candles per fetch (use higher limit for continuous data)

# Function to fetch the latest 5-minute candles
def fetch_candles():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Function to calculate the EMAs and spread
def calculate_indicators(df):
    # EMA
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # EMA spread
    df['ema_spread'] = df['ema_5'] - df['ema_20']

    # RSI (14-period by default)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volume Analysis (Simple Moving Average of volume)
    df['vol_sma'] = df['volume'].rolling(window=20).mean()

    return df


# Function to define buy/sell strategy based on EMA spread
def check_buy_sell(df):
    last_spread = df['ema_spread'].iloc[-1]
    prev_spread = df['ema_spread'].iloc[-2]
    rsi = df['rsi'].iloc[-1]
    last_price = df['close'].iloc[-1]
    volume = df['volume'].iloc[-1]
    vol_sma = df['vol_sma'].iloc[-1]
    # 1. **Buy Conditions:**
    # - EMA crossover: EMA 5 crosses above EMA 20
    # - Spread increasing (momentum confirmation)
    # - RSI is not overbought (below 70)
    # - Volume is higher than the 20-period average volume (confirm trend strength)
    if last_spread > 0 and last_spread > prev_spread and rsi < 70 and volume > vol_sma:
        return 'buy'

    # 2. **Sell Conditions:**
    # - EMA crossover: EMA 5 crosses below EMA 20
    # - Spread decreasing (momentum confirmation)
    # - RSI is not oversold (above 30)
    # - Volume is higher than the 20-period average volume (confirm trend strength)
    elif last_spread < 0 and last_spread < prev_spread and rsi > 30 and volume > vol_sma:
        return 'sell'

    # 3. **Hold Signal:**
    # If none of the conditions met, hold position
    return 'hold'

# Risk management: stop-loss and take-profit setup
def risk_management(last_price, action):
    stop_loss_pct = 0.2  # 0.2% stop-loss
    take_profit_pct = 0.3  # 0.3% take-profit

    stop_loss = last_price - (last_price * stop_loss_pct)
    take_profit = last_price + (last_price * take_profit_pct)

    if action == 'buy':
        return stop_loss, take_profit
    elif action == 'sell':
        return stop_loss, take_profit

# Function to place orders (simplified, modify for real trading)
def place_order(action, amount=0.001):  # Amount is in BTC for example
    if action == 'buy':
        print("Placing Buy Order")
        # Uncomment below to place a real order
        # binance.create_market_buy_order(symbol, amount)
    elif action == 'sell':
        print("Placing Sell Order")
        # Uncomment below to place a real order
        # binance.create_market_sell_order(symbol, amount)

# Main trading loop
def run_bot():
    while True:
        df = fetch_candles()
        df = calculate_indicators(df)

        action = check_buy_sell(df)
        print(f"Action: {action}")

        if action != 'hold':
            print(f"Action: {action}")

            last_price = df['close'].iloc[-1]
            stop_loss, take_profit = risk_management(last_price, action)

            print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

            place_order(action)

        # Sleep for 5 minutes before next iteration (or adjust as needed)
        time.sleep(300)

if __name__ == "__main__":
    run_bot()
