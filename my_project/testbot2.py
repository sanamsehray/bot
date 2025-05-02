import pandas as pd
import numpy as np
import ccxt
import pandas as pd
import time
import numpy as np
from binance.client import Client

# # Setup Binance API
api_key = 'nCGMS7LnW15hpORxSNnvS7Hqa7SDJ94wduZKuSjl8aLWQYO7JZ9MghtJ9QOB9qMi'
api_secret = 'WGjW8dSRjgzvT4215TfHJB6ZLLmZuINHfdTwq8TR95fnilgAHufcAqYVao4yMiwj'


binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True  # To avoid hitting rate limits
})

api_keytest = '9ccda910a3bcb7911eb25c62cbc87484424d01ec9de05bf0086f6317556957d1'
api_secrettest = '270ee1182cdb3190ff1001aeb740075c525da11f664f0deebe957d1b9cae15d0'

# client =  ccxt.binance({
#     'apiKey': api_keytest,
#     'secret': api_secrettest,
#     'enableRateLimit': True,
#     'options': {
#         'defaultType': 'future'
#     }
# })

# client.set_sandbox_mode(True)  

client = Client(api_keytest, api_secrettest)

client.FUTURES_URL = "https://testnet.binancefuture.com"  # Use the testnet endpoint for paper trading


# Define constants
symbol = 'XRP/USDT'
timeframe = '5m'
limit = 500  # Max candles per fetch (use higher limit for continuous data)

def fetch_candles():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_indicators(df):
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_spread'] = df['ema_5'] - df['ema_20']

    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['hl'] = df['high'] - df['low']
    df['hc'] = np.abs(df['high'] - df['close'].shift())
    df['lc'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    df['vol_sma'] = df['volume'].rolling(window=20).mean()

    return df

def generate_trade_signal(df, atr_threshold=20):
    df = calculate_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_prices = df['close'].tail(3)
    price_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100

    if (
        last['ema_5'] > last['ema_20'] and
        last['ema_spread'] > prev['ema_spread'] and
        40 < last['rsi'] < 70 and
        last['macd_hist'] > prev['macd_hist'] and
        last['volume'] > last['vol_sma'] and
        last['atr'] > atr_threshold and
        price_change_pct >= 0.5
    ):
        return 'buy'

    elif (
        last['ema_5'] < last['ema_20'] and
        last['ema_spread'] < prev['ema_spread'] and
        30 < last['rsi'] < 60 and
        last['macd_hist'] < prev['macd_hist'] and
        last['volume'] > last['vol_sma'] and
        last['atr'] > atr_threshold and
        price_change_pct <= -0.5
    ):
        return 'sell'

    return 'hold'

def place_order(action, amount=0.001):  # Amount is in BTC for example
    if action == 'buy':
        print("Placing Buy Order")
        # Uncomment below to place a real order
        # binance.create_market_buy_order(symbol, amount)
    elif action == 'sell':
        print("Placing Sell Order")
        # Uncomment below to place a real order
        # binance.create_market_sell_order(symbol, amount)

def place_test_limit_order(symbol, leverage, usdt_amount, desired_price):
    """
    Places a limit order on Binance Futures Testnet with specified leverage and order size in USD.
    Automatically sets stop-loss at -10% and take-profit at +32%.
    """
    # Set leverage
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    
    # Calculate quantity based on the leverage and USD amount
    notional_value = leverage * usdt_amount
    quantity = round(notional_value / desired_price, 1)  # Quantity based on price
    
    print(f"Placing Limit BUY for {quantity} {symbol} at price {desired_price}")

    # Place the limit order (entry)
    entry_order = client.futures_create_order(
        symbol=symbol,
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_GTC,
        quantity=quantity,
        price=str(desired_price)
    )

    # Calculate stop-loss and take-profit prices
    stop_loss_price = round(desired_price * 0.90, 2)  # 10% below entry
    take_profit_price = round(desired_price * 1.32, 2)  # 32% above entry

    # Place Stop-Loss (STOP_MARKET)
    stop_loss_order = client.futures_create_order(
        symbol=symbol,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_STOP_MARKET,
        stopPrice=str(stop_loss_price),
        closePosition=True,
        timeInForce=Client.TIME_IN_FORCE_GTC
    )

    # Place Take-Profit (TAKE_PROFIT_MARKET)
    take_profit_order = client.futures_create_order(
        symbol=symbol,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_TAKE_PROFIT_MARKET,
        stopPrice=str(take_profit_price),
        closePosition=True,
        timeInForce=Client.TIME_IN_FORCE_GTC
    )

    print("Entry Order, Stop-Loss, and Take-Profit Orders placed.")
    return entry_order, stop_loss_order, take_profit_order
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
    

# Main trading loop
def run_bot():
    while True:
        df = fetch_candles()
        df = calculate_indicators(df)

        action = generate_trade_signal(df)
        print(f"Action: {action}")

        if action == 'hold':
            print(f"Action: {action}")

            last_price = df['close'].iloc[-1]
            # stop_loss, take_profit = risk_management(last_price, action)

            # print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

            place_test_limit_order(symbol=symbol, leverage=32, usdt_amount=10.0, desired_price=last_price)

        # Sleep for 5 minutes before next iteration (or adjust as needed)
        time.sleep(200)

if __name__ == "__main__":
    run_bot()
