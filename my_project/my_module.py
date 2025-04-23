print('helloWorld')
from binance.client import Client
import pandas as pd

# Optional: Replace with your API keys if using private data
api_key = 'nCGMS7LnW15hpORxSNnvS7Hqa7SDJ94wduZKuSjl8aLWQYO7JZ9MghtJ9QOB9qMi.'
api_secret = 'WGjW8dSRjgzvT4215TfHJB6ZLLmZuINHfdTwq8TR95fnilgAHufcAqYVao4yMiwj'
client = Client(api_key, api_secret)

# Get historical klines (candlestick) data for BTC/USDT
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR  # 1h candles
limit = 100  # number of candles

klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'num_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])

# Convert timestamps
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

print(df[['open_time', 'open', 'high', 'low', 'close', 'volume']].head())
from binance.client import Client

# Initialize client (no API key needed for public price data)
client = Client()

# Get current price of BTC/USDT
ticker = client.get_symbol_ticker(symbol="XRPUSDT")
print(f"Current BTC/USDT price: {ticker['price']}")
