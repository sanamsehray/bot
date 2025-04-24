from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
import os

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Binance client setup
client = Client()

@app.get("/candles/{symbol}/{interval}")
def get_candlestick_data(symbol: str, interval: str):
    try:
        candles = client.get_klines(symbol=symbol.upper(), interval=interval, limit=100)
        formatted = [
            {
                "time": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
            } for candle in candles
        ]
        return formatted
    except Exception as e:
        return {"error": str(e)}
