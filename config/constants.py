from enum import Enum
from typing import List

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

# Supported trading symbols
SUPPORTED_SYMBOLS: List[str] = [
    "TQQQ", "SQQQ", "SPXL", "SPXS", "QQQ", "SPY",
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"
]

# Technical indicator parameters
TECHNICAL_PARAMS = {
    "RSI": {"period": 14, "overbought": 70, "oversold": 30},
    "BOLLINGER": {"period": 20, "std_dev": 2},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "VOLUME_SMA": {"period": 20}
}
