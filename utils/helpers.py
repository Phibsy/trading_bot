import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from functools import wraps

def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async retry decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(delay * (backoff ** attempt))
            return None
        return wrapper
    return decorator

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division to avoid division by zero."""
    return numerator / denominator if denominator != 0 else default

def calculate_position_size(
    account_value: float,
    position_size_pct: float,
    current_price: float,
    stop_loss_pct: float
) -> int:
    """Calculate position size using Kelly Criterion principles."""
    if account_value <= 0 or current_price <= 0 or stop_loss_pct <= 0:
        return 0
        
    max_position_value = account_value * position_size_pct
    risk_per_share = current_price * stop_loss_pct
    
    if risk_per_share > 0:
        shares = int(max_position_value / current_price)
        max_risk_shares = int((account_value * 0.02) / risk_per_share)  # Max 2% account risk
        return min(shares, max_risk_shares)
    return 0

def normalize_timestamp(timestamp: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Normalize various timestamp formats to datetime."""
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp).to_pydatetime()
    elif isinstance(timestamp, pd.Timestamp):
        return timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime):
        return timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

def is_market_hours() -> bool:
    """Check if current time is during market hours including extended hours (4:00 AM - 8:00 PM ET)."""
    from datetime import datetime
    import pytz
    
    # Get current time in ET
    et_tz = pytz.timezone('US/Eastern')
    et_now = datetime.now(et_tz)
    
    # Check if weekday (Monday = 0, Sunday = 6)
    if et_now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Get hour and minute in ET
    current_hour = et_now.hour
    current_minute = et_now.minute
    current_time = current_hour + current_minute / 60.0
    
    # Extended hours: 4:00 AM - 8:00 PM ET
    market_open = 4.0   # 4:00 AM
    market_close = 20.0 # 8:00 PM
    
    # Debug logging
    print(f"DEBUG: ET time: {et_now.strftime('%H:%M')} ET, Hour: {current_hour}, In range: {market_open <= current_time < market_close}")
    
    return market_open <= current_time < market_close

def format_currency(amount: float) -> str:
    """Format currency with proper formatting."""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with proper formatting."""
    return f"{value:.2%}"

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"{self.name} took {duration:.4f} seconds")
