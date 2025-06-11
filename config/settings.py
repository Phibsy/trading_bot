import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    data_url: str = field(default_factory=lambda: os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"))

@dataclass
class GroqConfig:
    """Groq AI configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "mixtral-8x7b-32768"
    max_tokens: int = 1000
    temperature: float = 0.1

@dataclass
class TradingConfig:
    """Trading configuration."""
    symbols: List[str] = field(default_factory=lambda: ["TQQQ", "SQQQ", "SPXL", "SPXS"])
    max_positions: int = field(default_factory=lambda: int(os.getenv("MAX_POSITIONS", "3")))
    position_size: float = field(default_factory=lambda: float(os.getenv("POSITION_SIZE", "0.1")))
    stop_loss: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS", "0.02")))
    take_profit: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT", "0.03")))
    daily_loss_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT", "0.05")))
    min_confidence: float = 75.0
    correlation_threshold: float = 0.8

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///trading_bot.db"))

@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    enabled: bool = field(default_factory=lambda: os.getenv("REDIS_ENABLED", "false").lower() == "true")

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/trading_bot.log"))
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class BotConfig:
    """Main bot configuration."""
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
