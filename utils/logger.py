import logging
import logging.handlers
import os
from typing import Optional
from config.settings import LoggingConfig

class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger(config: Optional[LoggingConfig] = None, name: Optional[str] = None) -> logging.Logger:
    """Setup logger with file and console handlers."""
    if config is None:
        config = LoggingConfig()
    
    logger = logging.getLogger(name or "trading_bot")
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_dir = os.path.dirname(config.file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        config.file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
