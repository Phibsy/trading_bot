from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class TradingSignal:
    """Trading signal data class."""
    
    def __init__(
        self,
        symbol: str,
        signal_type: Any,  # Using Any to avoid circular imports
        confidence: float,
        price: float,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.price = price
        self.reasoning = reasoning
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': str(self.signal_type) if hasattr(self.signal_type, 'value') else str(self.signal_type),
            'confidence': self.confidence,
            'price': self.price,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

class BaseStrategy(ABC):
    """Abstract base strategy class."""
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        self.enabled = True
        self.last_signals: Dict[str, TradingSignal] = {}
    
    @abstractmethod
    async def analyze(
        self, 
        market_data: Dict[str, pd.DataFrame],
        technical_analysis: Dict[str, Dict[str, Any]],
        ai_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[TradingSignal]:
        """
        Analyze market data and return trading signals.
        
        Args:
            market_data: Dictionary of symbol -> OHLCV DataFrame
            technical_analysis: Dictionary of symbol -> technical indicators
            ai_analysis: Optional AI analysis results
            
        Returns:
            List of trading signals
        """
        pass
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal."""
        if not signal.symbol or not signal.price or signal.price <= 0:
            return False
        
        if not 0 <= signal.confidence <= 100:
            return False
        
        # Import here to avoid circular imports
        try:
            from config.constants import SignalType
            if hasattr(signal.signal_type, 'value'):
                # It's an enum
                return signal.signal_type in SignalType
            else:
                # It's a string, check if it's valid
                return str(signal.signal_type) in ['BUY', 'SELL', 'HOLD']
        except ImportError:
            # Fallback validation
            return str(signal.signal_type) in ['BUY', 'SELL', 'HOLD']
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get strategy parameter."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set strategy parameter."""
        self.parameters[key] = value
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
