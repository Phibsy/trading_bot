import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional, Any
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from config.constants import TECHNICAL_PARAMS

class TechnicalAnalyzer:
    """Technical analysis indicator calculator."""
    
    def __init__(self):
        self.params = TECHNICAL_PARAMS
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            if len(data) < period:
                return pd.Series(index=data.index, dtype=float)
            return ta.momentum.RSIIndicator(
                close=data['close'], 
                window=period
            ).rsi()
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_bollinger_bands(
        self, 
        data: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            if len(data) < period:
                empty_series = pd.Series(index=data.index, dtype=float)
                return {
                    'upper': empty_series,
                    'middle': empty_series,
                    'lower': empty_series,
                    'width': empty_series,
                    'percent': empty_series
                }
                
            bb = ta.volatility.BollingerBands(
                close=data['close'],
                window=period,
                window_dev=std_dev
            )
            return {
                'upper': bb.bollinger_hband(),
                'middle': bb.bollinger_mavg(),
                'lower': bb.bollinger_lband(),
                'width': bb.bollinger_wband(),
                'percent': bb.bollinger_pband()
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return {
                'upper': empty_series,
                'middle': empty_series,
                'lower': empty_series,
                'width': empty_series,
                'percent': empty_series
            }
    
    def calculate_macd(
        self, 
        data: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        macd = ta.trend.MACD(
            close=data['close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        return {
            'volume_sma': ta.volume.VolumeSMAIndicator(
                close=data['close'],
                volume=data['volume'],
                window=self.params['VOLUME_SMA']['period']
            ).volume_sma(),
            'volume_ratio': data['volume'] / data['volume'].rolling(20).mean(),
            'obv': ta.volume.OnBalanceVolumeIndicator(
                close=data['close'],
                volume=data['volume']
            ).on_balance_volume()
        }
    
    @staticmethod
    def _calculate_support_resistance_numba(
        prices: np.ndarray, 
        window: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate support and resistance levels using Numba JIT if available."""
        if NUMBA_AVAILABLE:
            return TechnicalAnalyzer._calculate_support_resistance_jit(prices, window)
        else:
            return TechnicalAnalyzer._calculate_support_resistance_python(prices, window)
    
    @staticmethod
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda f: f
    def _calculate_support_resistance_jit(
        prices: np.ndarray, 
        window: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """JIT compiled version."""
        n = len(prices)
        support = np.zeros(n)
        resistance = np.zeros(n)
        
        for i in range(window, n):
            window_data = prices[i-window:i+1]
            support[i] = np.min(window_data)
            resistance[i] = np.max(window_data)
        
        return support, resistance
    
    @staticmethod
    def _calculate_support_resistance_python(
        prices: np.ndarray, 
        window: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Python fallback version."""
        n = len(prices)
        support = np.zeros(n)
        resistance = np.zeros(n)
        
        for i in range(window, n):
            window_data = prices[i-window:i+1]
            support[i] = np.min(window_data)
            resistance[i] = np.max(window_data)
        
        return support, resistance
    
    def calculate_support_resistance(
        self, 
        data: pd.DataFrame, 
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """Calculate support and resistance levels."""
        prices = data['close'].values
        support, resistance = self._calculate_support_resistance_numba(prices, window)
        
        return {
            'support': pd.Series(support, index=data.index),
            'resistance': pd.Series(resistance, index=data.index)
        }
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend strength indicators."""
        try:
            if len(data) < 14:  # Need minimum data for ADX
                return {
                    'adx': 0.0,
                    'slope': 0.0,
                    'roc': 0.0
                }
            
            close_prices = data['close']
            
            # ADX for trend strength
            try:
                adx = ta.trend.ADXIndicator(
                    high=data['high'],
                    low=data['low'],
                    close=data['close']
                ).adx().iloc[-1]
                adx = adx if not (np.isnan(adx) or np.isinf(adx)) else 0.0
            except Exception:
                adx = 0.0
            
            # Linear regression slope
            try:
                x = np.arange(len(close_prices))
                if len(x) > 1 and len(close_prices) > 1:
                    slope = np.polyfit(x, close_prices, 1)[0]
                    slope = slope if not (np.isnan(slope) or np.isinf(slope)) else 0.0
                else:
                    slope = 0.0
            except Exception:
                slope = 0.0
            
            # Rate of change
            try:
                roc = ta.momentum.ROCIndicator(close=close_prices).roc().iloc[-1]
                roc = roc if not (np.isnan(roc) or np.isinf(roc)) else 0.0
            except Exception:
                roc = 0.0
            
            return {
                'adx': float(adx),
                'slope': float(slope),
                'roc': float(roc)
            }
        
        except Exception as e:
            return {
                'adx': 0.0,
                'slope': 0.0,
                'roc': 0.0
            }
    
    def analyze_symbol(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Comprehensive technical analysis for a symbol."""
        if len(data) < 50:  # Need minimum data for analysis
            return {'error': 'Insufficient data for analysis'}
        
        try:
            # Calculate all indicators
            rsi = self.calculate_rsi(data)
            bb = self.calculate_bollinger_bands(data)
            macd = self.calculate_macd(data)
            volume = self.calculate_volume_indicators(data)
            sr = self.calculate_support_resistance(data)
            trend = self.calculate_trend_strength(data)
            
            # Get latest values
            latest_idx = data.index[-1]
            current_price = data['close'].iloc[-1]
            
            # RSI signals
            rsi_value = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
            rsi_signal = 'OVERSOLD' if rsi_value < 30 else 'OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL'
            
            # Bollinger Band signals
            bb_percent = bb['percent'].iloc[-1] if not np.isnan(bb['percent'].iloc[-1]) else 0.5
            bb_signal = 'OVERSOLD' if bb_percent < 0.1 else 'OVERBOUGHT' if bb_percent > 0.9 else 'NEUTRAL'
            
            # MACD signals
            macd_histogram = macd['histogram'].iloc[-1] if not np.isnan(macd['histogram'].iloc[-1]) else 0
            macd_signal = 'BULLISH' if macd_histogram > 0 else 'BEARISH'
            
            # Volume analysis
            volume_ratio = volume['volume_ratio'].iloc[-1] if not np.isnan(volume['volume_ratio'].iloc[-1]) else 1.0
            volume_signal = 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.5 else 'NORMAL'
            
            return {
                'symbol': symbol,
                'timestamp': latest_idx,
                'current_price': current_price,
                'indicators': {
                    'rsi': {
                        'value': rsi_value,
                        'signal': rsi_signal
                    },
                    'bollinger': {
                        'upper': bb['upper'].iloc[-1],
                        'lower': bb['lower'].iloc[-1],
                        'percent': bb_percent,
                        'signal': bb_signal
                    },
                    'macd': {
                        'macd': macd['macd'].iloc[-1],
                        'signal_line': macd['signal'].iloc[-1],
                        'histogram': macd_histogram,
                        'signal': macd_signal
                    },
                    'volume': {
                        'ratio': volume_ratio,
                        'signal': volume_signal
                    },
                    'support_resistance': {
                        'support': sr['support'].iloc[-1],
                        'resistance': sr['resistance'].iloc[-1]
                    },
                    'trend': trend
                }
            }
        
        except Exception as e:
            return {'error': f'Analysis error: {str(e)}'}
