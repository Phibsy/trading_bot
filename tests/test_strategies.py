import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy, TradingSignal
from strategies.rsi_strategy import RSIStrategy
from strategies.ml_strategy import MLStrategy
from config.constants import SignalType

class MockStrategy(BaseStrategy):
    """Mock strategy for testing base functionality."""
    
    async def analyze(self, market_data, technical_analysis, ai_analysis=None):
        return [TradingSignal(
            symbol="TEST",
            signal_type=SignalType.BUY,
            confidence=80.0,
            price=100.0,
            reasoning="Mock signal"
        )]

class TestBaseStrategy:
    """Test base strategy functionality."""
    
    def test_strategy_creation(self):
        """Test strategy creation and initialization."""
        strategy = MockStrategy("Test Strategy", {"param1": "value1"})
        
        assert strategy.name == "Test Strategy"
        assert strategy.enabled == True
        assert strategy.get_parameter("param1") == "value1"
        assert strategy.get_parameter("nonexistent", "default") == "default"
    
    def test_strategy_enable_disable(self):
        """Test strategy enable/disable functionality."""
        strategy = MockStrategy("Test Strategy")
        
        assert strategy.enabled == True
        
        strategy.disable()
        assert strategy.enabled == False
        
        strategy.enable()
        assert strategy.enabled == True
    
    def test_parameter_management(self):
        """Test parameter get/set functionality."""
        strategy = MockStrategy("Test Strategy")
        
        strategy.set_parameter("new_param", 42)
        assert strategy.get_parameter("new_param") == 42
        
        # Test default value
        assert strategy.get_parameter("missing_param", "default") == "default"
    
    def test_signal_validation(self):
        """Test signal validation logic."""
        strategy = MockStrategy("Test Strategy")
        
        # Valid signal
        valid_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=75.0,
            price=150.0
        )
        assert strategy.validate_signal(valid_signal) == True
        
        # Invalid signals
        invalid_cases = [
            ("", SignalType.BUY, 75.0, 150.0),  # Empty symbol
            ("AAPL", SignalType.BUY, 75.0, 0.0),  # Zero price
            ("AAPL", SignalType.BUY, 75.0, -10.0),  # Negative price
            ("AAPL", SignalType.BUY, -5.0, 150.0),  # Negative confidence
            ("AAPL", SignalType.BUY, 150.0, 150.0),  # Confidence > 100
        ]
        
        for symbol, signal_type, confidence, price in invalid_cases:
            invalid_signal = TradingSignal(symbol, signal_type, confidence, price)
            assert strategy.validate_signal(invalid_signal) == False
    
    @pytest.mark.asyncio
    async def test_strategy_analysis(self):
        """Test strategy analysis method."""
        strategy = MockStrategy("Test Strategy")
        
        # Mock data
        market_data = {"TEST": pd.DataFrame()}
        technical_analysis = {"TEST": {}}
        
        signals = await strategy.analyze(market_data, technical_analysis)
        
        assert len(signals) == 1
        assert signals[0].symbol == "TEST"
        assert signals[0].signal_type == SignalType.BUY

class TestRSIStrategy:
    """Test RSI strategy implementation."""
    
    @pytest.fixture
    def rsi_strategy(self):
        """Create RSI strategy for testing."""
        return RSIStrategy({
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'min_confidence': 60,
            'volume_threshold': 1.2
        })
    
    @pytest.fixture
    def sample_technical_analysis(self):
        """Create sample technical analysis data."""
        return {
            'TQQQ': {
                'symbol': 'TQQQ',
                'current_price': 50.0,
                'indicators': {
                    'rsi': {'value': 25.0, 'signal': 'OVERSOLD'},
                    'volume': {'ratio': 1.5, 'signal': 'HIGH'},
                    'bollinger': {'percent': 0.1},
                    'macd': {'histogram': 0.1},
                    'trend': {'adx': 25}
                }
            },
            'SQQQ': {
                'symbol': 'SQQQ',
                'current_price': 30.0,
                'indicators': {
                    'rsi': {'value': 75.0, 'signal': 'OVERBOUGHT'},
                    'volume': {'ratio': 1.3, 'signal': 'HIGH'},
                    'bollinger': {'percent': 0.9},
                    'macd': {'histogram': -0.1},
                    'trend': {'adx': 20}
                }
            }
        }
    
    def test_rsi_strategy_initialization(self, rsi_strategy):
        """Test RSI strategy initialization."""
        assert rsi_strategy.name == "RSI Strategy"
        assert rsi_strategy.get_parameter('oversold_threshold') == 30
        assert rsi_strategy.get_parameter('overbought_threshold') == 70
        assert rsi_strategy.enabled == True
    
    @pytest.mark.asyncio
    async def test_rsi_oversold_signal(self, rsi_strategy, sample_technical_analysis):
        """Test RSI oversold signal generation."""
        # Test with oversold RSI
        signals = await rsi_strategy.analyze(
            market_data={},
            technical_analysis={'TQQQ': sample_technical_analysis['TQQQ']},
            ai_analysis=None
        )
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == 'TQQQ'
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence >= 60  # Should meet minimum confidence
        assert 'RSI oversold' in signal.reasoning
    
    @pytest.mark.asyncio
    async def test_rsi_overbought_signal(self, rsi_strategy, sample_technical_analysis):
        """Test RSI overbought signal generation."""
        # Test with overbought RSI
        signals = await rsi_strategy.analyze(
            market_data={},
            technical_analysis={'SQQQ': sample_technical_analysis['SQQQ']},
            ai_analysis=None
        )
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == 'SQQQ'
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence >= 60
        assert 'RSI overbought' in signal.reasoning
    
    @pytest.mark.asyncio
    async def test_rsi_with_ai_confirmation(self, rsi_strategy, sample_technical_analysis):
        """Test RSI strategy with AI confirmation."""
        ai_analysis = {
            'TQQQ': {
                'signal': 'BUY',
                'confidence': 80,
                'reasoning': 'AI confirms buy signal'
            }
        }
        
        signals = await rsi_strategy.analyze(
            market_data={},
            technical_analysis={'TQQQ': sample_technical_analysis['TQQQ']},
            ai_analysis=ai_analysis
        )
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.confidence > 70  # Should be boosted by AI confirmation
        assert 'AI confirmation' in signal.reasoning
    
    @pytest.mark.asyncio
    async def test_rsi_with_ai_disagreement(self, rsi_strategy, sample_technical_analysis):
        """Test RSI strategy with AI disagreement."""
        ai_analysis = {
            'TQQQ': {
                'signal': 'SELL',  # AI disagrees with RSI oversold signal
                'confidence': 70,
                'reasoning': 'AI suggests sell'
            }
        }
        
        signals = await rsi_strategy.analyze(
            market_data={},
            technical_analysis={'TQQQ': sample_technical_analysis['TQQQ']},
            ai_analysis=ai_analysis
        )
        
        # Signal might be filtered out due to disagreement or have reduced confidence
        if signals:
            signal = signals[0]
            assert 'AI disagreement' in signal.reasoning
    
    @pytest.mark.asyncio
    async def test_rsi_insufficient_confidence(self, rsi_strategy):
        """Test RSI strategy with insufficient confidence."""
        # Create data that would generate low confidence
        weak_technical_analysis = {
            'TEST': {
                'symbol': 'TEST',
                'current_price': 50.0,
                'indicators': {
                    'rsi': {'value': 32.0, 'signal': 'NEUTRAL'},  # Borderline RSI
                    'volume': {'ratio': 0.8, 'signal': 'LOW'},   # Low volume
                    'bollinger': {'percent': 0.3},
                    'macd': {'histogram': 0.01},
                    'trend': {'adx': 15}
                }
            }
        }
        
        signals = await rsi_strategy.analyze(
            market_data={},
            technical_analysis=weak_technical_analysis,
            ai_analysis=None
        )
        
        # Should generate no signals due to insufficient confidence
        assert len(signals) == 0

class TestMLStrategy:
    """Test ML strategy implementation."""
    
    @pytest.fixture
    def ml_strategy(self):
        """Create ML strategy for testing."""
        return MLStrategy({
            'min_confidence': 70,
            'ai_weight': 0.4,
            'feature_weights': {
                'rsi': 0.25,
                'macd': 0.20,
                'bollinger': 0.20,
                'volume': 0.15,
                'trend': 0.20
            }
        })
    
    @pytest.fixture
    def sample_data_for_ml(self):
        """Create comprehensive sample data for ML strategy."""
        technical_analysis = {
            'AAPL': {
                'symbol': 'AAPL',
                'current_price': 175.0,
                'indicators': {
                    'rsi': {'value': 35.0, 'signal': 'OVERSOLD'},
                    'bollinger': {'percent': 0.2, 'signal': 'OVERSOLD'},
                    'macd': {'histogram': 0.5, 'signal': 'BULLISH'},
                    'volume': {'ratio': 1.8, 'signal': 'HIGH'},
                    'trend': {'adx': 35, 'slope': 0.2, 'roc': 0.03}
                }
            }
        }
        
        ai_analysis = {
            'AAPL': {
                'signal': 'BUY',
                'confidence': 85,
                'reasoning': 'Strong bullish indicators across multiple factors'
            }
        }
        
        return technical_analysis, ai_analysis
    
    def test_ml_strategy_initialization(self, ml_strategy):
        """Test ML strategy initialization."""
        assert ml_strategy.name == "ML Strategy"
        assert ml_strategy.get_parameter('ai_weight') == 0.4
        assert ml_strategy.get_parameter('min_confidence') == 70
    
    @pytest.mark.asyncio
    async def test_ml_strategy_requires_ai(self, ml_strategy):
        """Test that ML strategy requires AI analysis."""
        technical_analysis = {
            'TEST': {
                'symbol': 'TEST',
                'current_price': 100.0,
                'indicators': {}
            }
        }
        
        # Without AI analysis, should return empty signals
        signals = await ml_strategy.analyze(
            market_data={},
            technical_analysis=technical_analysis,
            ai_analysis=None
        )
        
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_ml_strategy_signal_generation(self, ml_strategy, sample_data_for_ml):
        """Test ML strategy signal generation."""
        technical_analysis, ai_analysis = sample_data_for_ml
        
        signals = await ml_strategy.analyze(
            market_data={},
            technical_analysis=technical_analysis,
            ai_analysis=ai_analysis
        )
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == 'AAPL'
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence >= 70
        assert 'ML composite' in signal.reasoning
    
    def test_technical_score_calculation(self, ml_strategy, sample_data_for_ml):
        """Test technical score calculation."""
        technical_analysis, _ = sample_data_for_ml
        feature_weights = ml_strategy.get_parameter('feature_weights')
        
        score = ml_strategy._calculate_technical_score(
            technical_analysis['AAPL'], 
            feature_weights
        )
        
        # Should be positive due to oversold conditions and bullish indicators
        assert -1.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_trend_confirmation(self, ml_strategy, sample_data_for_ml):
        """Test trend confirmation logic."""
        technical_analysis, _ = sample_data_for_ml
        
        # Test with bullish trend (should confirm BUY signal)
        bullish_confirmed = ml_strategy._confirm_trend(
            technical_analysis['AAPL'], 
            SignalType.BUY
        )
        assert bullish_confirmed == True
        
        # Test with bearish trend (should not confirm BUY signal)
        bearish_analysis = technical_analysis['AAPL'].copy()
        bearish_analysis['indicators']['trend']['slope'] = -0.2
        
        bearish_confirmed = ml_strategy._confirm_trend(
            bearish_analysis, 
            SignalType.BUY
        )
        assert bearish_confirmed == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
