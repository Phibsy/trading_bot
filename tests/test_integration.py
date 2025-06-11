import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BotConfig, TradingConfig, AlpacaConfig, GroqConfig
from config.constants import SignalType
from strategies.base_strategy import TradingSignal
from strategies.rsi_strategy import RSIStrategy
from analysis.technical import TechnicalAnalyzer
from core.risk_manager import RiskManager
from core.portfolio import PortfolioManager

class TestIntegration:
    """Integration tests for the trading bot."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return BotConfig(
            alpaca=AlpacaConfig(
                api_key="test_key",
                secret_key="test_secret",
                base_url="https://paper-api.alpaca.markets"
            ),
            groq=GroqConfig(
                api_key="test_groq_key"
            ),
            trading=TradingConfig(
                symbols=["TQQQ", "SQQQ"],
                max_positions=2,
                position_size=0.1,
                stop_loss=0.02,
                take_profit=0.03,
                min_confidence=60.0
            )
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for testing."""
        # Generate 100 bars of realistic price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        base_price = 50.0
        price_data = []
        
        for i in range(100):
            if i == 0:
                price = base_price
            else:
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)
                price = price_data[-1] * (1 + change)
            price_data.append(price)
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, price_data)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, close, high),
                'low': min(open_price, close, low),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return {
            'TQQQ': df.copy(),
            'SQQQ': df.copy() * 0.8  # Different price level
        }
    
    def test_technical_analyzer_integration(self, sample_market_data):
        """Test full technical analysis integration."""
        analyzer = TechnicalAnalyzer()
        
        for symbol, data in sample_market_data.items():
            analysis = analyzer.analyze_symbol(data, symbol)
            
            # Verify structure
            assert 'symbol' in analysis
            assert 'current_price' in analysis
            assert 'indicators' in analysis
            
            # Verify all indicators are present
            indicators = analysis['indicators']
            required_indicators = ['rsi', 'bollinger', 'macd', 'volume', 'trend']
            for indicator in required_indicators:
                assert indicator in indicators
            
            # Verify indicator values are reasonable
            rsi_value = indicators['rsi']['value']
            assert 0 <= rsi_value <= 100
            
            bollinger_percent = indicators['bollinger']['percent']
            assert 0 <= bollinger_percent <= 1
    
    @pytest.mark.asyncio
    async def test_strategy_integration(self, sample_market_data):
        """Test strategy integration with market data."""
        # Create technical analysis
        analyzer = TechnicalAnalyzer()
        technical_analysis = {}
        
        for symbol, data in sample_market_data.items():
            analysis = analyzer.analyze_symbol(data, symbol)
            technical_analysis[symbol] = analysis
        
        # Test RSI strategy
        rsi_strategy = RSIStrategy({
            'oversold_threshold': 40,  # More lenient for testing
            'overbought_threshold': 60,
            'min_confidence': 50
        })
        
        signals = await rsi_strategy.analyze(
            market_data=sample_market_data,
            technical_analysis=technical_analysis,
            ai_analysis=None
        )
        
        # Verify signals structure
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol in ['TQQQ', 'SQQQ']
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 100
            assert signal.price > 0
    
    @pytest.mark.asyncio
    async def test_risk_manager_integration(self, mock_config):
        """Test risk manager integration."""
        risk_manager = RiskManager(mock_config.trading)
        
        # Create test signal
        signal = TradingSignal(
            symbol="TQQQ",
            signal_type=SignalType.BUY,
            confidence=80.0,
            price=50.0
        )
        
        # Mock position and account data
        current_positions = []
        account_info = {
            'equity': 100000.0,
            'buying_power': 80000.0,
            'cash': 20000.0,
            'portfolio_value': 100000.0,
            'day_trade_count': 0,
            'pattern_day_trader': False
        }
        
        # Test risk evaluation
        approved, reason, metrics = await risk_manager.evaluate_signal(
            signal, current_positions, account_info
        )
        
        # Should be approved for valid signal
        assert isinstance(approved, bool)
        assert isinstance(reason, str)
        assert isinstance(metrics, dict)
        
        # Test with too many positions
        max_positions = mock_config.trading.max_positions
        mock_positions = [
            {'symbol': f'TEST{i}', 'quantity': 100} 
            for i in range(max_positions)
        ]
        
        approved, reason, metrics = await risk_manager.evaluate_signal(
            signal, mock_positions, account_info
        )
        
        # Should be rejected due to position limit
        assert not approved
        assert 'position limit' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_portfolio_manager_integration(self, mock_config):
        """Test portfolio manager integration."""
        portfolio = PortfolioManager(mock_config.trading)
        
        # Mock position data from Alpaca
        alpaca_positions = [
            {
                'symbol': 'TQQQ',
                'quantity': 100,
                'side': 'long',
                'market_value': 5000.0,
                'avg_entry_price': 50.0,
                'unrealized_pl': 500.0,
                'unrealized_plpc': 0.1
            },
            {
                'symbol': 'SQQQ',
                'quantity': -50,
                'side': 'short',
                'market_value': -2000.0,
                'avg_entry_price': 40.0,
                'unrealized_pl': -200.0,
                'unrealized_plpc': -0.1
            }
        ]
        
        # Update positions
        await portfolio.update_positions(alpaca_positions)
        
        # Verify positions were updated
        positions = portfolio.get_all_positions()
        assert len(positions) == 2
        assert 'TQQQ' in positions
        assert 'SQQQ' in positions
        
        # Test position retrieval
        tqqq_position = portfolio.get_position('TQQQ')
        assert tqqq_position is not None
        assert tqqq_position['quantity'] == 100
        
        # Test portfolio metrics
        account_info = {
            'equity': 100000.0,
            'cash': 20000.0,
            'buying_power': 80000.0
        }
        
        metrics = await portfolio.calculate_portfolio_metrics(account_info)
        assert 'account' in metrics
        assert 'positions' in metrics
        assert 'exposure' in metrics
        assert 'risk' in metrics
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_flow(self, sample_market_data, mock_config):
        """Test complete analysis flow from data to signals."""
        # 1. Technical Analysis
        analyzer = TechnicalAnalyzer()
        technical_analysis = {}
        
        for symbol, data in sample_market_data.items():
            analysis = analyzer.analyze_symbol(data, symbol)
            technical_analysis[symbol] = analysis
        
        # 2. Strategy Analysis
        strategy = RSIStrategy({
            'min_confidence': 40,  # Lower threshold for testing
            'oversold_threshold': 35,
            'overbought_threshold': 65
        })
        
        signals = await strategy.analyze(
            market_data=sample_market_data,
            technical_analysis=technical_analysis,
            ai_analysis=None
        )
        
        # 3. Risk Management (if signals generated)
        if signals:
            risk_manager = RiskManager(mock_config.trading)
            
            current_positions = []
            account_info = {
                'equity': 100000.0,
                'buying_power': 80000.0,
                'cash': 20000.0,
                'portfolio_value': 100000.0,
                'day_trade_count': 0,
                'pattern_day_trader': False
            }
            
            for signal in signals:
                approved, reason, metrics = await risk_manager.evaluate_signal(
                    signal, current_positions, account_info
                )
                
                # Verify risk evaluation completed
                assert isinstance(approved, bool)
                assert isinstance(reason, str)
                assert isinstance(metrics, dict)
        
        # Verify complete flow executed without errors
        assert isinstance(technical_analysis, dict)
        assert isinstance(signals, list)
    
    def test_configuration_validation(self, mock_config):
        """Test configuration validation."""
        # Test valid configuration
        assert mock_config.alpaca.api_key == "test_key"
        assert mock_config.trading.max_positions == 2
        assert len(mock_config.trading.symbols) == 2
        
        # Test configuration constraints
        assert 0 < mock_config.trading.position_size <= 1
        assert mock_config.trading.stop_loss > 0
        assert mock_config.trading.take_profit > 0
        assert 0 <= mock_config.trading.min_confidence <= 100
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various components."""
        # Test technical analysis with insufficient data
        analyzer = TechnicalAnalyzer()
        
        # Create minimal data (less than required for indicators)
        minimal_data = pd.DataFrame({
            'open': [50.0, 51.0],
            'high': [52.0, 53.0],
            'low': [49.0, 50.0],
            'close': [51.0, 52.0],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1H'))
        
        analysis = analyzer.analyze_symbol(minimal_data, "TEST")
        
        # Should handle gracefully with error or empty results
        assert 'error' in analysis or 'indicators' in analysis
        
        # Test strategy with invalid data
        strategy = RSIStrategy()
        
        invalid_technical_analysis = {
            'TEST': {'error': 'Insufficient data'}
        }
        
        signals = await strategy.analyze(
            market_data={},
            technical_analysis=invalid_technical_analysis,
            ai_analysis=None
        )
        
        # Should return empty signals list
        assert isinstance(signals, list)
        assert len(signals) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
