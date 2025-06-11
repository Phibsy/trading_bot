import asyncio
import signal
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Config imports
from config.settings import BotConfig
from config.constants import SignalType

# Utils
from utils.logger import setup_logger
from utils.helpers import is_market_hours, PerformanceTimer

class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = setup_logger(config.logging, "TradingBot")
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize components with lazy imports to avoid circular imports
        self.market_data = None
        self.database = None
        self.risk_manager = None
        self.portfolio = None
        self.order_manager = None
        self.technical_analyzer = None
        self.groq_analyzer = None
        self.strategies: List[Any] = []
        
        # Performance tracking
        self.last_analysis_time = datetime.now()
        self.analysis_count = 0
        
        self.logger.info("Trading bot initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize components with lazy imports."""
        try:
            # Import here to avoid circular imports
            from core.risk_manager import RiskManager
            from core.portfolio import PortfolioManager
            from data.market_data import MarketDataManager
            from data.database import DatabaseManager
            from analysis.technical import TechnicalAnalyzer
            from analysis.groq_analyzer import GroqAnalyzer
            from execution.order_manager import OrderManager
            
            # Initialize components
            self.market_data = MarketDataManager(self.config.alpaca)
            self.database = DatabaseManager(self.config.database.url)
            self.risk_manager = RiskManager(self.config.trading)
            self.portfolio = PortfolioManager(self.config.trading)
            self.order_manager = OrderManager(self.config.alpaca, self.config.trading)
            
            # Initialize analyzers
            self.technical_analyzer = TechnicalAnalyzer()
            self.groq_analyzer = GroqAnalyzer(self.config.groq) if self.config.groq.api_key else None
            
            # Initialize strategies
            self._initialize_strategies()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        try:
            from strategies.rsi_strategy import RSIStrategy
            from strategies.ml_strategy import MLStrategy
            
            # RSI Strategy
            rsi_strategy = RSIStrategy({
                'min_confidence': self.config.trading.min_confidence,
                'oversold_threshold': 25,  # More aggressive
                'overbought_threshold': 75
            })
            self.strategies.append(rsi_strategy)
            
            # ML Strategy (only if Groq is available)
            if self.groq_analyzer:
                ml_strategy = MLStrategy({
                    'min_confidence': self.config.trading.min_confidence + 5,  # Higher threshold
                    'ai_weight': 0.4
                })
                self.strategies.append(ml_strategy)
            
            self.logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {e}")
    
    async def start(self) -> None:
        """Start the trading bot."""
        try:
            self.running = True
            self.logger.info("Starting trading bot...")
            
            # Initialize components
            self._initialize_components()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Validate configuration
            if not await self._validate_configuration():
                return
            
            # Main trading loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
        finally:
            await self.shutdown()
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        self.logger.info("Entering main trading loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check if market is open
                if not is_market_hours():
                    self.logger.debug("Market closed, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                with PerformanceTimer("Full analysis cycle"):
                    # Run analysis and trading cycle
                    await self._trading_cycle()
                
                self.analysis_count += 1
                
                # Update pending orders
                await self.order_manager.update_pending_orders()
                
                # Print status every 10 cycles
                if self.analysis_count % 10 == 0:
                    await self._print_status()
                
                # Wait before next cycle (default: 1 minute)
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        try:
            # 1. Get market data
            symbols = self.config.trading.symbols
            market_data = await self.market_data.get_bars(symbols, timeframe="1Min", limit=200)
            
            if not market_data:
                self.logger.warning("No market data received")
                return
            
            # 2. Get account and position information
            account_info = await self.market_data.get_account_info()
            current_positions = await self.market_data.get_positions()
            
            # Update portfolio
            await self.portfolio.update_positions(current_positions)
            
            # 3. Perform technical analysis
            technical_analysis = {}
            for symbol, data in market_data.items():
                if not data.empty and len(data) >= 50:
                    analysis = self.technical_analyzer.analyze_symbol(data, symbol)
                    technical_analysis[symbol] = analysis
            
            # 4. Perform AI analysis if available
            ai_analysis = None
            if self.groq_analyzer and technical_analysis:
                ai_analysis = await self.groq_analyzer.batch_analyze(technical_analysis)
            
            # 5. Generate signals from strategies
            all_signals = []
            for strategy in self.strategies:
                if strategy.enabled:
                    try:
                        signals = await strategy.analyze(
                            market_data, technical_analysis, ai_analysis
                        )
                        # Validate signals are TradingSignal objects
                        for signal in signals:
                            if hasattr(signal, 'symbol') and hasattr(signal, 'signal_type'):
                                all_signals.append(signal)
                            else:
                                self.logger.warning(f"Invalid signal from {strategy.name}: {signal}")
                    except Exception as e:
                        self.logger.error(f"Error in strategy {strategy.name}: {e}")
            
            # 6. Process signals through risk management
            if all_signals:
                await self._process_signals(all_signals, current_positions, account_info)
            
            # 7. Update performance metrics
            await self.portfolio.calculate_portfolio_metrics(account_info, market_data)
            
            self.last_analysis_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    async def _process_signals(
        self, 
        signals: List[Any], 
        current_positions: List[Dict[str, Any]], 
        account_info: Dict[str, Any]
    ) -> None:
        """Process trading signals through risk management and execution."""
        try:
            # Import here to avoid circular imports
            from strategies.base_strategy import TradingSignal
            from config.constants import SignalType
            
            # Sort signals by confidence (highest first)
            valid_signals = []
            for signal in signals:
                if hasattr(signal, 'confidence') and hasattr(signal, 'symbol'):
                    valid_signals.append(signal)
                else:
                    self.logger.warning(f"Invalid signal object: {signal}")
            
            valid_signals.sort(key=lambda s: getattr(s, 'confidence', 0), reverse=True)
            
            for signal in valid_signals:
                # Check risk management
                approved, reason, risk_metrics = await self.risk_manager.evaluate_signal(
                    signal, current_positions, account_info
                )
                
                if not approved:
                    self.logger.info(f"Signal rejected for {getattr(signal, 'symbol', 'UNKNOWN')}: {reason}")
                    continue
                
                # Calculate position size
                symbol = getattr(signal, 'symbol', '')
                price = getattr(signal, 'price', 0)
                
                if not symbol or price <= 0:
                    self.logger.warning(f"Invalid signal data: symbol={symbol}, price={price}")
                    continue
                
                position_size = await self.portfolio.calculate_position_size(
                    symbol, price, account_info
                )
                
                if position_size <= 0:
                    self.logger.warning(f"Invalid position size for {symbol}")
                    continue
                
                # Calculate stop loss and take profit prices
                signal_type = getattr(signal, 'signal_type', None)
                stop_loss_price = None
                take_profit_price = None
                
                if signal_type == SignalType.BUY:
                    stop_loss_price = price * (1 - self.config.trading.stop_loss)
                    take_profit_price = price * (1 + self.config.trading.take_profit)
                elif signal_type == SignalType.SELL:
                    # For short positions (selling)
                    stop_loss_price = price * (1 + self.config.trading.stop_loss)
                    take_profit_price = price * (1 - self.config.trading.take_profit)
                
                # Execute the trade
                order_result = await self.order_manager.execute_signal(
                    signal=signal,
                    quantity=position_size,
                    order_type="market",
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                
                if order_result:
                    # Save signal to database
                    confidence = getattr(signal, 'confidence', 0)
                    metadata = getattr(signal, 'metadata', {})
                    
                    await self.database.save_signal({
                        'symbol': symbol,
                        'strategy': metadata.get('strategy', 'unknown'),
                        'signal_type': signal_type.value if signal_type else 'UNKNOWN',
                        'confidence': confidence,
                        'price': price,
                        'metadata': metadata
                    })
                    
                    self.logger.info(
                        f"Trade executed: {symbol} {signal_type.value if signal_type else 'UNKNOWN'} "
                        f"{position_size} shares at ${price:.2f} "
                        f"(confidence: {confidence:.1f}%)"
                    )
                else:
                    self.logger.error(f"Failed to execute trade for {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def _validate_configuration(self) -> bool:
        """Validate bot configuration."""
        try:
            # Check if components are initialized
            if not self.market_data:
                self.logger.error("Market data manager not initialized")
                return False
                
            # Check Alpaca credentials
            account_info = await self.market_data.get_account_info()
            if not account_info:
                self.logger.error("Invalid Alpaca credentials or connection failed")
                return False
            
            self.logger.info(f"Account validated - Equity: ${account_info.get('equity', 0):,.2f}")
            
            # Check Groq credentials if enabled
            if self.groq_analyzer and self.groq_analyzer.client:
                # Simple test to validate Groq API
                test_data = {
                    'current_price': 100,
                    'indicators': {
                        'rsi': {'value': 50, 'signal': 'NEUTRAL'},
                        'bollinger': {'percent': 0.5, 'signal': 'NEUTRAL'},
                        'macd': {'histogram': 0, 'signal': 'NEUTRAL'},
                        'volume': {'ratio': 1.0, 'signal': 'NORMAL'},
                        'trend': {'adx': 20, 'slope': 0, 'roc': 0}
                    }
                }
                test_result = await self.groq_analyzer.analyze_technical_data("TEST", test_data)
                if test_result.get('error'):
                    self.logger.warning("Groq API test failed, continuing without AI analysis")
                    self.groq_analyzer = None
                else:
                    self.logger.info("Groq AI validated successfully")
            elif self.groq_analyzer:
                self.logger.warning("Groq analyzer created but client not available")
                self.groq_analyzer = None
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _print_status(self) -> None:
        """Print current bot status."""
        try:
            account_info = await self.market_data.get_account_info()
            portfolio_metrics = await self.portfolio.calculate_portfolio_metrics(account_info)
            
            print(f"\n{'='*80}")
            print(f"TRADING BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            print(f"Analysis Cycles: {self.analysis_count}")
            print(f"Active Strategies: {len([s for s in self.strategies if s.enabled])}")
            print(f"Account Equity: ${account_info.get('equity', 0):,.2f}")
            print(f"Active Positions: {portfolio_metrics.get('positions', {}).get('count', 0)}")
            print(f"Pending Orders: {len(self.order_manager.get_pending_orders())}")
            
            # Print positions if any
            if self.portfolio.get_position_count() > 0:
                self.portfolio.print_portfolio_status(account_info)
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            self.logger.error(f"Error printing status: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the trading bot."""
        try:
            self.logger.info("Shutting down trading bot...")
            self.running = False
            
            # Cancel all pending orders
            canceled_orders = await self.order_manager.cancel_all_orders()
            if canceled_orders > 0:
                self.logger.info(f"Canceled {canceled_orders} pending orders")
            
            # Final status report
            account_info = await self.market_data.get_account_info()
            if account_info:
                await self._print_status()
            
            self.logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Public methods for external control
    
    def add_strategy(self, strategy: Any) -> None:
        """Add a new strategy to the bot."""
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name."""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                del self.strategies[i]
                self.logger.info(f"Removed strategy: {strategy_name}")
                return True
        return False
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy by name."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enable()
                self.logger.info(f"Enabled strategy: {strategy_name}")
                return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy by name."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.disable()
                self.logger.info(f"Disabled strategy: {strategy_name}")
                return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            'running': self.running,
            'analysis_count': self.analysis_count,
            'last_analysis': self.last_analysis_time,
            'strategies': [{
                'name': s.name,
                'enabled': s.enabled,
                'parameters': s.parameters
            } for s in self.strategies],
            'positions': self.portfolio.get_all_positions(),
            'pending_orders': len(self.order_manager.get_pending_orders())
        }
