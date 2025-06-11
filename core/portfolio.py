import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config.settings import TradingConfig
from utils.logger import setup_logger
from utils.helpers import calculate_position_size, format_currency, format_percentage

class PortfolioManager:
    """Portfolio management and tracking system."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logger(None, "Portfolio")
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.last_update = datetime.now()
    
    async def update_positions(self, alpaca_positions: List[Dict[str, Any]]) -> None:
        """Update positions from Alpaca API."""
        try:
            # Clear existing positions
            self.positions.clear()
            
            for pos in alpaca_positions:
                symbol = pos['symbol']
                quantity = pos['quantity']
                
                if quantity != 0:  # Only track non-zero positions
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'side': pos['side'],
                        'market_value': pos['market_value'],
                        'avg_entry_price': pos['avg_entry_price'],
                        'unrealized_pl': pos['unrealized_pl'],
                        'unrealized_plpc': pos['unrealized_plpc'],
                        'last_updated': datetime.now()
                    }
            
            self.last_update = datetime.now()
            self.logger.info(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        signal_price: float, 
        account_info: Dict[str, Any]
    ) -> int:
        """Calculate optimal position size for a new trade."""
        try:
            account_value = account_info.get('equity', 0)
            if account_value <= 0:
                return 0
            
            # Use helper function for position sizing
            shares = calculate_position_size(
                account_value=account_value,
                position_size_pct=self.config.position_size,
                current_price=signal_price,
                stop_loss_pct=self.config.stop_loss
            )
            
            self.logger.debug(f"Calculated position size for {symbol}: {shares} shares")
            return shares
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of active positions."""
        return len(self.positions)
    
    async def calculate_portfolio_metrics(
        self, 
        account_info: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        try:
            # Basic account metrics
            total_equity = account_info.get('equity', 0)
            cash = account_info.get('cash', 0)
            buying_power = account_info.get('buying_power', 0)
            
            # Position metrics
            total_market_value = sum(pos.get('market_value', 0) for pos in self.positions.values())
            total_unrealized_pl = sum(pos.get('unrealized_pl', 0) for pos in self.positions.values())
            
            # Calculate exposure
            long_exposure = sum(
                pos.get('market_value', 0) for pos in self.positions.values() 
                if pos.get('quantity', 0) > 0
            )
            short_exposure = sum(
                abs(pos.get('market_value', 0)) for pos in self.positions.values() 
                if pos.get('quantity', 0) < 0
            )
            
            # Risk metrics
            portfolio_beta = await self._calculate_portfolio_beta(market_data)
            concentration_risk = await self._calculate_concentration_risk()
            
            # Safe percentage calculations
            unrealized_pl_percent = 0.0
            cash_percentage = 0.0
            
            if total_equity > 0:
                unrealized_pl_percent = (total_unrealized_pl / total_equity * 100)
                cash_percentage = (cash / total_equity * 100)
            
            metrics = {
                'timestamp': datetime.now(),
                'account': {
                    'total_equity': total_equity,
                    'cash': cash,
                    'buying_power': buying_power,
                    'invested_amount': total_market_value
                },
                'positions': {
                    'count': len(self.positions),
                    'total_market_value': total_market_value,
                    'unrealized_pl': total_unrealized_pl,
                    'unrealized_pl_percent': unrealized_pl_percent
                },
                'exposure': {
                    'long_exposure': long_exposure,
                    'short_exposure': short_exposure,
                    'net_exposure': long_exposure - short_exposure,
                    'gross_exposure': long_exposure + short_exposure
                },
                'risk': {
                    'portfolio_beta': portfolio_beta,
                    'concentration_risk': concentration_risk,
                    'cash_percentage': cash_percentage
                }
            }
            
            # Store for history tracking
            self.performance_history.append(metrics)
            
            # Keep only last 30 days of history
            cutoff_date = datetime.now() - timedelta(days=30)
            self.performance_history = [
                m for m in self.performance_history 
                if m.get('timestamp', datetime.now()) > cutoff_date
            ]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'timestamp': datetime.now(),
                'account': {'total_equity': 0, 'cash': 0, 'buying_power': 0, 'invested_amount': 0},
                'positions': {'count': 0, 'total_market_value': 0, 'unrealized_pl': 0, 'unrealized_pl_percent': 0},
                'exposure': {'long_exposure': 0, 'short_exposure': 0, 'net_exposure': 0, 'gross_exposure': 0},
                'risk': {'portfolio_beta': 1.0, 'concentration_risk': 0.0, 'cash_percentage': 0}
            }
    
    async def _calculate_portfolio_beta(self, market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate portfolio beta (simplified)."""
        try:
            if not market_data or not self.positions:
                return 1.0
            
            # Simplified beta calculation using position weights
            # In a real implementation, you'd calculate beta against a benchmark
            total_value = sum(pos['market_value'] for pos in self.positions.values())
            
            if total_value == 0:
                return 1.0
            
            # Assign approximate betas for different symbol types
            beta_map = {
                'TQQQ': 3.0, 'SQQQ': -3.0,  # 3x leveraged ETFs
                'SPXL': 3.0, 'SPXS': -3.0,
                'QQQ': 1.2, 'SPY': 1.0,      # Index ETFs
                'AAPL': 1.3, 'MSFT': 1.1,   # Individual stocks (approximate)
                'GOOGL': 1.4, 'AMZN': 1.5,
                'TSLA': 2.0, 'NVDA': 1.8
            }
            
            weighted_beta = 0.0
            for symbol, position in self.positions.items():
                weight = position['market_value'] / total_value
                symbol_beta = beta_map.get(symbol, 1.0)
                weighted_beta += weight * symbol_beta
            
            return weighted_beta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    async def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (Herfindahl-Hirschman Index)."""
        try:
            if not self.positions:
                return 0.0
            
            total_value = sum(abs(pos['market_value']) for pos in self.positions.values())
            
            if total_value == 0:
                return 0.0
            
            # Calculate HHI
            hhi = sum(
                (abs(pos['market_value']) / total_value) ** 2 
                for pos in self.positions.values()
            )
            
            return hhi
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        try:
            if not self.performance_history:
                return {}
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_history = [
                m for m in self.performance_history 
                if m['timestamp'] > cutoff_date
            ]
            
            if len(recent_history) < 2:
                return {}
            
            # Calculate changes over the period
            start_metrics = recent_history[0]
            end_metrics = recent_history[-1]
            
            start_equity = start_metrics['account']['total_equity']
            end_equity = end_metrics['account']['total_equity']
            
            total_return = ((end_equity - start_equity) / start_equity * 100) if start_equity > 0 else 0
            
            # Calculate daily returns for volatility
            daily_returns = []
            for i in range(1, len(recent_history)):
                prev_equity = recent_history[i-1]['account']['total_equity']
                curr_equity = recent_history[i]['account']['total_equity']
                daily_return = ((curr_equity - prev_equity) / prev_equity) if prev_equity > 0 else 0
                daily_returns.append(daily_return)
            
            volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0  # Annualized
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if daily_returns and np.std(daily_returns) > 0 else 0
            
            return {
                'period_days': days,
                'total_return_percent': total_return,
                'annualized_volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'start_equity': start_equity,
                'end_equity': end_equity,
                'max_positions': max(m['positions']['count'] for m in recent_history),
                'avg_positions': np.mean([m['positions']['count'] for m in recent_history])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def print_portfolio_status(self, account_info: Dict[str, Any]) -> None:
        """Print formatted portfolio status to console."""
        try:
            print("\n" + "="*60)
            print("PORTFOLIO STATUS")
            print("="*60)
            
            # Account summary
            equity = account_info.get('equity', 0)
            cash = account_info.get('cash', 0)
            buying_power = account_info.get('buying_power', 0)
            
            print(f"Account Equity: {format_currency(equity)}")
            print(f"Cash: {format_currency(cash)}")
            print(f"Buying Power: {format_currency(buying_power)}")
            
            # Positions
            if self.positions:
                print(f"\nActive Positions ({len(self.positions)}):")
                print("-" * 60)
                
                total_unrealized_pl = 0
                for symbol, pos in self.positions.items():
                    quantity = pos['quantity']
                    market_value = pos['market_value']
                    unrealized_pl = pos['unrealized_pl']
                    unrealized_plpc = pos['unrealized_plpc']
                    
                    total_unrealized_pl += unrealized_pl
                    
                    side_str = "LONG" if quantity > 0 else "SHORT"
                    print(f"{symbol:6} | {side_str:5} | {abs(quantity):4.0f} shares | "
                          f"{format_currency(market_value):>12} | "
                          f"{format_currency(unrealized_pl):>10} ({format_percentage(unrealized_plpc/100):>8})")
                
                print("-" * 60)
                print(f"Total Unrealized P&L: {format_currency(total_unrealized_pl)}")
                
            else:
                print("\nNo active positions")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Error printing portfolio status: {e}")
