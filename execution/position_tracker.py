import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from config.settings import TradingConfig
from utils.logger import setup_logger
from utils.helpers import format_currency, format_percentage

class PositionTracker:
    """Track and manage individual positions."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logger(None, "PositionTracker")
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
    
    async def update_position(
        self, 
        symbol: str, 
        quantity: int, 
        avg_price: float,
        current_price: Optional[float] = None
    ) -> None:
        """Update or create a position."""
        try:
            if quantity == 0:
                # Position closed
                if symbol in self.positions:
                    closed_position = self.positions.pop(symbol)
                    closed_position['closed_at'] = datetime.now()
                    closed_position['status'] = 'closed'
                    self.position_history.append(closed_position)
                    self.logger.info(f"Position closed: {symbol}")
                return
            
            # Update existing or create new position
            if symbol in self.positions:
                # Update existing position
                old_quantity = self.positions[symbol]['quantity']
                old_avg_price = self.positions[symbol]['avg_price']
                
                # Calculate new average price if adding to position
                if (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                    total_value = (old_quantity * old_avg_price) + (quantity * avg_price)
                    new_quantity = old_quantity + quantity
                    new_avg_price = total_value / new_quantity if new_quantity != 0 else avg_price
                    
                    self.positions[symbol].update({
                        'quantity': new_quantity,
                        'avg_price': new_avg_price,
                        'last_updated': datetime.now()
                    })
                else:
                    # Position direction change or complete replacement
                    self.positions[symbol].update({
                        'quantity': quantity,
                        'avg_price': avg_price,
                        'last_updated': datetime.now()
                    })
            else:
                # Create new position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price or avg_price,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_percent': 0.0,
                    'side': 'long' if quantity > 0 else 'short',
                    'opened_at': datetime.now(),
                    'last_updated': datetime.now(),
                    'status': 'open'
                }
                self.logger.info(f"New position opened: {symbol} {quantity} shares at ${avg_price:.2f}")
            
            # Update current price and PnL if provided
            if current_price:
                await self.update_position_price(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
    
    async def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update the current price and PnL for a position."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            quantity = position['quantity']
            avg_price = position['avg_price']
            
            # Calculate unrealized PnL
            if quantity > 0:  # Long position
                unrealized_pnl = (current_price - avg_price) * quantity
            else:  # Short position
                unrealized_pnl = (avg_price - current_price) * abs(quantity)
            
            unrealized_pnl_percent = (unrealized_pnl / (avg_price * abs(quantity))) * 100 if avg_price > 0 else 0
            
            # Update position data
            position.update({
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_percent': unrealized_pnl_percent,
                'market_value': current_price * quantity,
                'last_updated': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error updating position price for {symbol}: {e}")
    
    async def update_all_prices(self, price_data: Dict[str, float]) -> None:
        """Update prices for all positions."""
        try:
            for symbol, current_price in price_data.items():
                if symbol in self.positions:
                    await self.update_position_price(symbol, current_price)
        except Exception as e:
            self.logger.error(f"Error updating all position prices: {e}")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a specific symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        return self.positions.copy()
    
    def get_position_symbols(self) -> List[str]:
        """Get list of symbols with active positions."""
        return list(self.positions.keys())
    
    def get_long_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all long positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos['quantity'] > 0}
    
    def get_short_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all short positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos['quantity'] < 0}
    
    def calculate_total_pnl(self) -> Dict[str, float]:
        """Calculate total unrealized PnL across all positions."""
        try:
            if not self.positions:
                return {
                    'total_unrealized_pnl': 0.0,
                    'total_unrealized_pnl_percent': 0.0,
                    'total_market_value': 0.0
                }
            
            total_pnl = sum(pos.get('unrealized_pnl', 0.0) for pos in self.positions.values())
            total_market_value = sum(abs(pos.get('market_value', 0.0)) for pos in self.positions.values())
            
            total_pnl_percent = 0.0
            if total_market_value > 0:
                total_pnl_percent = (total_pnl / total_market_value * 100)
            
            return {
                'total_unrealized_pnl': total_pnl,
                'total_unrealized_pnl_percent': total_pnl_percent,
                'total_market_value': total_market_value
            }
        except Exception as e:
            self.logger.error(f"Error calculating total PnL: {e}")
            return {
                'total_unrealized_pnl': 0.0, 
                'total_unrealized_pnl_percent': 0.0, 
                'total_market_value': 0.0
            }
    
    def get_position_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics for all positions."""
        try:
            if not self.positions:
                return {}
            
            # Calculate position sizes relative to total
            total_market_value = sum(abs(pos['market_value']) for pos in self.positions.values())
            
            position_weights = {}
            max_position_weight = 0.0
            
            for symbol, position in self.positions.items():
                weight = abs(position['market_value']) / total_market_value if total_market_value > 0 else 0
                position_weights[symbol] = weight
                max_position_weight = max(max_position_weight, weight)
            
            # Calculate sector concentration (simplified)
            sector_exposure = self._calculate_sector_exposure()
            
            return {
                'position_count': len(self.positions),
                'max_position_weight': max_position_weight,
                'position_weights': position_weights,
                'sector_exposure': sector_exposure,
                'long_short_ratio': self._calculate_long_short_ratio()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk metrics: {e}")
            return {}
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate exposure by sector (simplified classification)."""
        try:
            sector_map = {
                'TQQQ': 'Technology Bull', 'SQQQ': 'Technology Bear',
                'SPXL': 'Broad Market Bull', 'SPXS': 'Broad Market Bear',
                'QQQ': 'Technology', 'SPY': 'Broad Market',
                'AAPL': 'Technology', 'MSFT': 'Technology',
                'GOOGL': 'Technology', 'AMZN': 'Consumer Discretionary',
                'TSLA': 'Automotive', 'NVDA': 'Technology'
            }
            
            sector_exposure = {}
            total_value = sum(abs(pos['market_value']) for pos in self.positions.values())
            
            for symbol, position in self.positions.items():
                sector = sector_map.get(symbol, 'Other')
                exposure = abs(position['market_value']) / total_value if total_value > 0 else 0
                
                if sector in sector_exposure:
                    sector_exposure[sector] += exposure
                else:
                    sector_exposure[sector] = exposure
            
            return sector_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating sector exposure: {e}")
            return {}
    
    def _calculate_long_short_ratio(self) -> float:
        """Calculate long/short ratio."""
        try:
            long_value = sum(pos['market_value'] for pos in self.positions.values() if pos['quantity'] > 0)
            short_value = sum(abs(pos['market_value']) for pos in self.positions.values() if pos['quantity'] < 0)
            
            if short_value == 0:
                return float('inf') if long_value > 0 else 0.0
            
            return long_value / short_value
            
        except Exception as e:
            self.logger.error(f"Error calculating long/short ratio: {e}")
            return 0.0
    
    def get_positions_by_performance(self, sort_by: str = 'pnl_percent') -> List[Dict[str, Any]]:
        """Get positions sorted by performance."""
        try:
            positions_list = list(self.positions.values())
            
            if sort_by == 'pnl_percent':
                positions_list.sort(key=lambda x: x.get('unrealized_pnl_percent', 0), reverse=True)
            elif sort_by == 'pnl_absolute':
                positions_list.sort(key=lambda x: x.get('unrealized_pnl', 0), reverse=True)
            elif sort_by == 'market_value':
                positions_list.sort(key=lambda x: abs(x.get('market_value', 0)), reverse=True)
            
            return positions_list
            
        except Exception as e:
            self.logger.error(f"Error sorting positions by {sort_by}: {e}")
            return list(self.positions.values())
    
    def print_positions_summary(self) -> None:
        """Print a formatted summary of all positions."""
        try:
            if not self.positions:
                print("No active positions")
                return
            
            print("\n" + "="*80)
            print("POSITION SUMMARY")
            print("="*80)
            
            total_pnl_data = self.calculate_total_pnl()
            print(f"Total Unrealized P&L: {format_currency(total_pnl_data['total_unrealized_pnl'])} "
                  f"({format_percentage(total_pnl_data['total_unrealized_pnl_percent']/100)})")
            print(f"Total Market Value: {format_currency(total_pnl_data['total_market_value'])}")
            print()
            
            # Sort by performance
            sorted_positions = self.get_positions_by_performance('pnl_percent')
            
            print(f"{'Symbol':<8} {'Side':<6} {'Qty':<8} {'Avg Price':<12} {'Current':<12} {'P&L':<12} {'P&L %':<10}")
            print("-" * 80)
            
            for position in sorted_positions:
                symbol = position['symbol']
                side = position['side'].upper()
                quantity = position['quantity']
                avg_price = position['avg_price']
                current_price = position['current_price']
                pnl = position['unrealized_pnl']
                pnl_percent = position['unrealized_pnl_percent']
                
                print(f"{symbol:<8} {side:<6} {quantity:<8.0f} {format_currency(avg_price):<12} "
                      f"{format_currency(current_price):<12} {format_currency(pnl):<12} "
                      f"{format_percentage(pnl_percent/100):<10}")
            
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Error printing positions summary: {e}")
    
    def get_position_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get closed position history."""
        return self.position_history[-limit:] if self.position_history else []
    
    def clear_old_history(self, days: int = 30) -> None:
        """Clear position history older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            self.position_history = [
                pos for pos in self.position_history
                if pos.get('closed_at', datetime.now()) > cutoff_date
            ]
            self.logger.info(f"Cleared position history older than {days} days")
        except Exception as e:
            self.logger.error(f"Error clearing old history: {e}")
