import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import alpaca_trade_api as tradeapi
from config.settings import AlpacaConfig, TradingConfig
from config.constants import OrderSide, OrderType, TimeInForce, SignalType
from utils.logger import setup_logger
from utils.helpers import retry_async
from strategies.base_strategy import TradingSignal

class OrderManager:
    """Order execution and management system."""
    
    def __init__(self, alpaca_config: AlpacaConfig, trading_config: TradingConfig):
        self.alpaca_config = alpaca_config
        self.trading_config = trading_config
        self.api = tradeapi.REST(
            alpaca_config.api_key,
            alpaca_config.secret_key,
            alpaca_config.base_url,
            api_version='v2'
        )
        self.logger = setup_logger(None, "OrderManager")
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
    
    @retry_async(max_retries=3, delay=1.0)
    async def execute_signal(
        self, 
        signal: Any,  # Using Any to avoid circular import 
        quantity: int,
        order_type: str = "market",
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a trading signal."""
        try:
            # Import here to avoid circular imports
            from config.constants import SignalType, OrderSide
            
            # Extract signal attributes safely
            signal_type = getattr(signal, 'signal_type', None)
            symbol = getattr(signal, 'symbol', '')
            
            if not symbol:
                self.logger.error("Signal missing symbol")
                return None
            
            # Determine order side
            if signal_type == SignalType.BUY:
                side = OrderSide.BUY
            elif signal_type == SignalType.SELL:
                side = OrderSide.SELL
            else:
                self.logger.error(f"Invalid signal type: {signal_type}")
                return None
            
            # Create main order
            order_result = await self._submit_order(
                symbol=symbol,
                quantity=quantity,
                side=side.value,
                order_type=order_type,
                time_in_force="day",
                extended_hours=True  # Enable extended hours for European trading
            )
            
            if not order_result:
                return None
            
            order_id = order_result['id']
            
            # Create stop loss and take profit orders if specified
            bracket_orders = []
            if side == OrderSide.BUY:  # Long position
                if stop_loss_price:
                    stop_order = await self._submit_order(
                        symbol=symbol,
                        quantity=quantity,
                        side=OrderSide.SELL.value,
                        order_type="stop",
                        stop_price=stop_loss_price,
                        time_in_force="gtc",
                        extended_hours=True
                    )
                    if stop_order:
                        bracket_orders.append(stop_order)
                
                if take_profit_price:
                    profit_order = await self._submit_order(
                        symbol=symbol,
                        quantity=quantity,
                        side=OrderSide.SELL.value,
                        order_type="limit",
                        limit_price=take_profit_price,
                        time_in_force="gtc",
                        extended_hours=True
                    )
                    if profit_order:
                        bracket_orders.append(profit_order)
            
            # Track the order
            order_info = {
                'signal': {
                    'symbol': symbol,
                    'signal_type': str(signal_type),
                    'confidence': getattr(signal, 'confidence', 0),
                    'price': getattr(signal, 'price', 0),
                    'reasoning': getattr(signal, 'reasoning', ''),
                    'metadata': getattr(signal, 'metadata', {}),
                    'timestamp': getattr(signal, 'timestamp', None)
                },
                'main_order': order_result,
                'bracket_orders': bracket_orders,
                'timestamp': datetime.now(),
                'status': 'submitted'
            }
            
            self.pending_orders[order_id] = order_info
            self.order_history.append(order_info)
            
            self.logger.info(f"Order executed: {symbol} {side.value} {quantity} shares (Extended Hours)")
            return order_info
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @retry_async(max_retries=3, delay=0.5)
    async def _submit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = True  # Enable extended hours by default for European trading
    ) -> Optional[Dict[str, Any]]:
        """Submit an order to Alpaca with extended hours support."""
        try:
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force,
                'extended_hours': extended_hours  # Add extended hours parameter
            }
            
            # During extended hours, only limit orders are allowed
            if extended_hours and order_type == "market":
                # Get current quote to convert market order to limit order
                loop = asyncio.get_event_loop()
                quote = await loop.run_in_executor(
                    None,
                    lambda: self.api.get_latest_quote(symbol)
                )
                
                # Use ask price for buy orders, bid price for sell orders
                if side == "buy":
                    limit_price = float(quote.ap) * 1.001  # Slightly above ask
                else:
                    limit_price = float(quote.bp) * 0.999  # Slightly below bid
                
                order_params['type'] = 'limit'
                order_params['limit_price'] = limit_price
                self.logger.info(f"Converting market order to limit order for extended hours: {limit_price}")
            
            if limit_price and order_type == "limit":
                order_params['limit_price'] = limit_price
            
            if stop_price:
                order_params['stop_price'] = stop_price
            
            # Use asyncio to run the sync API call in a thread
            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self.api.submit_order(**order_params)
            )
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side,
                'type': order.order_type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'extended_hours': extended_hours
            }
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    async def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of an order."""
        try:
            # Use asyncio to run the sync API call in a thread
            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self.api.get_order(order_id)
            )
            
            return {
                'id': order.id,
                'status': order.status,
                'filled_qty': int(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0),
                'updated_at': order.updated_at
            }
        except Exception as e:
            self.logger.error(f"Error checking order status {order_id}: {e}")
            return None
    
    async def update_pending_orders(self) -> None:
        """Update status of all pending orders."""
        completed_orders = []
        
        for order_id, order_info in self.pending_orders.items():
            try:
                status = await self.check_order_status(order_id)
                if status:
                    order_info['main_order'].update(status)
                    
                    # Check if order is completed
                    if status['status'] in ['filled', 'canceled', 'rejected']:
                        completed_orders.append(order_id)
                        self.logger.info(f"Order {order_id} completed with status: {status['status']}")
                        
                        # Cancel any related bracket orders if main order was canceled/rejected
                        if status['status'] in ['canceled', 'rejected']:
                            await self._cancel_bracket_orders(order_info.get('bracket_orders', []))
            
            except Exception as e:
                self.logger.error(f"Error updating order {order_id}: {e}")
        
        # Remove completed orders from pending
        for order_id in completed_orders:
            self.pending_orders.pop(order_id, None)
    
    async def _cancel_bracket_orders(self, bracket_orders: List[Dict[str, Any]]) -> None:
        """Cancel bracket orders (stop loss, take profit)."""
        for bracket_order in bracket_orders:
            try:
                order_id = bracket_order['id']
                # Use asyncio to run the sync API call in a thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.api.cancel_order(order_id)
                )
                self.logger.info(f"Canceled bracket order {order_id}")
            except Exception as e:
                self.logger.error(f"Error canceling bracket order {bracket_order.get('id')}: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        try:
            # Use asyncio to run the sync API call in a thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.api.cancel_order(order_id)
            )
            
            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = 'canceled'
                del self.pending_orders[order_id]
            
            self.logger.info(f"Order {order_id} canceled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        try:
            # Use asyncio to run the sync API call in a thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.api.cancel_all_orders()
            )
            
            canceled_count = len(self.pending_orders)
            self.pending_orders.clear()
            
            self.logger.info(f"Canceled {canceled_count} orders")
            return canceled_count
            
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return 0
    
    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending orders."""
        return self.pending_orders.copy()
    
    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history."""
        return self.order_history[-limit:] if self.order_history else []
    
    async def create_stop_loss_order(
        self, 
        symbol: str, 
        quantity: int, 
        stop_price: float,
        is_long_position: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Create a stop loss order for an existing position."""
        try:
            side = OrderSide.SELL if is_long_position else OrderSide.BUY
            
            return await self._submit_order(
                symbol=symbol,
                quantity=quantity,
                side=side.value,
                order_type="stop",
                stop_price=stop_price,
                time_in_force="gtc",
                extended_hours=True
            )
            
        except Exception as e:
            self.logger.error(f"Error creating stop loss order for {symbol}: {e}")
            return None
    
    async def create_take_profit_order(
        self, 
        symbol: str, 
        quantity: int, 
        limit_price: float,
        is_long_position: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Create a take profit order for an existing position."""
        try:
            side = OrderSide.SELL if is_long_position else OrderSide.BUY
            
            return await self._submit_order(
                symbol=symbol,
                quantity=quantity,
                side=side.value,
                order_type="limit",
                limit_price=limit_price,
                time_in_force="gtc",
                extended_hours=True
            )
            
        except Exception as e:
            self.logger.error(f"Error creating take profit order for {symbol}: {e}")
            return None
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics."""
        try:
            if not self.order_history:
                return {}
            
            total_orders = len(self.order_history)
            filled_orders = sum(1 for order in self.order_history 
                              if order.get('main_order', {}).get('status') == 'filled')
            
            success_rate = (filled_orders / total_orders * 100) if total_orders > 0 else 0
            
            # Calculate by signal type
            buy_orders = sum(1 for order in self.order_history 
                           if order.get('signal', {}).get('signal_type') == 'BUY')
            sell_orders = total_orders - buy_orders
            
            return {
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'success_rate_percent': success_rate,
                'buy_orders': buy_orders,
                'sell_orders': sell_orders,
                'pending_orders': len(self.pending_orders),
                'extended_hours_enabled': True
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order statistics: {e}")
            return {}
