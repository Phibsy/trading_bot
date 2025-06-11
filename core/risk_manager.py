import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config.settings import TradingConfig
from config.constants import SignalType
from utils.logger import setup_logger

class RiskManager:
    """Risk management system for the trading bot."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logger(None, "RiskManager")
        self.daily_pnl = 0.0
        self.max_daily_loss = config.daily_loss_limit
        self.position_correlations: Dict[str, float] = {}
        self.last_reset_date = datetime.now().date()
    
    async def evaluate_signal(
        self,
        signal: Any,  # Using Any to avoid circular import
        current_positions: List[Dict[str, Any]],
        account_info: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate if a trading signal should be executed.
        
        Returns:
            (approved, reason, risk_metrics)
        """
        # Reset daily PnL if new day
        self._check_daily_reset()
        
        # Run all risk checks
        checks = [
            self._check_daily_loss_limit(),
            self._check_position_limits(current_positions, signal),
            self._check_position_size(signal, account_info),
            self._check_correlation_risk(signal, current_positions, market_data),
            self._check_confidence_threshold(signal),
            self._check_market_conditions(market_data),
            self._check_account_status(account_info)
        ]
        
        # Execute all checks
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        risk_metrics = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Risk check {i} failed: {result}")
                return False, f"Risk check error: {result}", {}
            
            approved, reason, metrics = result
            risk_metrics.update(metrics)
            
            if not approved:
                self.logger.warning(f"Signal rejected: {reason}")
                return False, reason, risk_metrics
        
        self.logger.info(f"Signal approved for {signal.symbol}: {signal.signal_type.value}")
        return True, "All risk checks passed", risk_metrics
    
    async def _check_daily_loss_limit(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if daily loss limit is exceeded."""
        loss_pct = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        
        metrics = {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.max_daily_loss,
            'loss_percentage': loss_pct
        }
        
        if loss_pct >= self.max_daily_loss:
            return False, f"Daily loss limit exceeded: {loss_pct:.2%}", metrics
        
        return True, "Daily loss check passed", metrics
    
    async def _check_position_limits(
        self, 
        current_positions: List[Dict[str, Any]], 
        signal: Any
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check position limits."""
        active_positions = len([p for p in current_positions if p.get('quantity', 0) != 0])
        
        metrics = {
            'active_positions': active_positions,
            'max_positions': self.config.max_positions
        }
        
        signal_type = getattr(signal, 'signal_type', None)
        symbol = getattr(signal, 'symbol', '')
        
        # If buying and we're at max positions, reject
        if signal_type == SignalType.BUY and active_positions >= self.config.max_positions:
            # Check if we already have a position in this symbol
            existing_position = next((p for p in current_positions if p.get('symbol') == symbol), None)
            if not existing_position or existing_position.get('quantity', 0) == 0:
                return False, f"Max position limit reached: {active_positions}/{self.config.max_positions}", metrics
        
        return True, "Position limit check passed", metrics
    
    async def _check_position_size(
        self, 
        signal: Any, 
        account_info: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check position size constraints."""
        account_value = account_info.get('equity', 0)
        buying_power = account_info.get('buying_power', 0)
        
        if account_value <= 0:
            return False, "Invalid account value", {}
        
        signal_price = getattr(signal, 'price', 0)
        position_value = account_value * self.config.position_size
        required_buying_power = position_value / signal_price if signal_price > 0 else 0
        
        metrics = {
            'account_value': account_value,
            'buying_power': buying_power,
            'position_value': position_value,
            'required_buying_power': required_buying_power
        }
        
        signal_type = getattr(signal, 'signal_type', None)
        if signal_type == SignalType.BUY and required_buying_power > buying_power:
            return False, f"Insufficient buying power: {buying_power:.2f} < {required_buying_power:.2f}", metrics
        
        return True, "Position size check passed", metrics
    
    async def _check_correlation_risk(
        self, 
        signal: Any, 
        current_positions: List[Dict[str, Any]], 
        market_data: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check correlation risk between positions."""
        if not current_positions or not market_data:
            return True, "Correlation check skipped", {}
        
        signal_symbol = getattr(signal, 'symbol', '')
        signal_type = getattr(signal, 'signal_type', None)
        
        # Simple correlation check based on sector/type
        correlation_groups = {
            'TECH_BULL': ['TQQQ', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'],
            'TECH_BEAR': ['SQQQ'],
            'BROAD_BULL': ['SPXL', 'SPY'],
            'BROAD_BEAR': ['SPXS']
        }
        
        signal_group = None
        for group, symbols in correlation_groups.items():
            if signal_symbol in symbols:
                signal_group = group
                break
        
        if not signal_group:
            return True, "No correlation group found", {}
        
        # Check for conflicting positions
        conflicting_positions = []
        for position in current_positions:
            if position.get('quantity', 0) == 0:
                continue
                
            pos_symbol = position.get('symbol')
            for group, symbols in correlation_groups.items():
                if pos_symbol in symbols:
                    # Check for opposing positions (bull vs bear)
                    if (('BULL' in signal_group and 'BEAR' in group) or 
                        ('BEAR' in signal_group and 'BULL' in group)):
                        conflicting_positions.append(pos_symbol)
                    break
        
        metrics = {
            'signal_group': signal_group,
            'conflicting_positions': conflicting_positions
        }
        
        if conflicting_positions and signal_type == SignalType.BUY:
            return False, f"Correlation risk: conflicting positions {conflicting_positions}", metrics
        
        return True, "Correlation check passed", metrics
    
    async def _check_confidence_threshold(self, signal: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if signal confidence meets minimum threshold."""
        metrics = {
            'signal_confidence': getattr(signal, 'confidence', 0),
            'min_confidence': self.config.min_confidence
        }
        
        signal_confidence = getattr(signal, 'confidence', 0)
        if signal_confidence < self.config.min_confidence:
            return False, f"Confidence too low: {signal_confidence:.1f}% < {self.config.min_confidence:.1f}%", metrics
        
        return True, "Confidence check passed", metrics
    
    async def _check_market_conditions(self, market_data: Optional[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check general market conditions."""
        from utils.helpers import is_market_hours
        
        # Check market hours
        market_open = is_market_hours()
        
        now = datetime.now()
        metrics = {
            'is_weekday': now.weekday() < 5,
            'is_market_hours': market_open,
            'current_time': now,
            'hour': now.hour
        }
        
        if not market_open:
            if now.weekday() >= 5:
                return False, "Market closed: weekend", metrics
            else:
                return False, f"Market closed: outside trading hours ({now.hour}:00)", metrics
        
        return True, "Market conditions check passed", metrics
    
    async def _check_account_status(self, account_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check account status and restrictions."""
        day_trade_count = account_info.get('day_trade_count', 0)
        is_pdt = account_info.get('pattern_day_trader', False)
        account_value = account_info.get('equity', 0)
        
        metrics = {
            'day_trade_count': day_trade_count,
            'is_pattern_day_trader': is_pdt,
            'account_value': account_value
        }
        
        # PDT rule check (simplified)
        if not is_pdt and account_value < 25000 and day_trade_count >= 3:
            return False, "PDT rule violation risk", metrics
        
        return True, "Account status check passed", metrics
    
    def _check_daily_reset(self) -> None:
        """Reset daily tracking if new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self.logger.info("Daily risk metrics reset")
    
    def update_daily_pnl(self, pnl_change: float) -> None:
        """Update daily PnL tracking."""
        self.daily_pnl += pnl_change
        self.logger.debug(f"Daily PnL updated: {self.daily_pnl:.2f}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.max_daily_loss,
            'loss_percentage': abs(self.daily_pnl) if self.daily_pnl < 0 else 0,
            'last_reset_date': self.last_reset_date.isoformat()
        }
