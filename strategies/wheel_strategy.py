import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import alpaca_trade_api as tradeapi
from strategies.base_strategy import BaseStrategy, TradingSignal
from config.constants import SignalType
from utils.logger import setup_logger

class WheelStrategy(BaseStrategy):
    """
    Advanced Options Wheel Strategy implementation with full options support.
    
    The wheel strategy involves:
    1. Selling cash-secured puts (CSP) to potentially buy stocks at a discount
    2. If assigned, selling covered calls (CC) on the stock
    3. If called away, restart with CSP
    
    This implementation includes:
    - Options chain analysis
    - Greeks calculation (Delta, Gamma, Theta, Vega)
    - IV Rank/Percentile calculation
    - Actual options order execution
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_confidence': 70,
            'min_atr': 0.5,              # Minimum Average True Range for volatility
            'max_atr': 5.0,              # Maximum ATR to avoid too volatile stocks
            'min_premium_percentage': 0.01,  # Minimum 1% premium per trade
            'days_to_expiration_min': 25,    # Minimum DTE
            'days_to_expiration_max': 45,    # Maximum DTE
            'delta_target_put': -0.30,    # Target delta for puts (30 delta)
            'delta_target_call': 0.30,    # Target delta for calls
            'max_positions': 2,           # Max wheel positions
            'use_technical_filters': True,
            'rsi_oversold': 30,           # RSI threshold for put selling
            'rsi_overbought': 70,         # RSI threshold for call selling
            'iv_rank_min': 30,            # Minimum IV rank (percentile)
            'iv_percentile_lookback': 252, # Days for IV percentile calculation
            'min_bid_ask_spread': 0.05,   # Max bid-ask spread
            'min_open_interest': 100,      # Minimum open interest
            'min_volume': 10,              # Minimum daily volume
            'risk_free_rate': 0.05,        # Risk-free rate for Greeks
            'etf_whitelist': [            # ETFs suitable for wheel strategy
                'SPY', 'QQQ', 'IWM',     # Major indices
                'EWG', 'FEZ', 'VGK', 'EFA',  # Europe ETFs
                'GLD', 'SLV',            # Commodities
                'TLT', 'IEF',            # Bonds
                'XLF', 'XLE', 'XLK'      # Sectors
            ]
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Wheel Strategy", default_params)
        self.logger = setup_logger(None, "WheelStrategy")
        self.active_wheels: Dict[str, Dict[str, Any]] = {}
        self.iv_history: Dict[str, pd.DataFrame] = {}
        
        # Initialize Alpaca API for options
        self.api = None
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize Alpaca API for options trading."""
        try:
            from config.settings import AlpacaConfig
            config = AlpacaConfig()
            self.api = tradeapi.REST(
                config.api_key,
                config.secret_key,
                config.base_url,
                api_version='v2'
            )
            self.logger.info("Alpaca API initialized for options trading")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca API: {e}")
    
    async def analyze(
        self,
        market_data: Dict[str, pd.DataFrame],
        technical_analysis: Dict[str, Dict[str, Any]],
        ai_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[TradingSignal]:
        """Analyze market data and generate wheel strategy signals."""
        signals = []
        
        # Get eligible symbols for wheel strategy
        eligible_symbols = self._filter_eligible_symbols(
            list(market_data.keys()),
            technical_analysis
        )
        
        for symbol in eligible_symbols:
            if symbol not in market_data or market_data[symbol].empty:
                continue
            
            # Update IV history
            await self._update_iv_history(symbol)
            
            # Get options chain
            options_chain = await self._get_options_chain(symbol)
            if options_chain is None or options_chain.empty:
                continue
            
            # Check if we already have an active wheel position
            if symbol in self.active_wheels:
                # Check if we should sell covered calls
                signal = await self._analyze_covered_call_advanced(
                    symbol,
                    market_data[symbol],
                    technical_analysis.get(symbol, {}),
                    ai_analysis.get(symbol, {}) if ai_analysis else None,
                    options_chain
                )
            else:
                # Check if we should sell cash-secured puts
                signal = await self._analyze_cash_secured_put_advanced(
                    symbol,
                    market_data[symbol],
                    technical_analysis.get(symbol, {}),
                    ai_analysis.get(symbol, {}) if ai_analysis else None,
                    options_chain
                )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    async def _get_options_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get options chain data from Alpaca."""
        try:
            if not self.api:
                return None
            
            # Get current date and calculate expiration range
            today = datetime.now()
            min_expiry = today + timedelta(days=self.get_parameter('days_to_expiration_min'))
            max_expiry = today + timedelta(days=self.get_parameter('days_to_expiration_max'))
            
            # Fetch options contracts
            loop = asyncio.get_event_loop()
            contracts = await loop.run_in_executor(
                None,
                lambda: self.api.list_options_contracts(
                    underlying_symbols=symbol,
                    expiration_date_gte=min_expiry.strftime('%Y-%m-%d'),
                    expiration_date_lte=max_expiry.strftime('%Y-%m-%d'),
                    status='active'
                )
            )
            
            if not contracts:
                return None
            
            # Convert to DataFrame
            options_data = []
            for contract in contracts:
                # Get latest quote for the option
                quote = await self._get_option_quote(contract.symbol)
                if quote:
                    options_data.append({
                        'symbol': contract.symbol,
                        'underlying': symbol,
                        'type': contract.type,  # 'call' or 'put'
                        'strike': float(contract.strike_price),
                        'expiration': contract.expiration_date,
                        'bid': quote['bid'],
                        'ask': quote['ask'],
                        'mid': (quote['bid'] + quote['ask']) / 2,
                        'spread': quote['ask'] - quote['bid'],
                        'volume': quote['volume'],
                        'open_interest': quote['open_interest'],
                        'implied_volatility': quote.get('implied_volatility', 0)
                    })
            
            if not options_data:
                return None
            
            df = pd.DataFrame(options_data)
            
            # Calculate Greeks for each option
            current_price = await self._get_current_price(symbol)
            df = self._calculate_greeks_for_chain(df, current_price)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {symbol}: {e}")
            return None
    
    async def _get_option_quote(self, option_symbol: str) -> Optional[Dict[str, float]]:
        """Get current quote for an option."""
        try:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None,
                lambda: self.api.get_latest_option_quote(option_symbol)
            )
            
            return {
                'bid': float(quote.bid_price) if quote.bid_price else 0,
                'ask': float(quote.ask_price) if quote.ask_price else 0,
                'volume': int(quote.bid_size + quote.ask_size) if quote.bid_size and quote.ask_size else 0,
                'open_interest': 0,  # Would need historical data
                'implied_volatility': self._estimate_implied_volatility(quote) if quote else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting option quote for {option_symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for underlying."""
        try:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None,
                lambda: self.api.get_latest_quote(symbol)
            )
            return float((quote.bid_price + quote.ask_price) / 2)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _calculate_greeks_for_chain(self, options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """Calculate Greeks for entire options chain."""
        options_df['days_to_expiry'] = (pd.to_datetime(options_df['expiration']) - datetime.now()).dt.days
        options_df['time_to_expiry'] = options_df['days_to_expiry'] / 365.0
        
        # Calculate Greeks for each option
        greeks_data = []
        for _, row in options_df.iterrows():
            greeks = self._calculate_greeks(
                spot_price=spot_price,
                strike=row['strike'],
                time_to_expiry=row['time_to_expiry'],
                volatility=row['implied_volatility'] or 0.20,  # Default 20% if missing
                risk_free_rate=self.get_parameter('risk_free_rate'),
                option_type=row['type']
            )
            greeks_data.append(greeks)
        
        # Add Greeks to DataFrame
        greeks_df = pd.DataFrame(greeks_data)
        return pd.concat([options_df, greeks_df], axis=1)
    
    def _calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: str
    ) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes model."""
        if time_to_expiry <= 0 or volatility <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Calculate d1 and d2
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) 
                    - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) 
                    + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  # Per 1% change in volatility
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2)) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _estimate_implied_volatility(self, quote: Any) -> float:
        """Estimate implied volatility from option quote."""
        # Simplified IV estimation - in production, use proper IV calculation
        # This is a placeholder that returns a reasonable estimate
        try:
            if hasattr(quote, 'implied_volatility'):
                return float(quote.implied_volatility)
            
            # Simple estimation based on bid-ask spread
            mid_price = (quote.bid_price + quote.ask_price) / 2
            spread_ratio = (quote.ask_price - quote.bid_price) / mid_price if mid_price > 0 else 0
            
            # Higher spread often indicates higher volatility
            base_iv = 0.20  # 20% base volatility
            iv_adjustment = spread_ratio * 0.5  # Up to 50% adjustment
            
            return base_iv + iv_adjustment
        except:
            return 0.20  # Default 20% volatility
    
    async def _update_iv_history(self, symbol: str):
        """Update implied volatility history for IV rank calculation."""
        try:
            if symbol not in self.iv_history:
                self.iv_history[symbol] = pd.DataFrame()
            
            # Get ATM options for IV calculation
            current_price = await self._get_current_price(symbol)
            if current_price <= 0:
                return
                
            options_chain = await self._get_options_chain(symbol)
            
            if options_chain is None or options_chain.empty:
                return
            
            # Find ATM options (closest to current price)
            atm_options = options_chain[
                (options_chain['strike'] >= current_price * 0.98) & 
                (options_chain['strike'] <= current_price * 1.02)
            ]
            
            if atm_options.empty:
                return
            
            # Calculate average IV for ATM options
            avg_iv = atm_options['implied_volatility'].mean()
            
            # Add to history
            new_row = pd.DataFrame({
                'date': [datetime.now()],
                'iv': [avg_iv],
                'symbol': [symbol]
            })
            
            self.iv_history[symbol] = pd.concat([self.iv_history[symbol], new_row], ignore_index=True)
            
            # Keep only last N days
            lookback_days = self.get_parameter('iv_percentile_lookback')
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            self.iv_history[symbol] = self.iv_history[symbol][
                pd.to_datetime(self.iv_history[symbol]['date']) > cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error updating IV history for {symbol}: {e}")
    
    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> float:
        """Calculate IV rank (percentile) for a symbol."""
        try:
            if symbol not in self.iv_history or self.iv_history[symbol].empty:
                return 50.0  # Default to middle if no history
            
            iv_values = self.iv_history[symbol]['iv'].values
            if len(iv_values) < 20:  # Need minimum history
                return 50.0
            
            # Calculate percentile rank
            rank = (iv_values < current_iv).sum() / len(iv_values) * 100
            return rank
            
        except Exception as e:
            self.logger.error(f"Error calculating IV rank for {symbol}: {e}")
            return 50.0
    
    async def _analyze_cash_secured_put_advanced(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_analysis: Dict[str, Any],
        ai_analysis: Optional[Dict[str, Any]],
        options_chain: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """Advanced analysis for cash-secured put with full options analysis."""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Filter put options
            puts = options_chain[options_chain['type'] == 'put'].copy()
            if puts.empty:
                return None
            
            # Apply filters
            puts = puts[
                (puts['open_interest'] >= self.get_parameter('min_open_interest')) &
                (puts['volume'] >= self.get_parameter('min_volume')) &
                (puts['spread'] <= self.get_parameter('min_bid_ask_spread')) &
                (puts['delta'] >= self.get_parameter('delta_target_put') - 0.05) &
                (puts['delta'] <= self.get_parameter('delta_target_put') + 0.05)
            ]
            
            if puts.empty:
                return None
            
            # Calculate IV rank for current ATM IV
            atm_iv = puts.loc[puts['strike'].sub(current_price).abs().idxmin(), 'implied_volatility']
            iv_rank = self._calculate_iv_rank(symbol, atm_iv)
            
            # Check IV rank threshold
            if iv_rank < self.get_parameter('iv_rank_min'):
                self.logger.debug(f"IV rank too low for {symbol}: {iv_rank:.1f}%")
                return None
            
            # Score each put option
            puts['score'] = self._score_put_options(puts, current_price, technical_analysis, iv_rank)
            
            # Select best put
            best_put = puts.loc[puts['score'].idxmax()]
            
            # Calculate confidence
            confidence = self._calculate_put_confidence(
                best_put, technical_analysis, ai_analysis, iv_rank
            )
            
            if confidence < self.get_parameter('min_confidence'):
                return None
            
            # Create signal with full options data
            metadata = {
                'strategy': self.name,
                'option_type': 'PUT',
                'option_action': 'SELL',
                'option_symbol': best_put['symbol'],
                'strike': best_put['strike'],
                'expiration': best_put['expiration'],
                'days_to_expiry': best_put['days_to_expiry'],
                'premium': best_put['bid'],  # We receive the bid when selling
                'delta': best_put['delta'],
                'gamma': best_put['gamma'],
                'theta': best_put['theta'],
                'vega': best_put['vega'],
                'implied_volatility': best_put['implied_volatility'],
                'iv_rank': iv_rank,
                'open_interest': best_put['open_interest'],
                'volume': best_put['volume'],
                'wheel_stage': 'CSP',
                'execute_options_order': True  # Flag for actual execution
            }
            
            reasoning = (
                f"Sell CSP: Strike ${best_put['strike']:.2f}, "
                f"Premium ${best_put['bid']:.2f}, "
                f"Delta {best_put['delta']:.2f}, "
                f"IV Rank {iv_rank:.1f}%, "
                f"DTE {best_put['days_to_expiry']}"
            )
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=confidence,
                price=current_price,
                reasoning=reasoning,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in advanced CSP analysis for {symbol}: {e}")
            return None
    
    async def _analyze_covered_call_advanced(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_analysis: Dict[str, Any],
        ai_analysis: Optional[Dict[str, Any]],
        options_chain: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """Advanced analysis for covered call with full options analysis."""
        try:
            current_price = market_data['close'].iloc[-1]
            wheel_data = self.active_wheels.get(symbol, {})
            cost_basis = wheel_data.get('cost_basis', current_price)
            
            # Filter call options
            calls = options_chain[options_chain['type'] == 'call'].copy()
            if calls.empty:
                return None
            
            # Apply filters - calls above cost basis
            calls = calls[
                (calls['strike'] >= cost_basis * 1.01) &  # At least 1% above cost basis
                (calls['open_interest'] >= self.get_parameter('min_open_interest')) &
                (calls['volume'] >= self.get_parameter('min_volume')) &
                (calls['spread'] <= self.get_parameter('min_bid_ask_spread')) &
                (calls['delta'] >= self.get_parameter('delta_target_call') - 0.05) &
                (calls['delta'] <= self.get_parameter('delta_target_call') + 0.05)
            ]
            
            if calls.empty:
                return None
            
            # Calculate IV rank
            atm_iv = calls.loc[calls['strike'].sub(current_price).abs().idxmin(), 'implied_volatility']
            iv_rank = self._calculate_iv_rank(symbol, atm_iv)
            
            # Score each call option
            calls['score'] = self._score_call_options(
                calls, current_price, cost_basis, technical_analysis, iv_rank
            )
            
            # Select best call
            best_call = calls.loc[calls['score'].idxmax()]
            
            # Calculate confidence
            confidence = self._calculate_call_confidence(
                best_call, technical_analysis, ai_analysis, iv_rank, cost_basis, current_price
            )
            
            if confidence < self.get_parameter('min_confidence'):
                return None
            
            # Calculate profit if called
            profit_if_called = ((best_call['strike'] - cost_basis) / cost_basis) * 100
            
            # Create signal with full options data
            metadata = {
                'strategy': self.name,
                'option_type': 'CALL',
                'option_action': 'SELL',
                'option_symbol': best_call['symbol'],
                'strike': best_call['strike'],
                'expiration': best_call['expiration'],
                'days_to_expiry': best_call['days_to_expiry'],
                'premium': best_call['bid'],
                'delta': best_call['delta'],
                'gamma': best_call['gamma'],
                'theta': best_call['theta'],
                'vega': best_call['vega'],
                'implied_volatility': best_call['implied_volatility'],
                'iv_rank': iv_rank,
                'open_interest': best_call['open_interest'],
                'volume': best_call['volume'],
                'wheel_stage': 'CC',
                'cost_basis': cost_basis,
                'profit_if_called': profit_if_called,
                'execute_options_order': True
            }
            
            reasoning = (
                f"Sell CC: Strike ${best_call['strike']:.2f}, "
                f"Premium ${best_call['bid']:.2f}, "
                f"Delta {best_call['delta']:.2f}, "
                f"IV Rank {iv_rank:.1f}%, "
                f"Profit if called {profit_if_called:.1f}%"
            )
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=confidence,
                price=current_price,
                reasoning=reasoning,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in advanced CC analysis for {symbol}: {e}")
            return None
    
    def _score_put_options(
        self,
        puts: pd.DataFrame,
        current_price: float,
        technical_analysis: Dict[str, Any],
        iv_rank: float
    ) -> pd.Series:
        """Score put options for selection."""
        scores = pd.Series(index=puts.index, dtype=float)
        
        # Premium yield score (higher is better)
        premium_yield = (puts['bid'] / puts['strike']) * 100
        scores += premium_yield * 10
        
        # Delta score (closer to target is better)
        delta_target = abs(self.get_parameter('delta_target_put'))
        delta_diff = abs(abs(puts['delta']) - delta_target)
        scores -= delta_diff * 50
        
        # Theta score (higher theta is better for sellers)
        scores += abs(puts['theta']) * 100
        
        # Liquidity score
        liquidity_score = np.log1p(puts['open_interest']) + np.log1p(puts['volume'])
        scores += liquidity_score * 5
        
        # IV rank bonus
        if iv_rank > 50:
            scores += (iv_rank - 50) * 0.5
        
        # Technical score
        rsi = technical_analysis.get('indicators', {}).get('rsi', {}).get('value', 50)
        if rsi < self.get_parameter('rsi_oversold'):
            scores += (self.get_parameter('rsi_oversold') - rsi) * 0.5
        
        return scores
    
    def _score_call_options(
        self,
        calls: pd.DataFrame,
        current_price: float,
        cost_basis: float,
        technical_analysis: Dict[str, Any],
        iv_rank: float
    ) -> pd.Series:
        """Score call options for selection."""
        scores = pd.Series(index=calls.index, dtype=float)
        
        # Premium yield score
        premium_yield = (calls['bid'] / current_price) * 100
        scores += premium_yield * 10
        
        # Profit if called score
        profit_if_called = ((calls['strike'] - cost_basis) / cost_basis) * 100
        scores += profit_if_called * 5
        
        # Delta score
        delta_target = self.get_parameter('delta_target_call')
        delta_diff = abs(calls['delta'] - delta_target)
        scores -= delta_diff * 50
        
        # Theta score
        scores += abs(calls['theta']) * 100
        
        # Liquidity score
        liquidity_score = np.log1p(calls['open_interest']) + np.log1p(calls['volume'])
        scores += liquidity_score * 5
        
        # IV rank bonus
        if iv_rank > 50:
            scores += (iv_rank - 50) * 0.5
        
        # Technical score
        rsi = technical_analysis.get('indicators', {}).get('rsi', {}).get('value', 50)
        if rsi > self.get_parameter('rsi_overbought'):
            scores += (rsi - self.get_parameter('rsi_overbought')) * 0.5
        
        return scores
    
    def _calculate_put_confidence(
        self,
        put_option: pd.Series,
        technical_analysis: Dict[str, Any],
        ai_analysis: Optional[Dict[str, Any]],
        iv_rank: float
    ) -> float:
        """Calculate confidence for put selling."""
        confidence = 50.0
        
        # IV rank component (0-20 points)
        if iv_rank >= self.get_parameter('iv_rank_min'):
            confidence += min(20, (iv_rank - self.get_parameter('iv_rank_min')) * 0.4)
        
        # Premium yield component (0-15 points)
        premium_yield = (put_option['bid'] / put_option['strike']) * 100
        if premium_yield >= self.get_parameter('min_premium_percentage'):
            confidence += min(15, premium_yield * 5)
        
        # Greeks component (0-15 points)
        # Good theta (time decay in our favor)
        if put_option['theta'] < -0.01:  # Negative theta benefits option sellers
            confidence += min(10, abs(put_option['theta']) * 100)
        
        # Reasonable delta
        if abs(put_option['delta'] - self.get_parameter('delta_target_put')) < 0.05:
            confidence += 5
        
        # Technical analysis (0-10 points)
        rsi = technical_analysis.get('indicators', {}).get('rsi', {}).get('value', 50)
        if rsi <= self.get_parameter('rsi_oversold'):
            confidence += min(10, (self.get_parameter('rsi_oversold') - rsi) * 0.5)
        
        # AI analysis boost (0-10 points)
        if ai_analysis and ai_analysis.get('signal') == 'BUY':
            ai_confidence = ai_analysis.get('confidence', 50)
            confidence += (ai_confidence - 50) * 0.2
        
        return min(95, confidence)
    
    def _calculate_call_confidence(
        self,
        call_option: pd.Series,
        technical_analysis: Dict[str, Any],
        ai_analysis: Optional[Dict[str, Any]],
        iv_rank: float,
        cost_basis: float,
        current_price: float
    ) -> float:
        """Calculate confidence for call selling."""
        confidence = 50.0
        
        # IV rank component
        if iv_rank >= self.get_parameter('iv_rank_min'):
            confidence += min(20, (iv_rank - self.get_parameter('iv_rank_min')) * 0.4)
        
        # Premium yield component
        premium_yield = (call_option['bid'] / current_price) * 100
        if premium_yield >= self.get_parameter('min_premium_percentage'):
            confidence += min(15, premium_yield * 5)
        
        # Profit component (0-10 points)
        current_profit = ((current_price - cost_basis) / cost_basis) * 100
        if current_profit > 0:
            confidence += min(10, current_profit)
        
        # Greeks component
        if call_option['theta'] < -0.01:
            confidence += min(10, abs(call_option['theta']) * 100)
        
        if abs(call_option['delta'] - self.get_parameter('delta_target_call')) < 0.05:
            confidence += 5
        
        # Technical analysis
        rsi = technical_analysis.get('indicators', {}).get('rsi', {}).get('value', 50)
        if rsi >= self.get_parameter('rsi_overbought'):
            confidence += min(10, (rsi - self.get_parameter('rsi_overbought')) * 0.5)
        
        # AI analysis
        if ai_analysis and ai_analysis.get('signal') == 'SELL':
            ai_confidence = ai_analysis.get('confidence', 50)
            confidence += (ai_confidence - 50) * 0.2
        
        return min(95, confidence)
    
    async def execute_options_order(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute actual options order through Alpaca API."""
        try:
            if not self.api:
                self.logger.error("Alpaca API not initialized")
                return None
            
            metadata = signal.metadata
            
            # Prepare order parameters
            order_params = {
                'symbol': metadata['option_symbol'],
                'qty': 1,  # Number of contracts
                'side': 'sell',  # Always selling for wheel strategy
                'type': 'limit',
                'time_in_force': 'day',
                'limit_price': metadata['premium']
            }
            
            # Submit order
            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self.api.submit_order(**order_params)
            )
            
            self.logger.info(
                f"Options order submitted: {metadata['option_action']} "
                f"{metadata['option_type']} {metadata['option_symbol']} "
                f"at ${metadata['premium']:.2f}"
            )
            
            return {
                'order_id': order.id,
                'symbol': metadata['option_symbol'],
                'underlying': signal.symbol,
                'type': metadata['option_type'],
                'action': metadata['option_action'],
                'strike': metadata['strike'],
                'expiration': metadata['expiration'],
                'premium': metadata['premium'],
                'status': order.status
            }
            
        except Exception as e:
            self.logger.error(f"Error executing options order: {e}")
            return None
    
    def _filter_eligible_symbols(
        self,
        symbols: List[str],
        technical_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Filter symbols eligible for wheel strategy."""
        eligible = []
        whitelist = self.get_parameter('etf_whitelist')
        
        for symbol in symbols:
            # Check if in whitelist
            if symbol not in whitelist:
                continue
            
            # Check technical criteria if enabled
            if self.get_parameter('use_technical_filters'):
                tech_data = technical_analysis.get(symbol, {})
                if self._passes_technical_filters(tech_data):
                    eligible.append(symbol)
            else:
                eligible.append(symbol)
        
        return eligible
    
    def _passes_technical_filters(self, tech_data: Dict[str, Any]) -> bool:
        """Check if symbol passes technical filters."""
        if tech_data.get('error') or 'indicators' not in tech_data:
            return False
        
        indicators = tech_data['indicators']
        
        # Check ATR (volatility)
        atr = self._calculate_atr_from_data(tech_data)
        min_atr = self.get_parameter('min_atr')
        max_atr = self.get_parameter('max_atr')
        
        if not (min_atr <= atr <= max_atr):
            return False
        
        # Additional filters can be added here
        return True
    
    def _calculate_atr_from_data(self, tech_data: Dict[str, Any]) -> float:
        """Calculate ATR from technical data."""
        # Simplified ATR calculation using trend data
        trend_data = tech_data.get('indicators', {}).get('trend', {})
        
        # Use slope and ROC as proxy for volatility
        slope = abs(trend_data.get('slope', 0))
        roc = abs(trend_data.get('roc', 0))
        
        # Approximate ATR (this is simplified, real implementation would use actual ATR)
        estimated_atr = (slope * 100 + roc * 10) / 2
        
        return max(0.1, estimated_atr)  # Minimum 0.1 to avoid division by zero
    
    def update_wheel_position(self, symbol: str, action: str, details: Dict[str, Any]) -> None:
        """Update wheel position tracking."""
        if action == 'ASSIGNED_PUT':
            # Stock was assigned from put
            self.active_wheels[symbol] = {
                'cost_basis': details.get('price', 0),
                'quantity': details.get('quantity', 100),
                'entry_date': datetime.now(),
                'stage': 'STOCK_OWNED'
            }
            self.logger.info(f"Put assigned for {symbol}, now own stock at ${details.get('price', 0):.2f}")
        
        elif action == 'CALLED_AWAY':
            # Stock was called away
            if symbol in self.active_wheels:
                profit = details.get('profit', 0)
                self.logger.info(f"Stock called away for {symbol}, profit: ${profit:.2f}")
                del self.active_wheels[symbol]
        
        elif action == 'MANUAL_CLOSE':
            # Manual position close
            if symbol in self.active_wheels:
                del self.active_wheels[symbol]
    
    def get_active_wheels(self) -> Dict[str, Dict[str, Any]]:
        """Get current active wheel positions."""
        return self.active_wheels.copy()
