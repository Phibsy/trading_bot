import pandas as pd
from typing import Dict, List, Any, Optional
from strategies.base_strategy import BaseStrategy, TradingSignal

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'min_confidence': 60,
            'volume_threshold': 1.2
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("RSI Strategy", default_params)
    
    async def analyze(
        self,
        market_data: Dict[str, pd.DataFrame],
        technical_analysis: Dict[str, Dict[str, Any]],
        ai_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[TradingSignal]:
        """Analyze using RSI strategy."""
        # Import here to avoid circular imports
        from config.constants import SignalType
        
        signals = []
        
        oversold = self.get_parameter('oversold_threshold')
        overbought = self.get_parameter('overbought_threshold')
        min_confidence = self.get_parameter('min_confidence')
        volume_threshold = self.get_parameter('volume_threshold')
        
        for symbol, analysis in technical_analysis.items():
            if analysis.get('error') or 'indicators' not in analysis:
                continue
            
            indicators = analysis['indicators']
            rsi_data = indicators.get('rsi', {})
            volume_data = indicators.get('volume', {})
            
            rsi_value = rsi_data.get('value', 50)
            volume_ratio = volume_data.get('ratio', 1.0)
            current_price = analysis.get('current_price', 0)
            
            if current_price <= 0:
                continue
            
            signal_type = None
            confidence = 0
            reasoning = ""
            
            # RSI Oversold Signal
            if rsi_value <= oversold:
                signal_type = SignalType.BUY
                confidence = min(95, 60 + (oversold - rsi_value) * 2)
                reasoning = f"RSI oversold at {rsi_value:.1f} (threshold: {oversold})"
                
                # Boost confidence with volume confirmation
                if volume_ratio >= volume_threshold:
                    confidence += 10
                    reasoning += f", high volume confirmation ({volume_ratio:.1f}x)"
            
            # RSI Overbought Signal
            elif rsi_value >= overbought:
                signal_type = SignalType.SELL
                confidence = min(95, 60 + (rsi_value - overbought) * 2)
                reasoning = f"RSI overbought at {rsi_value:.1f} (threshold: {overbought})"
                
                # Boost confidence with volume confirmation
                if volume_ratio >= volume_threshold:
                    confidence += 10
                    reasoning += f", high volume confirmation ({volume_ratio:.1f}x)"
            
            # Apply AI analysis boost if available
            if ai_analysis and symbol in ai_analysis:
                ai_signal = ai_analysis[symbol]
                ai_signal_type = ai_signal.get('signal', 'HOLD')
                ai_confidence = ai_signal.get('confidence', 50)
                
                # If AI agrees with RSI signal, boost confidence
                if (signal_type == SignalType.BUY and ai_signal_type == 'BUY') or \
                   (signal_type == SignalType.SELL and ai_signal_type == 'SELL'):
                    confidence = min(98, confidence + (ai_confidence * 0.2))
                    reasoning += f", AI confirmation ({ai_confidence:.0f}%)"
                
                # If AI disagrees significantly, reduce confidence
                elif (signal_type == SignalType.BUY and ai_signal_type == 'SELL') or \
                     (signal_type == SignalType.SELL and ai_signal_type == 'BUY'):
                    confidence *= 0.7
                    reasoning += f", AI disagreement"
            
            # Only create signal if confidence meets minimum threshold
            if signal_type and confidence >= min_confidence:
                metadata = {
                    'rsi_value': rsi_value,
                    'volume_ratio': volume_ratio,
                    'strategy': self.name,
                    'thresholds': {
                        'oversold': oversold,
                        'overbought': overbought
                    }
                }
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    reasoning=reasoning,
                    metadata=metadata
                )
                
                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signals[symbol] = signal
        
        return signals
