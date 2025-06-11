import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from strategies.base_strategy import BaseStrategy, TradingSignal

class MLStrategy(BaseStrategy):
    """Machine Learning enhanced trading strategy."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 50,
            'feature_weights': {
                'rsi': 0.25,
                'macd': 0.20,
                'bollinger': 0.20,
                'volume': 0.15,
                'trend': 0.20
            },
            'ai_weight': 0.4,
            'min_confidence': 70,
            'trend_confirmation_required': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("ML Strategy", default_params)
    
    async def analyze(
        self,
        market_data: Dict[str, pd.DataFrame],
        technical_analysis: Dict[str, Dict[str, Any]],
        ai_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[TradingSignal]:
        """Analyze using ML-enhanced strategy."""
        # Import here to avoid circular imports
        from config.constants import SignalType
        
        signals = []
        
        if not ai_analysis:
            return signals  # ML strategy requires AI analysis
        
        feature_weights = self.get_parameter('feature_weights')
        ai_weight = self.get_parameter('ai_weight')
        min_confidence = self.get_parameter('min_confidence')
        trend_confirmation = self.get_parameter('trend_confirmation_required')
        
        for symbol in technical_analysis.keys():
            if symbol not in ai_analysis:
                continue
            
            tech_analysis = technical_analysis[symbol]
            ai_result = ai_analysis[symbol]
            
            if tech_analysis.get('error') or ai_result.get('error'):
                continue
            
            # Calculate composite signal
            signal_result = self._calculate_composite_signal(
                tech_analysis, ai_result, feature_weights, ai_weight
            )
            
            if not signal_result:
                continue
            
            signal_type, confidence, reasoning = signal_result
            current_price = tech_analysis.get('current_price', 0)
            
            # Apply trend confirmation filter
            if trend_confirmation and not self._confirm_trend(tech_analysis, signal_type):
                confidence *= 0.7  # Reduce confidence if trend doesn't confirm
                reasoning += " (weak trend confirmation)"
            
            # Only create signal if confidence meets threshold
            if confidence >= min_confidence:
                metadata = {
                    'strategy': self.name,
                    'ai_confidence': ai_result.get('confidence', 0),
                    'technical_score': self._calculate_technical_score(tech_analysis, feature_weights),
                    'composite_method': 'weighted_average',
                    'feature_weights': feature_weights
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
    
    def _calculate_composite_signal(
        self,
        tech_analysis: Dict[str, Any],
        ai_result: Dict[str, Any],
        feature_weights: Dict[str, float],
        ai_weight: float
    ) -> Optional[tuple]:
        """Calculate composite signal from technical and AI analysis."""
        try:
            # Import here to avoid circular imports
            from config.constants import SignalType
            
            # Get technical score
            tech_score = self._calculate_technical_score(tech_analysis, feature_weights)
            
            # Get AI signal and confidence
            ai_signal = ai_result.get('signal', 'HOLD')
            ai_confidence = ai_result.get('confidence', 50) / 100.0  # Normalize to 0-1
            
            # Convert AI signal to score (-1 to 1)
            ai_score_map = {'SELL': -1.0, 'HOLD': 0.0, 'BUY': 1.0}
            ai_score = ai_score_map.get(ai_signal, 0.0) * ai_confidence
            
            # Calculate weighted composite score
            technical_weight = 1.0 - ai_weight
            composite_score = (tech_score * technical_weight) + (ai_score * ai_weight)
            
            # Determine signal type and confidence
            if composite_score > 0.3:
                signal_type = SignalType.BUY
                confidence = min(95, 50 + (composite_score * 45))
            elif composite_score < -0.3:
                signal_type = SignalType.SELL
                confidence = min(95, 50 + (abs(composite_score) * 45))
            else:
                return None  # No strong signal
            
            # Create reasoning
            reasoning = f"ML composite: tech={tech_score:.2f}, ai={ai_score:.2f}, final={composite_score:.2f}"
            if ai_result.get('reasoning'):
                reasoning += f" | AI: {ai_result['reasoning'][:50]}..."
            
            return signal_type, confidence, reasoning
        
        except Exception as e:
            # Log error but don't crash
            return None
    
    def _calculate_technical_score(
        self,
        analysis: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """Calculate technical analysis score (-1 to 1)."""
        indicators = analysis.get('indicators', {})
        score = 0.0
        total_weight = 0.0
        
        # RSI Score
        rsi_data = indicators.get('rsi', {})
        if rsi_data:
            rsi_value = rsi_data.get('value', 50)
            if rsi_value <= 30:
                rsi_score = 1.0  # Strong buy
            elif rsi_value >= 70:
                rsi_score = -1.0  # Strong sell
            else:
                rsi_score = (50 - rsi_value) / 20  # Normalize between -1 and 1
            
            score += rsi_score * weights.get('rsi', 0)
            total_weight += weights.get('rsi', 0)
        
        # MACD Score
        macd_data = indicators.get('macd', {})
        if macd_data:
            histogram = macd_data.get('histogram', 0)
            macd_score = np.tanh(histogram * 1000)  # Normalize using tanh
            
            score += macd_score * weights.get('macd', 0)
            total_weight += weights.get('macd', 0)
        
        # Bollinger Bands Score
        bb_data = indicators.get('bollinger', {})
        if bb_data:
            bb_percent = bb_data.get('percent', 0.5)
            if bb_percent <= 0.1:
                bb_score = 1.0  # Oversold
            elif bb_percent >= 0.9:
                bb_score = -1.0  # Overbought
            else:
                bb_score = (0.5 - bb_percent) * 2  # Normalize
            
            score += bb_score * weights.get('bollinger', 0)
            total_weight += weights.get('bollinger', 0)
        
        # Volume Score
        volume_data = indicators.get('volume', {})
        if volume_data:
            volume_ratio = volume_data.get('ratio', 1.0)
            volume_score = min(1.0, (volume_ratio - 1.0) * 0.5)  # Higher volume = positive
            
            score += volume_score * weights.get('volume', 0)
            total_weight += weights.get('volume', 0)
        
        # Trend Score
        trend_data = indicators.get('trend', {})
        if trend_data:
            slope = trend_data.get('slope', 0)
            adx = trend_data.get('adx', 0)
            
            # Combine slope direction with trend strength
            trend_score = np.tanh(slope * 100) * min(1.0, adx / 25)
            
            score += trend_score * weights.get('trend', 0)
            total_weight += weights.get('trend', 0)
        
        # Normalize by total weight
        return score / total_weight if total_weight > 0 else 0.0
    
    def _confirm_trend(self, analysis: Dict[str, Any], signal_type: Any) -> bool:
        """Confirm if the signal aligns with the current trend."""
        from config.constants import SignalType
        
        indicators = analysis.get('indicators', {})
        trend_data = indicators.get('trend', {})
        
        slope = trend_data.get('slope', 0)
        adx = trend_data.get('adx', 0)
        
        # Consider trend strong if ADX > 25
        strong_trend = adx > 25
        
        if not strong_trend:
            return True  # Don't penalize in weak trend conditions
        
        # Check if signal aligns with trend direction
        if signal_type == SignalType.BUY and slope > 0:
            return True
        elif signal_type == SignalType.SELL and slope < 0:
            return True
        
        return False
