import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from groq import AsyncGroq
from config.settings import GroqConfig
from utils.logger import setup_logger
from utils.helpers import retry_async

class GroqAnalyzer:
    """Groq AI integration for trading signal analysis."""
    
    def __init__(self, config: GroqConfig):
        self.config = config
        self.logger = setup_logger(None, "GroqAnalyzer")
        self._signal_cache: Dict[str, Dict] = {}
        
        # Initialize client with error handling
        try:
            if not config.api_key:
                self.logger.warning("No Groq API key provided - AI analysis will be disabled")
                self.client = None
            else:
                self.client = AsyncGroq(api_key=config.api_key)
                self.logger.info("Groq AI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    @retry_async(max_retries=3, delay=2.0)
    async def analyze_technical_data(
        self, 
        symbol: str, 
        technical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze technical data using Groq AI."""
        if not self.client:
            self.logger.warning("Groq client not available - returning default response")
            return self._default_analysis_response(symbol)
            
        try:
            prompt = self._create_analysis_prompt(symbol, technical_analysis)
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantitative trading analyst. Analyze the provided technical indicators and provide a JSON response with trading signals, confidence scores, and reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Validate and normalize the response
            return self._normalize_analysis_response(symbol, analysis)
        
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return self._default_analysis_response(symbol)
    
    async def batch_analyze(
        self, 
        symbols_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch analyze multiple symbols."""
        tasks = []
        for symbol, data in symbols_data.items():
            if 'indicators' in data and not data.get('error'):
                task = self.analyze_technical_data(symbol, data)
                tasks.append((symbol, task))
        
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks], 
                return_exceptions=True
            )
            
            for (symbol, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    self.logger.error(f"Error analyzing {symbol}: {result}")
                    results[symbol] = self._default_analysis_response(symbol)
                else:
                    results[symbol] = result
        
        return results
    
    def _create_analysis_prompt(self, symbol: str, data: Dict[str, Any]) -> str:
        """Create analysis prompt for Groq AI."""
        indicators = data.get('indicators', {})
        
        prompt = f"""
Analyze the following technical indicators for {symbol}:

Current Price: ${data.get('current_price', 0):.2f}

RSI: {indicators.get('rsi', {}).get('value', 50):.2f} ({indicators.get('rsi', {}).get('signal', 'NEUTRAL')})
Bollinger Bands: 
- Upper: ${indicators.get('bollinger', {}).get('upper', 0):.2f}
- Lower: ${indicators.get('bollinger', {}).get('lower', 0):.2f}
- Position: {indicators.get('bollinger', {}).get('percent', 0.5):.2%} ({indicators.get('bollinger', {}).get('signal', 'NEUTRAL')})

MACD:
- MACD: {indicators.get('macd', {}).get('macd', 0):.4f}
- Signal: {indicators.get('macd', {}).get('signal_line', 0):.4f}
- Histogram: {indicators.get('macd', {}).get('histogram', 0):.4f} ({indicators.get('macd', {}).get('signal', 'NEUTRAL')})

Volume: {indicators.get('volume', {}).get('ratio', 1.0):.2f}x average ({indicators.get('volume', {}).get('signal', 'NORMAL')})

Support/Resistance:
- Support: ${indicators.get('support_resistance', {}).get('support', 0):.2f}
- Resistance: ${indicators.get('support_resistance', {}).get('resistance', 0):.2f}

Trend Analysis:
- ADX: {indicators.get('trend', {}).get('adx', 0):.2f}
- Slope: {indicators.get('trend', {}).get('slope', 0):.4f}
- ROC: {indicators.get('trend', {}).get('roc', 0):.2%}

Provide analysis in this JSON format:
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0-100,
    "reasoning": "Brief explanation of the decision",
    "risk_level": "LOW|MEDIUM|HIGH",
    "target_price": "Optional target price",
    "stop_loss": "Optional stop loss price",
    "time_horizon": "SHORT|MEDIUM|LONG",
    "key_factors": ["List of 2-3 key factors influencing the decision"]
}}
"""
        return prompt
    
    def _normalize_analysis_response(
        self, 
        symbol: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize and validate AI response."""
        try:
            signal = analysis.get('signal', 'HOLD').upper()
            if signal not in ['BUY', 'SELL', 'HOLD']:
                signal = 'HOLD'
            
            confidence = float(analysis.get('confidence', 50))
            confidence = max(0, min(100, confidence))  # Clamp between 0-100
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                'risk_level': analysis.get('risk_level', 'MEDIUM'),
                'target_price': analysis.get('target_price'),
                'stop_loss': analysis.get('stop_loss'),
                'time_horizon': analysis.get('time_horizon', 'MEDIUM'),
                'key_factors': analysis.get('key_factors', []),
                'timestamp': datetime.now(),
                'ai_provider': 'groq'
            }
        
        except Exception as e:
            self.logger.error(f"Error normalizing response for {symbol}: {e}")
            return self._default_analysis_response(symbol)
    
    def _default_analysis_response(self, symbol: str) -> Dict[str, Any]:
        """Default response when AI analysis fails."""
        return {
            'symbol': symbol,
            'signal': 'HOLD',
            'confidence': 50.0,
            'reasoning': 'AI analysis unavailable',
            'risk_level': 'HIGH',
            'target_price': None,
            'stop_loss': None,
            'time_horizon': 'MEDIUM',
            'key_factors': ['AI analysis failed'],
            'timestamp': datetime.now(),
            'ai_provider': 'groq',
            'error': True
        }
