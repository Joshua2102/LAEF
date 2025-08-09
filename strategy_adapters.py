# === strategy_adapters.py - Adapters for existing strategies to comply with BaseStrategy interface ===

import logging
from datetime import datetime
from typing import Dict, Tuple, Any
from core.base_strategy import BaseStrategy
from core.fifo_portfolio import FIFOPortfolio

# Import existing engines
from core.dual_model_trading_logic import DualModelTradingEngine
from core.hybrid_trading_engine import HybridTradingEngine
from core.momentum_scalping_engine import MomentumScalpingEngine
from core.ai_momentum_scalping_engine import AIMomentumScalpingEngine

class DualModelStrategyAdapter(BaseStrategy):
    """Adapter for DualModelTradingEngine to comply with BaseStrategy interface"""
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict[str, Any]):
        super().__init__(portfolio, config)
        self.engine = DualModelTradingEngine(portfolio, custom_config=config)
        self.last_trade_time = {}
    
    def should_trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if conditions are right for trading"""
        
        # Check timing
        timing_valid, timing_reason = self.validate_trade_timing(
            symbol, 
            market_data.get('timestamp', datetime.now())
        )
        if not timing_valid:
            return False, timing_reason
        
        # Check basic requirements
        q_value = market_data.get('q_value', 0)
        ml_confidence = market_data.get('ml_confidence', 0)
        
        if q_value < 0.1 and ml_confidence < 0.1:
            return False, "Q-value and ML confidence too low"
        
        # Check if we have necessary data
        if 'indicators' not in market_data:
            return False, "Missing technical indicators"
        
        return True, "Trading conditions met"
    
    def trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Execute trading decision"""
        
        # Extract required parameters
        q_value = market_data.get('q_value', 0.5)
        ml_confidence = market_data.get('ml_confidence', 0.5)
        indicators = market_data.get('indicators', {})
        current_price = market_data.get('current_price')
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Call original engine
        action, confidence, reason, action_data = self.engine.evaluate_trade_decision(
            symbol=symbol,
            q_value=q_value,
            ml_confidence=ml_confidence,
            indicators=indicators,
            current_price=current_price,
            timestamp=timestamp
        )
        
        # Update last trade time if action taken
        if action in ['buy', 'sell']:
            self.last_trade_time[symbol] = timestamp
        
        return action, confidence, reason, action_data
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Dual Model Trading (AI Momentum Scalping)',
            'version': '2.0',
            'description': 'AI-powered momentum scalping with Q-Learning and ML',
            'engine_type': 'ai_momentum_scalping',
            'parameters': self.config
        }

class HybridTradingStrategyAdapter(BaseStrategy):
    """Adapter for HybridTradingEngine to comply with BaseStrategy interface"""
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict[str, Any]):
        super().__init__(portfolio, config)
        self.engine = HybridTradingEngine(portfolio, custom_config=config)
        self.last_trade_time = {}
    
    def should_trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if conditions are right for trading"""
        
        # Check timing
        timing_valid, timing_reason = self.validate_trade_timing(
            symbol, 
            market_data.get('timestamp', datetime.now())
        )
        if not timing_valid:
            return False, timing_reason
        
        # Hybrid engine needs Q-value and ML confidence
        if 'q_value' not in market_data or 'ml_confidence' not in market_data:
            return False, "Missing AI signals for hybrid strategy"
        
        # Check market hours (hybrid does day trading)
        current_hour = market_data.get('timestamp', datetime.now()).hour
        if current_hour < 9 or current_hour > 16:
            return False, "Outside market hours for day trading"
        
        return True, "Hybrid trading conditions met"
    
    def trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Execute trading decision"""
        
        # Extract required parameters
        q_value = market_data.get('q_value', 0.5)
        ml_confidence = market_data.get('ml_confidence', 0.5)
        indicators = market_data.get('indicators', {})
        current_price = market_data.get('current_price')
        current_time = market_data.get('timestamp', datetime.now())
        force_sell = market_data.get('force_sell', False)
        
        # Call original engine
        action, confidence, reason, action_data = self.engine.evaluate_trade_decision(
            symbol=symbol,
            q_value=q_value,
            ml_confidence=ml_confidence,
            indicators=indicators,
            current_price=current_price,
            current_time=current_time,
            force_sell=force_sell
        )
        
        # Update last trade time if action taken
        if action in ['buy', 'sell']:
            self.last_trade_time[symbol] = current_time
        
        return action, confidence, reason, action_data
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Hybrid Trading Engine',
            'version': '1.0',
            'description': 'Day trading + Swing trading for high volatility',
            'engine_type': 'hybrid',
            'parameters': self.config
        }

class MomentumScalpingStrategyAdapter(BaseStrategy):
    """Adapter for MomentumScalpingEngine to comply with BaseStrategy interface"""
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict[str, Any]):
        super().__init__(portfolio, config)
        self.engine = MomentumScalpingEngine(portfolio, custom_config=config)
        self.last_trade_time = {}
    
    def should_trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if conditions are right for trading"""
        
        # Check timing
        timing_valid, timing_reason = self.validate_trade_timing(
            symbol, 
            market_data.get('timestamp', datetime.now())
        )
        if not timing_valid:
            return False, timing_reason
        
        # Momentum scalping needs price and indicators
        if 'current_price' not in market_data:
            return False, "Missing current price"
        
        if 'indicators' not in market_data:
            return False, "Missing technical indicators"
        
        # Check if we have momentum data
        indicators = market_data.get('indicators', {})
        if 'momentum' not in indicators and 'roc' not in indicators:
            return False, "Missing momentum indicators"
        
        return True, "Momentum scalping conditions met"
    
    def trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Execute trading decision"""
        
        # Extract required parameters
        current_price = market_data.get('current_price')
        indicators = market_data.get('indicators', {})
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Call original engine
        action, confidence, reason, action_data = self.engine.evaluate_trade_decision(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators,
            timestamp=timestamp
        )
        
        # Update last trade time if action taken
        if action in ['buy', 'sell']:
            self.last_trade_time[symbol] = timestamp
        
        return action, confidence, reason, action_data
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'Momentum Scalping Engine',
            'version': '1.0',
            'description': 'Ultra-fast momentum-based micro-scalping',
            'engine_type': 'momentum_scalping',
            'parameters': self.config
        }

class AIMomentumScalpingStrategyAdapter(BaseStrategy):
    """Adapter for AIMomentumScalpingEngine to comply with BaseStrategy interface"""
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict[str, Any]):
        super().__init__(portfolio, config)
        self.engine = AIMomentumScalpingEngine(portfolio, custom_config=config)
        self.last_trade_time = {}
    
    def should_trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if conditions are right for trading"""
        
        # Check timing
        timing_valid, timing_reason = self.validate_trade_timing(
            symbol, 
            market_data.get('timestamp', datetime.now())
        )
        if not timing_valid:
            return False, timing_reason
        
        # AI momentum needs all signals
        required_fields = ['q_value', 'ml_confidence', 'current_price', 'indicators']
        missing = [field for field in required_fields if field not in market_data]
        
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        
        # Check if AI signals are meaningful
        q_value = market_data.get('q_value', 0)
        ml_confidence = market_data.get('ml_confidence', 0)
        
        if q_value < 0.1 and ml_confidence < 0.1:
            return False, "AI signals too weak"
        
        return True, "AI momentum scalping conditions met"
    
    def trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Execute trading decision"""
        
        # Extract required parameters
        q_value = market_data.get('q_value', 0.5)
        ml_confidence = market_data.get('ml_confidence', 0.5)
        indicators = market_data.get('indicators', {})
        current_price = market_data.get('current_price')
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Call original engine
        action, confidence, reason, action_data = self.engine.evaluate_trade_decision(
            symbol=symbol,
            q_value=q_value,
            ml_confidence=ml_confidence,
            indicators=indicators,
            current_price=current_price,
            timestamp=timestamp
        )
        
        # Update last trade time if action taken
        if action in ['buy', 'sell']:
            self.last_trade_time[symbol] = timestamp
        
        return action, confidence, reason, action_data
    
    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            'name': 'AI Momentum Scalping Engine',
            'version': '1.0',
            'description': 'AI-powered momentum scalping with Q-Learning + ML',
            'engine_type': 'ai_momentum_scalping',
            'parameters': self.config
        }

def create_strategy_adapter(strategy_type: str, portfolio: FIFOPortfolio, 
                          config: Dict[str, Any]) -> BaseStrategy:
    """Factory function to create appropriate strategy adapter"""
    
    strategy_map = {
        'dual_model': DualModelStrategyAdapter,
        'hybrid': HybridTradingStrategyAdapter,
        'momentum_scalping': MomentumScalpingStrategyAdapter,
        'ai_momentum_scalping': AIMomentumScalpingStrategyAdapter
    }
    
    adapter_class = strategy_map.get(strategy_type)
    if not adapter_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return adapter_class(portfolio, config)