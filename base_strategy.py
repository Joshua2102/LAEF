# === base_strategy.py - Base Strategy Interface ===

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from core.fifo_portfolio import FIFOPortfolio

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the standardized interface that all strategies must implement.
    """
    
    def __init__(self, portfolio: FIFOPortfolio, config: Dict[str, Any]):
        self.portfolio = portfolio
        self.config = config
        self.strategy_name = self.__class__.__name__
        self.initialized_at = datetime.now()
        
    @abstractmethod
    def should_trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if conditions are right for trading this symbol.
        
        Args:
            symbol: Stock symbol
            market_data: Dictionary containing all relevant market data
                - current_price: float
                - indicators: dict (rsi, macd, volume, etc.)
                - q_value: float (optional, for AI strategies)
                - ml_confidence: float (optional, for AI strategies)
                - timestamp: datetime
        
        Returns:
            tuple: (should_trade: bool, reason: str)
        """
        pass
    
    @abstractmethod
    def trade(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Execute trading decision for the given symbol.
        
        Args:
            symbol: Stock symbol
            market_data: Dictionary containing all relevant market data
        
        Returns:
            tuple: (action, confidence, reason, action_data)
                - action: 'buy', 'sell', or 'hold'
                - confidence: float between 0 and 1
                - reason: string explanation
                - action_data: dict with position size, stop loss, etc.
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            dict: Strategy metadata including name, version, parameters
        """
        pass
    
    def get_position_size(self, symbol: str, current_price: float, 
                         risk_level: str = 'normal') -> Dict[str, Any]:
        """
        Calculate position size based on portfolio and risk management.
        Can be overridden by specific strategies.
        """
        available_cash = self.portfolio.get_available_cash()
        max_position_pct = self.config.get('max_position_size', 0.1)
        
        if risk_level == 'aggressive':
            max_position_pct *= 1.5
        elif risk_level == 'conservative':
            max_position_pct *= 0.5
            
        max_position_value = available_cash * max_position_pct
        shares = int(max_position_value / current_price)
        
        return {
            'shares': shares,
            'position_value': shares * current_price,
            'position_pct': (shares * current_price) / self.portfolio.total_value,
            'can_buy': shares > 0 and available_cash >= shares * current_price
        }
    
    def validate_trade_timing(self, symbol: str, timestamp: datetime) -> Tuple[bool, str]:
        """
        Validate if enough time has passed since last trade.
        Can be overridden by specific strategies.
        """
        cooldown = self.config.get('cooldown_seconds', 60)
        
        if hasattr(self, 'last_trade_time') and symbol in self.last_trade_time:
            time_since_last = (timestamp - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < cooldown:
                return False, f"Cooldown period active ({cooldown - time_since_last:.0f}s remaining)"
                
        return True, "Timing validated"