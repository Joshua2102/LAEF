# === unified_trading_engine.py - Example Integration of Unified Strategy Manager ===

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.fifo_portfolio import FIFOPortfolio
from core.unified_strategy_manager import UnifiedStrategyManager, StrategyMode
from core.strategy_selector import StrategySelector
from core.strategy_adapters import create_strategy_adapter
from core.strategy_config_schemas import StrategyConfigManager
from core.config_manager import ConfigurationManager

class UnifiedTradingEngine:
    """
    Main trading engine that uses the unified strategy management system.
    
    This replaces individual strategy engines with a unified approach that can:
    - Dynamically select strategies based on market conditions
    - Blend multiple strategies
    - Use ensemble voting
    - Switch between single strategy mode
    """
    
    def __init__(self, portfolio: FIFOPortfolio, mode: str = 'adaptive', custom_config: Dict[str, Any] = None):
        self.portfolio = portfolio
        self.custom_config = custom_config or {}
        
        # Initialize configuration managers
        self.config_manager = ConfigurationManager()
        self.strategy_config_manager = StrategyConfigManager(self.config_manager)
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector()
        
        # Initialize unified strategy manager
        strategy_mode = StrategyMode(mode)
        self.strategy_manager = UnifiedStrategyManager(portfolio, strategy_mode)
        
        # Register all available strategies
        self._register_strategies()
        
        logging.info(f"[UNIFIED ENGINE] Initialized in {mode} mode with {len(self.strategy_manager.strategies)} strategies")
    
    def _register_strategies(self):
        """Register all available strategies with the manager"""
        
        # Define available strategies and their default weights
        strategies = [
            ('ai_momentum_scalping', 1.2),  # Higher weight for AI strategies
            ('momentum_scalping', 1.0),
            ('hybrid_trading', 0.9),
            ('dual_model', 1.1)
        ]
        
        for strategy_name, weight in strategies:
            try:
                # Get configuration for strategy
                config = self.strategy_config_manager.get_strategy_config(
                    strategy_name, 
                    self.custom_config
                )
                
                # Create strategy adapter
                strategy = create_strategy_adapter(strategy_name, self.portfolio, config)
                
                # Register with manager
                self.strategy_manager.register_strategy(strategy_name, strategy, weight)
                
                logging.info(f"[UNIFIED ENGINE] Registered {strategy_name} (weight: {weight})")
                
            except Exception as e:
                logging.error(f"[UNIFIED ENGINE] Failed to register {strategy_name}: {e}")
    
    def evaluate_trade_decision(self, symbol: str, q_value: float = None, ml_confidence: float = None,
                              indicators: Dict[str, Any] = None, current_price: float = None,
                              timestamp: datetime = None, force_sell: bool = False) -> tuple:
        """
        Main entry point for trade decisions using unified strategy management.
        
        Args:
            symbol: Stock symbol
            q_value: Q-Learning value (optional, for AI strategies)
            ml_confidence: ML model confidence (optional, for AI strategies)
            indicators: Technical indicators dictionary
            current_price: Current market price
            timestamp: Current timestamp
            force_sell: Force sell signal
            
        Returns:
            tuple: (action, confidence, reason, action_data)
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Build market data dictionary
        market_data = {
            'symbol': symbol,
            'current_price': current_price,
            'indicators': indicators or {},
            'timestamp': timestamp,
            'force_sell': force_sell
        }
        
        # Add AI signals if available
        if q_value is not None:
            market_data['q_value'] = q_value
        if ml_confidence is not None:
            market_data['ml_confidence'] = ml_confidence
        
        # If in adaptive mode, use strategy selector for intelligent selection
        if self.strategy_manager.mode == StrategyMode.ADAPTIVE:
            # Get Q-values for strategy selection
            q_values = {symbol: q_value} if q_value is not None else None
            
            # Select best strategy
            selected_strategy, selection_confidence, selection_metadata = \
                self.strategy_selector.select_strategy(
                    self.strategy_manager.get_active_strategies(),
                    market_data,
                    q_values
                )
            
            # Update active strategy for adaptive mode
            self.strategy_manager.active_strategy = selected_strategy
            
            # Log selection
            logging.debug(f"[UNIFIED ENGINE] Selected {selected_strategy} for {symbol} "
                         f"(confidence: {selection_confidence:.3f})")
        
        # Execute trade decision through strategy manager
        action, confidence, reason, action_data = self.strategy_manager.evaluate_trade_decision(
            symbol, market_data
        )
        
        # Add unified engine metadata
        action_data['engine'] = 'unified'
        action_data['mode'] = self.strategy_manager.mode.value
        
        return action, confidence, reason, action_data
    
    def update_performance(self, symbol: str, trade_result: Dict[str, Any]):
        """Update performance metrics after trade completion"""
        
        strategy_used = trade_result.get('strategy_used')
        if not strategy_used:
            return
        
        # Update strategy manager performance
        self.strategy_manager.update_performance(strategy_used, trade_result)
        
        # Update strategy selector performance
        market_regime = trade_result.get('market_regime', 'normal')
        self.strategy_selector.update_performance(strategy_used, market_regime, trade_result)
        
        logging.info(f"[UNIFIED ENGINE] Updated performance for {strategy_used} - "
                    f"Return: {trade_result.get('return_pct', 0):.2%}")
    
    def set_mode(self, mode: str):
        """Change strategy execution mode"""
        
        try:
            strategy_mode = StrategyMode(mode)
            self.strategy_manager.set_mode(strategy_mode)
            logging.info(f"[UNIFIED ENGINE] Switched to {mode} mode")
        except ValueError:
            logging.error(f"[UNIFIED ENGINE] Invalid mode: {mode}")
    
    def set_active_strategy(self, strategy_name: str):
        """Set active strategy for single mode"""
        
        if strategy_name in self.strategy_manager.strategies:
            self.strategy_manager.set_active_strategy(strategy_name)
            logging.info(f"[UNIFIED ENGINE] Active strategy set to {strategy_name}")
        else:
            logging.error(f"[UNIFIED ENGINE] Strategy not found: {strategy_name}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about all registered strategies"""
        
        info = {
            'mode': self.strategy_manager.mode.value,
            'active_strategy': self.strategy_manager.active_strategy,
            'registered_strategies': {}
        }
        
        for name, strategy in self.strategy_manager.strategies.items():
            info['registered_strategies'][name] = strategy.get_strategy_info()
        
        return info
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get strategy selection statistics"""
        
        return {
            'strategy_manager_stats': {
                'mode': self.strategy_manager.mode.value,
                'performance': self.strategy_manager.strategy_performance
            },
            'selector_stats': self.strategy_selector.get_selection_stats()
        }


def create_unified_trading_engine(portfolio: FIFOPortfolio, custom_config: Dict[str, Any] = None) -> UnifiedTradingEngine:
    """
    Factory function to create unified trading engine.
    
    This is the main entry point that replaces individual engine creation.
    """
    
    # Determine mode from config
    mode = custom_config.get('strategy_mode', 'adaptive') if custom_config else 'adaptive'
    
    return UnifiedTradingEngine(portfolio, mode, custom_config)


# Backward compatibility functions
def create_dual_model_trading_engine(portfolio: FIFOPortfolio, custom_config: Dict[str, Any] = None) -> UnifiedTradingEngine:
    """Backward compatibility: Create unified engine in single mode with dual_model strategy"""
    
    config = custom_config or {}
    config['strategy_mode'] = 'single'
    
    engine = UnifiedTradingEngine(portfolio, 'single', config)
    engine.set_active_strategy('dual_model')
    
    return engine


def create_hybrid_trading_engine(portfolio: FIFOPortfolio, custom_config: Dict[str, Any] = None) -> UnifiedTradingEngine:
    """Backward compatibility: Create unified engine in single mode with hybrid strategy"""
    
    config = custom_config or {}
    config['strategy_mode'] = 'single'
    
    engine = UnifiedTradingEngine(portfolio, 'single', config)
    engine.set_active_strategy('hybrid_trading')
    
    return engine