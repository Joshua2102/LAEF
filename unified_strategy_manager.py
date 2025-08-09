# === unified_strategy_manager.py - Unified Strategy Management System ===

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from core.base_strategy import BaseStrategy
from core.fifo_portfolio import FIFOPortfolio

class StrategyMode(Enum):
    """Strategy execution modes"""
    SINGLE = "single"          # Use only one strategy
    BLEND = "blend"            # Blend multiple strategies
    ADAPTIVE = "adaptive"      # Dynamically select based on market conditions
    ENSEMBLE = "ensemble"      # Vote-based ensemble approach

class UnifiedStrategyManager:
    """
    Unified Strategy Manager that orchestrates multiple trading strategies.
    
    Features:
    - Dynamic strategy selection based on market conditions
    - Strategy blending with weighted decisions
    - Centralized configuration management
    - Performance tracking per strategy
    - Risk management across strategies
    """
    
    def __init__(self, portfolio: FIFOPortfolio, mode: StrategyMode = StrategyMode.ADAPTIVE):
        self.portfolio = portfolio
        self.mode = mode
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        self.active_strategy = None
        self.strategy_weights: Dict[str, float] = {}
        
        # Market regime detection
        self.current_market_regime = "normal"  # normal, volatile, trending, ranging
        self.regime_history = []
        
        logging.info(f"[STRATEGY MANAGER] Initialized in {mode.value} mode")
    
    def register_strategy(self, name: str, strategy: BaseStrategy, weight: float = 1.0):
        """Register a new strategy with the manager"""
        self.strategies[name] = strategy
        self.strategy_weights[name] = weight
        self.strategy_performance[name] = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'last_used': None
        }
        logging.info(f"[STRATEGY MANAGER] Registered strategy: {name} (weight: {weight})")
    
    def evaluate_trade_decision(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Main entry point for trade decisions.
        Routes to appropriate strategy based on mode.
        """
        if not self.strategies:
            return 'hold', 0.0, "No strategies registered", {}
        
        # Update market regime if needed
        self._update_market_regime(market_data)
        
        # Route based on mode
        if self.mode == StrategyMode.SINGLE:
            return self._execute_single_strategy(symbol, market_data)
        elif self.mode == StrategyMode.BLEND:
            return self._execute_blended_strategies(symbol, market_data)
        elif self.mode == StrategyMode.ADAPTIVE:
            return self._execute_adaptive_strategy(symbol, market_data)
        elif self.mode == StrategyMode.ENSEMBLE:
            return self._execute_ensemble_strategies(symbol, market_data)
        else:
            return 'hold', 0.0, f"Unknown mode: {self.mode}", {}
    
    def _execute_single_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Execute using a single active strategy"""
        if not self.active_strategy or self.active_strategy not in self.strategies:
            self.active_strategy = list(self.strategies.keys())[0]
        
        strategy = self.strategies[self.active_strategy]
        
        # Check if should trade
        should_trade, pre_check_reason = strategy.should_trade(symbol, market_data)
        if not should_trade:
            return 'hold', 0.0, f"[{self.active_strategy}] {pre_check_reason}", {}
        
        # Execute trade decision
        action, confidence, reason, action_data = strategy.trade(symbol, market_data)
        
        # Add strategy metadata
        action_data['strategy_used'] = self.active_strategy
        reason = f"[{self.active_strategy}] {reason}"
        
        return action, confidence, reason, action_data
    
    def _execute_adaptive_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Adaptively select best strategy based on market conditions"""
        
        # Select best strategy for current conditions
        best_strategy = self._select_best_strategy(symbol, market_data)
        
        if not best_strategy:
            return 'hold', 0.0, "No suitable strategy for current conditions", {}
        
        strategy = self.strategies[best_strategy]
        
        # Check if should trade
        should_trade, pre_check_reason = strategy.should_trade(symbol, market_data)
        if not should_trade:
            return 'hold', 0.0, f"[{best_strategy}] {pre_check_reason}", {}
        
        # Execute trade decision
        action, confidence, reason, action_data = strategy.trade(symbol, market_data)
        
        # Adjust confidence based on strategy performance
        performance_multiplier = self._get_performance_multiplier(best_strategy)
        adjusted_confidence = min(1.0, confidence * performance_multiplier)
        
        # Add metadata
        action_data['strategy_used'] = best_strategy
        action_data['market_regime'] = self.current_market_regime
        action_data['original_confidence'] = confidence
        reason = f"[{best_strategy}] {reason} (regime: {self.current_market_regime})"
        
        return action, adjusted_confidence, reason, action_data
    
    def _execute_blended_strategies(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Blend decisions from multiple strategies"""
        decisions = []
        total_weight = 0
        
        for name, strategy in self.strategies.items():
            # Check if should trade
            should_trade, _ = strategy.should_trade(symbol, market_data)
            if not should_trade:
                continue
            
            # Get decision
            action, confidence, reason, action_data = strategy.trade(symbol, market_data)
            
            # Weight by strategy weight and performance
            weight = self.strategy_weights[name] * self._get_performance_multiplier(name)
            decisions.append({
                'strategy': name,
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'action_data': action_data,
                'weight': weight
            })
            total_weight += weight
        
        if not decisions:
            return 'hold', 0.0, "No strategies recommend trading", {}
        
        # Blend decisions
        blended_action, blended_confidence, blended_reason, blended_data = self._blend_decisions(decisions, total_weight)
        
        return blended_action, blended_confidence, blended_reason, blended_data
    
    def _execute_ensemble_strategies(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """Use ensemble voting from all strategies"""
        votes = {'buy': 0, 'sell': 0, 'hold': 0}
        strategy_decisions = []
        
        for name, strategy in self.strategies.items():
            # Check if should trade
            should_trade, _ = strategy.should_trade(symbol, market_data)
            if not should_trade:
                votes['hold'] += self.strategy_weights[name]
                continue
            
            # Get decision
            action, confidence, reason, action_data = strategy.trade(symbol, market_data)
            
            # Record vote weighted by confidence and strategy weight
            vote_weight = confidence * self.strategy_weights[name]
            votes[action] += vote_weight
            
            strategy_decisions.append({
                'strategy': name,
                'action': action,
                'confidence': confidence,
                'reason': reason
            })
        
        # Determine winning action
        winning_action = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        
        if total_votes == 0 or winning_action == 'hold':
            return 'hold', 0.0, "Ensemble voted to hold", {}
        
        # Calculate ensemble confidence
        ensemble_confidence = votes[winning_action] / total_votes
        
        # Build ensemble reason
        supporting_strategies = [d['strategy'] for d in strategy_decisions if d['action'] == winning_action]
        ensemble_reason = f"Ensemble {winning_action}: {', '.join(supporting_strategies)} agree"
        
        # Use action_data from highest confidence strategy that voted for winning action
        best_supporting = max(
            [d for d in strategy_decisions if d['action'] == winning_action],
            key=lambda x: x['confidence'],
            default=None
        )
        
        action_data = {'strategy_used': 'ensemble', 'strategies_agreed': supporting_strategies}
        
        return winning_action, ensemble_confidence, ensemble_reason, action_data
    
    def _select_best_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Optional[str]:
        """Select the best strategy for current market conditions"""
        
        scores = {}
        
        for name, strategy in self.strategies.items():
            score = 0.0
            
            # Score based on market regime fit
            regime_score = self._get_regime_fitness_score(name, self.current_market_regime)
            score += regime_score * 0.4
            
            # Score based on recent performance
            perf_score = self._get_performance_score(name)
            score += perf_score * 0.3
            
            # Score based on strategy-specific indicators
            indicator_score = self._get_indicator_fitness_score(name, market_data)
            score += indicator_score * 0.3
            
            scores[name] = score
        
        # Select strategy with highest score
        if scores:
            best_strategy = max(scores, key=scores.get)
            logging.debug(f"[STRATEGY SELECTOR] Selected {best_strategy} (score: {scores[best_strategy]:.3f})")
            return best_strategy
        
        return None
    
    def _update_market_regime(self, market_data: Dict[str, Any]):
        """Detect and update current market regime"""
        indicators = market_data.get('indicators', {})
        
        # Simple regime detection based on volatility and trend
        volatility = indicators.get('volatility', 0.02)  # Default 2%
        trend_strength = abs(indicators.get('trend', 0))
        
        if volatility > 0.04:  # High volatility
            self.current_market_regime = "volatile"
        elif trend_strength > 0.02:  # Strong trend
            self.current_market_regime = "trending"
        elif volatility < 0.01:  # Low volatility
            self.current_market_regime = "ranging"
        else:
            self.current_market_regime = "normal"
        
        # Update history
        self.regime_history.append({
            'timestamp': market_data.get('timestamp', datetime.now()),
            'regime': self.current_market_regime
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def _get_regime_fitness_score(self, strategy_name: str, regime: str) -> float:
        """Get how well a strategy fits the current market regime"""
        
        # Define regime fitness matrix
        fitness_matrix = {
            'ai_momentum_scalping': {'volatile': 0.9, 'trending': 0.8, 'normal': 0.7, 'ranging': 0.5},
            'momentum_scalping': {'volatile': 0.8, 'trending': 0.9, 'normal': 0.6, 'ranging': 0.4},
            'hybrid_trading': {'volatile': 0.7, 'trending': 0.7, 'normal': 0.8, 'ranging': 0.7},
            'dual_model': {'volatile': 0.6, 'trending': 0.7, 'normal': 0.9, 'ranging': 0.8}
        }
        
        # Get fitness score, default to 0.5 if not defined
        strategy_fitness = fitness_matrix.get(strategy_name, {})
        return strategy_fitness.get(regime, 0.5)
    
    def _get_performance_score(self, strategy_name: str) -> float:
        """Calculate performance score for a strategy"""
        perf = self.strategy_performance.get(strategy_name, {})
        
        total_trades = perf.get('trades', 0)
        if total_trades == 0:
            return 0.5  # Neutral score for untested strategies
        
        win_rate = perf.get('wins', 0) / total_trades
        sharpe = perf.get('sharpe_ratio', 0)
        
        # Combine win rate and sharpe ratio
        score = (win_rate * 0.7) + (min(sharpe / 2, 1.0) * 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _get_indicator_fitness_score(self, strategy_name: str, market_data: Dict[str, Any]) -> float:
        """Score based on how well current indicators match strategy preferences"""
        indicators = market_data.get('indicators', {})
        
        # Strategy-specific indicator preferences
        if 'momentum' in strategy_name.lower():
            # Momentum strategies prefer high momentum
            momentum = indicators.get('momentum', 0)
            return min(1.0, abs(momentum) / 0.05)  # Normalize to 0-1
            
        elif 'ai' in strategy_name.lower():
            # AI strategies use Q-value and ML confidence
            q_value = market_data.get('q_value', 0.5)
            ml_conf = market_data.get('ml_confidence', 0.5)
            return (q_value + ml_conf) / 2
            
        else:
            # Default fitness
            return 0.5
    
    def _get_performance_multiplier(self, strategy_name: str) -> float:
        """Get performance multiplier for confidence adjustment"""
        score = self._get_performance_score(strategy_name)
        # Map score to multiplier (0.8 to 1.2)
        return 0.8 + (score * 0.4)
    
    def _blend_decisions(self, decisions: List[Dict], total_weight: float) -> Tuple[str, float, str, Dict[str, Any]]:
        """Blend multiple strategy decisions"""
        
        # Aggregate by action
        action_weights = {'buy': 0, 'sell': 0, 'hold': 0}
        action_confidences = {'buy': [], 'sell': [], 'hold': []}
        reasons = []
        
        for decision in decisions:
            action = decision['action']
            weight = decision['weight']
            action_weights[action] += weight
            action_confidences[action].append(decision['confidence'] * weight)
            reasons.append(f"{decision['strategy']}: {decision['action']}")
        
        # Determine blended action
        blended_action = max(action_weights, key=action_weights.get)
        
        # Calculate blended confidence
        if action_confidences[blended_action]:
            blended_confidence = sum(action_confidences[blended_action]) / action_weights[blended_action]
        else:
            blended_confidence = 0.0
        
        # Build blended reason
        blended_reason = f"Blended decision ({', '.join(reasons)})"
        
        # Use action_data from highest weighted decision with winning action
        matching_decisions = [d for d in decisions if d['action'] == blended_action]
        if matching_decisions:
            best_decision = max(matching_decisions, key=lambda x: x['weight'])
            blended_data = best_decision['action_data'].copy()
            blended_data['strategy_used'] = 'blended'
            blended_data['strategies_involved'] = [d['strategy'] for d in decisions]
        else:
            blended_data = {'strategy_used': 'blended'}
        
        return blended_action, blended_confidence, blended_reason, blended_data
    
    def update_performance(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategy_performance:
            return
        
        perf = self.strategy_performance[strategy_name]
        perf['trades'] += 1
        perf['last_used'] = datetime.now()
        
        if trade_result.get('profit', 0) > 0:
            perf['wins'] += 1
        else:
            perf['losses'] += 1
        
        perf['total_return'] += trade_result.get('return_pct', 0)
        
        # Update Sharpe ratio (simplified)
        if perf['trades'] > 1:
            returns = trade_result.get('return_pct', 0)
            # Simplified Sharpe calculation
            perf['sharpe_ratio'] = perf['total_return'] / (perf['trades'] ** 0.5)
    
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies"""
        return list(self.strategies.keys())
    
    def set_mode(self, mode: StrategyMode):
        """Change strategy execution mode"""
        self.mode = mode
        logging.info(f"[STRATEGY MANAGER] Switched to {mode.value} mode")
    
    def set_active_strategy(self, strategy_name: str):
        """Set active strategy for SINGLE mode"""
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            logging.info(f"[STRATEGY MANAGER] Active strategy set to: {strategy_name}")
        else:
            logging.error(f"[STRATEGY MANAGER] Strategy not found: {strategy_name}")