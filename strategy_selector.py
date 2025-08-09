# === strategy_selector.py - Intelligent Strategy Selection Module ===

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json

class StrategySelector:
    """
    Intelligent strategy selection module that uses:
    - Live market state analysis
    - Q-values from reinforcement learning
    - Pattern history and recognition
    - Performance tracking
    
    To dynamically select the best trading strategy in real-time.
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        
        # Market state tracking
        self.market_states = deque(maxlen=lookback_periods)
        self.pattern_history = deque(maxlen=lookback_periods)
        
        # Strategy performance tracking
        self.strategy_performance_history = {}
        self.strategy_regime_performance = {}  # Track performance by market regime
        
        # Pattern recognition
        self.recognized_patterns = {
            'breakout': {'indicators': ['volume', 'price_change'], 'threshold': 0.7},
            'reversal': {'indicators': ['rsi', 'macd'], 'threshold': 0.65},
            'continuation': {'indicators': ['trend', 'momentum'], 'threshold': 0.6},
            'consolidation': {'indicators': ['volatility', 'range'], 'threshold': 0.5}
        }
        
        # Q-value integration
        self.strategy_q_values = {}  # Strategy -> Q-value mapping
        self.q_decay_rate = 0.95  # How fast Q-values decay
        
        logging.info("[STRATEGY SELECTOR] Initialized with intelligent selection capabilities")
    
    def select_strategy(self, available_strategies: List[str], 
                       market_data: Dict[str, Any],
                       q_values: Optional[Dict[str, float]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Select the best strategy based on current market conditions.
        
        Args:
            available_strategies: List of strategy names
            market_data: Current market data including indicators
            q_values: Optional Q-values from RL agent
            
        Returns:
            tuple: (selected_strategy, confidence, selection_metadata)
        """
        
        # Analyze current market state
        market_state = self._analyze_market_state(market_data)
        self.market_states.append(market_state)
        
        # Detect patterns
        detected_patterns = self._detect_patterns(market_data, market_state)
        
        # Score each strategy
        strategy_scores = {}
        for strategy in available_strategies:
            score = self._score_strategy(
                strategy, 
                market_state, 
                detected_patterns,
                market_data,
                q_values
            )
            strategy_scores[strategy] = score
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = strategy_scores[best_strategy]
        
        # Build selection metadata
        metadata = {
            'market_state': market_state,
            'detected_patterns': detected_patterns,
            'all_scores': strategy_scores,
            'selection_reason': self._build_selection_reason(best_strategy, market_state, detected_patterns),
            'timestamp': datetime.now()
        }
        
        # Update Q-values if provided
        if q_values:
            self._update_strategy_q_values(best_strategy, q_values)
        
        logging.info(f"[STRATEGY SELECTOR] Selected {best_strategy} (confidence: {confidence:.3f})")
        
        return best_strategy, confidence, metadata
    
    def _analyze_market_state(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and categorize current market state"""
        
        indicators = market_data.get('indicators', {})
        
        # Calculate market metrics
        volatility = self._calculate_volatility(market_data)
        trend_strength = self._calculate_trend_strength(market_data)
        momentum = indicators.get('momentum', 0)
        volume_ratio = self._calculate_volume_ratio(market_data)
        
        # Determine market regime
        regime = self._determine_regime(volatility, trend_strength, momentum)
        
        # Analyze price action
        price_action = self._analyze_price_action(market_data)
        
        return {
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'volume_ratio': volume_ratio,
            'price_action': price_action,
            'timestamp': market_data.get('timestamp', datetime.now())
        }
    
    def _detect_patterns(self, market_data: Dict[str, Any], 
                        market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect trading patterns in current market"""
        
        detected = []
        indicators = market_data.get('indicators', {})
        
        for pattern_name, pattern_config in self.recognized_patterns.items():
            confidence = self._evaluate_pattern(
                pattern_name,
                pattern_config,
                indicators,
                market_state
            )
            
            if confidence >= pattern_config['threshold']:
                detected.append({
                    'pattern': pattern_name,
                    'confidence': confidence,
                    'indicators_used': pattern_config['indicators']
                })
        
        # Store in history
        self.pattern_history.append({
            'timestamp': datetime.now(),
            'patterns': detected
        })
        
        return detected
    
    def _score_strategy(self, strategy_name: str, 
                       market_state: Dict[str, Any],
                       detected_patterns: List[Dict[str, Any]],
                       market_data: Dict[str, Any],
                       q_values: Optional[Dict[str, float]]) -> float:
        """Calculate comprehensive score for a strategy"""
        
        score = 0.0
        weights = {
            'regime_fit': 0.25,
            'pattern_match': 0.20,
            'performance': 0.20,
            'q_value': 0.20,
            'indicator_alignment': 0.15
        }
        
        # 1. Regime fitness score
        regime_score = self._get_regime_fitness(strategy_name, market_state['regime'])
        score += regime_score * weights['regime_fit']
        
        # 2. Pattern matching score
        pattern_score = self._get_pattern_match_score(strategy_name, detected_patterns)
        score += pattern_score * weights['pattern_match']
        
        # 3. Historical performance score
        perf_score = self._get_performance_score(strategy_name, market_state['regime'])
        score += perf_score * weights['performance']
        
        # 4. Q-value score (if available)
        if q_values and strategy_name in self.strategy_q_values:
            q_score = self._get_q_value_score(strategy_name)
            score += q_score * weights['q_value']
        else:
            # Redistribute q_value weight to other factors
            score += 0.5 * weights['q_value']  # Neutral score
        
        # 5. Indicator alignment score
        indicator_score = self._get_indicator_alignment_score(strategy_name, market_data)
        score += indicator_score * weights['indicator_alignment']
        
        return min(1.0, max(0.0, score))
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate current market volatility"""
        indicators = market_data.get('indicators', {})
        
        # Use ATR if available
        if 'atr' in indicators:
            current_price = market_data.get('current_price', 100)
            return indicators['atr'] / current_price
        
        # Fallback to simple volatility calculation
        return indicators.get('volatility', 0.02)
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend strength"""
        indicators = market_data.get('indicators', {})
        
        # Combine multiple trend indicators
        sma_trend = indicators.get('sma_trend', 0)
        ema_trend = indicators.get('ema_trend', 0)
        macd_trend = indicators.get('macd', 0)
        
        # Average and normalize
        trend_strength = (sma_trend + ema_trend + np.sign(macd_trend) * 0.5) / 2.5
        
        return abs(trend_strength)
    
    def _calculate_volume_ratio(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume ratio vs average"""
        indicators = market_data.get('indicators', {})
        
        current_volume = indicators.get('volume', 0)
        avg_volume = indicators.get('volume_avg', 1)
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0
    
    def _determine_regime(self, volatility: float, trend_strength: float, momentum: float) -> str:
        """Determine market regime based on metrics"""
        
        if volatility > 0.04:
            return 'high_volatility'
        elif trend_strength > 0.7 and abs(momentum) > 0.02:
            return 'strong_trend'
        elif trend_strength < 0.3 and volatility < 0.015:
            return 'ranging'
        elif abs(momentum) > 0.03:
            return 'momentum'
        else:
            return 'normal'
    
    def _analyze_price_action(self, market_data: Dict[str, Any]) -> str:
        """Analyze recent price action"""
        indicators = market_data.get('indicators', {})
        
        # Simple price action analysis
        close = market_data.get('current_price', 0)
        open_price = indicators.get('open', close)
        high = indicators.get('high', close)
        low = indicators.get('low', close)
        
        body_size = abs(close - open_price) / open_price if open_price > 0 else 0
        upper_wick = (high - max(close, open_price)) / open_price if open_price > 0 else 0
        lower_wick = (min(close, open_price) - low) / open_price if open_price > 0 else 0
        
        if body_size > 0.02:
            return 'strong_move'
        elif upper_wick > body_size * 2:
            return 'rejection_high'
        elif lower_wick > body_size * 2:
            return 'rejection_low'
        else:
            return 'neutral'
    
    def _evaluate_pattern(self, pattern_name: str, pattern_config: Dict,
                         indicators: Dict[str, Any], market_state: Dict[str, Any]) -> float:
        """Evaluate if a pattern is present"""
        
        confidence = 0.0
        required_indicators = pattern_config['indicators']
        
        if pattern_name == 'breakout':
            # High volume + price change
            volume_ratio = market_state.get('volume_ratio', 1.0)
            momentum = market_state.get('momentum', 0)
            confidence = min(1.0, (volume_ratio - 1.0) * 0.5 + abs(momentum) * 10)
            
        elif pattern_name == 'reversal':
            # Oversold/overbought + momentum shift
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            
            if rsi < 30 or rsi > 70:
                confidence = abs(50 - rsi) / 50
                if np.sign(macd) != np.sign(market_state.get('momentum', 0)):
                    confidence *= 1.5
                    
        elif pattern_name == 'continuation':
            # Strong trend + momentum alignment
            trend = market_state.get('trend_strength', 0)
            momentum = market_state.get('momentum', 0)
            confidence = trend * 0.7 + min(abs(momentum) * 10, 0.3)
            
        elif pattern_name == 'consolidation':
            # Low volatility + ranging
            volatility = market_state.get('volatility', 0.02)
            confidence = max(0, 1.0 - volatility * 20)
        
        return min(1.0, max(0.0, confidence))
    
    def _get_regime_fitness(self, strategy_name: str, regime: str) -> float:
        """Get strategy fitness for market regime"""
        
        fitness_map = {
            'ai_momentum_scalping': {
                'high_volatility': 0.9,
                'strong_trend': 0.8,
                'momentum': 0.95,
                'ranging': 0.4,
                'normal': 0.7
            },
            'momentum_scalping': {
                'high_volatility': 0.8,
                'strong_trend': 0.9,
                'momentum': 1.0,
                'ranging': 0.3,
                'normal': 0.6
            },
            'hybrid_trading': {
                'high_volatility': 0.7,
                'strong_trend': 0.7,
                'momentum': 0.7,
                'ranging': 0.7,
                'normal': 0.8
            },
            'dual_model': {
                'high_volatility': 0.6,
                'strong_trend': 0.7,
                'momentum': 0.6,
                'ranging': 0.8,
                'normal': 0.9
            }
        }
        
        strategy_map = fitness_map.get(strategy_name, {})
        return strategy_map.get(regime, 0.5)
    
    def _get_pattern_match_score(self, strategy_name: str, 
                                detected_patterns: List[Dict[str, Any]]) -> float:
        """Score based on pattern matches"""
        
        if not detected_patterns:
            return 0.5  # Neutral
        
        pattern_preferences = {
            'ai_momentum_scalping': {'breakout': 0.9, 'continuation': 0.8, 'reversal': 0.6},
            'momentum_scalping': {'breakout': 1.0, 'continuation': 0.9, 'reversal': 0.4},
            'hybrid_trading': {'breakout': 0.7, 'reversal': 0.8, 'continuation': 0.7},
            'dual_model': {'reversal': 0.8, 'consolidation': 0.7, 'continuation': 0.6}
        }
        
        strategy_prefs = pattern_preferences.get(strategy_name, {})
        
        total_score = 0.0
        total_confidence = 0.0
        
        for pattern in detected_patterns:
            pattern_name = pattern['pattern']
            confidence = pattern['confidence']
            preference = strategy_prefs.get(pattern_name, 0.5)
            
            total_score += preference * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            return total_score / total_confidence
        return 0.5
    
    def _get_performance_score(self, strategy_name: str, regime: str) -> float:
        """Get historical performance score"""
        
        # Check regime-specific performance
        regime_perf = self.strategy_regime_performance.get(strategy_name, {}).get(regime, {})
        
        if regime_perf:
            trades = regime_perf.get('trades', 0)
            if trades > 0:
                win_rate = regime_perf.get('wins', 0) / trades
                avg_return = regime_perf.get('avg_return', 0)
                
                # Combine win rate and return
                score = win_rate * 0.6 + min(avg_return * 10, 0.4)
                return min(1.0, max(0.0, score))
        
        # Fallback to overall performance
        overall_perf = self.strategy_performance_history.get(strategy_name, {})
        if overall_perf:
            total_trades = overall_perf.get('total_trades', 0)
            if total_trades > 0:
                win_rate = overall_perf.get('total_wins', 0) / total_trades
                return win_rate
        
        return 0.5  # Neutral for new strategies
    
    def _get_q_value_score(self, strategy_name: str) -> float:
        """Get Q-value based score"""
        
        q_data = self.strategy_q_values.get(strategy_name, {})
        if not q_data:
            return 0.5
        
        # Get recent Q-value with decay
        current_q = q_data.get('current', 0.5)
        last_update = q_data.get('last_update', datetime.now())
        
        # Apply time decay
        time_diff = (datetime.now() - last_update).total_seconds() / 3600  # Hours
        decayed_q = current_q * (self.q_decay_rate ** time_diff)
        
        return min(1.0, max(0.0, decayed_q))
    
    def _get_indicator_alignment_score(self, strategy_name: str, 
                                     market_data: Dict[str, Any]) -> float:
        """Score based on indicator alignment with strategy preferences"""
        
        indicators = market_data.get('indicators', {})
        
        # Strategy-specific indicator preferences
        if 'momentum' in strategy_name.lower():
            # Momentum strategies like strong momentum and volume
            momentum = abs(indicators.get('momentum', 0))
            volume_ratio = indicators.get('volume', 0) / indicators.get('volume_avg', 1)
            roc = abs(indicators.get('roc', 0))
            
            score = min(1.0, momentum * 10) * 0.4 + min(1.0, volume_ratio - 1) * 0.3 + min(1.0, roc * 5) * 0.3
            
        elif 'ai' in strategy_name.lower():
            # AI strategies use multiple indicators
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            
            # Good conditions: RSI not extreme, MACD aligned
            rsi_score = 1.0 - abs(50 - rsi) / 50
            macd_score = min(1.0, abs(macd) * 20)
            
            score = rsi_score * 0.5 + macd_score * 0.5
            
        elif 'hybrid' in strategy_name.lower():
            # Hybrid strategies are flexible
            score = 0.7  # Generally good alignment
            
        else:
            # Default scoring
            score = 0.5
        
        return min(1.0, max(0.0, score))
    
    def _update_strategy_q_values(self, strategy_name: str, q_values: Dict[str, float]):
        """Update Q-values for strategy"""
        
        # Extract relevant Q-value (could be symbol-specific or general)
        avg_q = np.mean(list(q_values.values())) if q_values else 0.5
        
        self.strategy_q_values[strategy_name] = {
            'current': avg_q,
            'last_update': datetime.now(),
            'history': self.strategy_q_values.get(strategy_name, {}).get('history', []) + [avg_q]
        }
        
        # Keep only recent history
        if len(self.strategy_q_values[strategy_name]['history']) > 100:
            self.strategy_q_values[strategy_name]['history'] = \
                self.strategy_q_values[strategy_name]['history'][-100:]
    
    def _build_selection_reason(self, strategy_name: str, 
                              market_state: Dict[str, Any],
                              detected_patterns: List[Dict[str, Any]]) -> str:
        """Build human-readable selection reason"""
        
        reasons = [f"Selected {strategy_name}"]
        
        # Add regime reason
        regime = market_state['regime']
        reasons.append(f"Market regime: {regime}")
        
        # Add pattern reasons
        if detected_patterns:
            pattern_names = [p['pattern'] for p in detected_patterns[:2]]  # Top 2
            reasons.append(f"Patterns detected: {', '.join(pattern_names)}")
        
        # Add key metrics
        if market_state['volatility'] > 0.03:
            reasons.append("High volatility environment")
        if market_state['momentum'] > 0.02:
            reasons.append("Strong momentum present")
        
        return " | ".join(reasons)
    
    def update_performance(self, strategy_name: str, regime: str, 
                          trade_result: Dict[str, Any]):
        """Update strategy performance tracking"""
        
        # Update overall performance
        if strategy_name not in self.strategy_performance_history:
            self.strategy_performance_history[strategy_name] = {
                'total_trades': 0,
                'total_wins': 0,
                'total_return': 0.0
            }
        
        perf = self.strategy_performance_history[strategy_name]
        perf['total_trades'] += 1
        if trade_result.get('profit', 0) > 0:
            perf['total_wins'] += 1
        perf['total_return'] += trade_result.get('return_pct', 0)
        
        # Update regime-specific performance
        if strategy_name not in self.strategy_regime_performance:
            self.strategy_regime_performance[strategy_name] = {}
        
        if regime not in self.strategy_regime_performance[strategy_name]:
            self.strategy_regime_performance[strategy_name][regime] = {
                'trades': 0,
                'wins': 0,
                'avg_return': 0.0
            }
        
        regime_perf = self.strategy_regime_performance[strategy_name][regime]
        regime_perf['trades'] += 1
        if trade_result.get('profit', 0) > 0:
            regime_perf['wins'] += 1
        
        # Update average return
        old_avg = regime_perf['avg_return']
        new_return = trade_result.get('return_pct', 0)
        regime_perf['avg_return'] = (old_avg * (regime_perf['trades'] - 1) + new_return) / regime_perf['trades']
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get strategy selection statistics"""
        
        return {
            'total_market_states': len(self.market_states),
            'recent_patterns': list(self.pattern_history)[-10:] if self.pattern_history else [],
            'strategy_performance': self.strategy_performance_history,
            'regime_performance': self.strategy_regime_performance,
            'current_q_values': self.strategy_q_values
        }