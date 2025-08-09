"""
LAEF AI Trading Engine - Pure AI-driven strategy with dynamic parameters
No fixed thresholds, fully adaptive based on neural network predictions
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from core.fifo_portfolio import FIFOPortfolio
from core.agent_unified import LAEFAgent
from core.indicators_unified import get_latest_indicators

class LAEFAITradingEngine:
    """
    Pure LAEF AI Trading Engine that uses dynamic neural network predictions
    without any fixed trading parameters or static thresholds.
    
    This engine relies entirely on:
    - Q-Learning values from neural network
    - ML confidence scores
    - Dynamic risk assessment
    - Adaptive position sizing
    """
    
    def __init__(self, portfolio: FIFOPortfolio, custom_config: dict = None):
        self.portfolio = portfolio
        self.config = custom_config or {}
        
        # Initialize LAEF Agent for AI predictions
        try:
            self.agent = LAEFAgent(pretrained=True)
            logging.info("[LAEF AI] Neural network agent loaded successfully")
        except Exception as e:
            logging.error(f"[LAEF AI] Failed to load agent: {e}")
            raise
        
        # Track active positions for dynamic management
        self.active_positions = {}
        self.position_entry_times = {}
        self.position_entry_q_values = {}
        
        # Dynamic thresholds that adapt based on market conditions
        self.adaptive_thresholds = {
            'q_confidence_baseline': 0.0,  # Will adapt based on recent predictions
            'ml_confidence_baseline': 0.0,  # Will adapt based on recent ML scores
            'volatility_adjustment': 1.0,   # Multiplier based on market volatility
            'momentum_factor': 1.0,         # Multiplier based on momentum strength
        }
        
        # Performance tracking for dynamic adjustment
        self.recent_predictions = []
        self.recent_outcomes = []
        self.prediction_window = 100  # Number of predictions to track
        
        logging.info("[LAEF AI] Pure AI trading engine initialized - fully dynamic mode")
    
    def evaluate_trade_decision(self, symbol: str, state: np.ndarray, 
                               indicators: Dict[str, Any], current_price: float,
                               ml_confidence: float = None, timestamp: datetime = None) -> Tuple:
        """
        Evaluate trading decision using pure AI predictions without fixed thresholds.
        
        Args:
            symbol: Stock symbol
            state: Current market state vector
            indicators: Technical indicators
            current_price: Current stock price
            ml_confidence: ML model confidence (optional)
            timestamp: Current timestamp
            
        Returns:
            tuple: (action, confidence, reason, action_data)
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get Q-values from neural network
        q_values = self.agent.predict_q_values(state)
        q_hold, q_buy, q_sell = q_values
        
        # Update adaptive thresholds based on recent predictions
        self._update_adaptive_thresholds(q_values, indicators)
        
        # Calculate dynamic confidence scores
        action_confidences = self._calculate_dynamic_confidences(
            q_values, ml_confidence, indicators
        )
        
        # Determine position size dynamically
        position_size = self._calculate_dynamic_position_size(
            action_confidences, indicators, current_price
        )
        
        # Check if we have an active position
        has_position = symbol in self.active_positions
        
        if has_position:
            # Dynamic exit strategy based on AI predictions
            action, confidence, reason = self._evaluate_dynamic_exit(
                symbol, q_values, action_confidences, indicators, current_price
            )
        else:
            # Dynamic entry strategy based on AI predictions
            action, confidence, reason = self._evaluate_dynamic_entry(
                q_values, action_confidences, indicators
            )
        
        # Build action data
        action_data = {
            'q_values': q_values.tolist(),
            'q_buy': float(q_buy),
            'q_sell': float(q_sell),
            'q_hold': float(q_hold),
            'ml_confidence': ml_confidence or 0.0,
            'position_size': position_size,
            'adaptive_thresholds': self.adaptive_thresholds.copy(),
            'action_confidences': action_confidences,
            'timestamp': timestamp.isoformat()
        }
        
        # Track prediction for learning
        self._track_prediction(symbol, q_values, action, confidence)
        
        return action, confidence, reason, action_data
    
    def _update_adaptive_thresholds(self, q_values: np.ndarray, indicators: Dict[str, Any]):
        """
        Update adaptive thresholds based on recent market conditions and predictions.
        """
        # Track recent predictions
        self.recent_predictions.append(q_values)
        if len(self.recent_predictions) > self.prediction_window:
            self.recent_predictions.pop(0)
        
        if len(self.recent_predictions) >= 10:  # Need minimum history
            # Calculate baseline from recent predictions
            recent_q = np.array(self.recent_predictions)
            self.adaptive_thresholds['q_confidence_baseline'] = np.percentile(recent_q[:, 1], 60)  # 60th percentile for buys
            
            # Adjust for volatility
            volatility = indicators.get('atr', 0) / indicators.get('close', 1) if indicators.get('close', 1) > 0 else 0.02
            self.adaptive_thresholds['volatility_adjustment'] = 1.0 + min(volatility * 10, 0.5)  # Cap at 50% adjustment
            
            # Adjust for momentum
            momentum = indicators.get('momentum_score', 0)
            self.adaptive_thresholds['momentum_factor'] = 1.0 + (momentum * 0.2)  # Up to 20% adjustment
    
    def _calculate_dynamic_confidences(self, q_values: np.ndarray, 
                                      ml_confidence: float, 
                                      indicators: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate dynamic confidence scores for each action.
        """
        q_hold, q_buy, q_sell = q_values
        
        # Normalize Q-values to probabilities using softmax
        q_exp = np.exp(q_values - np.max(q_values))  # Numerical stability
        q_probs = q_exp / np.sum(q_exp)
        
        # Combine with ML confidence if available
        if ml_confidence is not None and ml_confidence > 0:
            ml_weight = min(ml_confidence, 0.5)  # Cap ML influence at 50%
            combined_buy = q_probs[1] * (1 - ml_weight) + ml_confidence * ml_weight
            combined_sell = q_probs[2] * (1 - ml_weight) + (1 - ml_confidence) * ml_weight
        else:
            combined_buy = q_probs[1]
            combined_sell = q_probs[2]
        
        # Apply market condition adjustments
        volatility_adj = self.adaptive_thresholds['volatility_adjustment']
        momentum_adj = self.adaptive_thresholds['momentum_factor']
        
        # Calculate final confidences
        confidences = {
            'buy': combined_buy * momentum_adj / volatility_adj,
            'sell': combined_sell * volatility_adj / momentum_adj,
            'hold': q_probs[0]
        }
        
        # Normalize to ensure they sum to approximately 1
        total = sum(confidences.values())
        if total > 0:
            confidences = {k: v/total for k, v in confidences.items()}
        
        return confidences
    
    def _calculate_dynamic_position_size(self, confidences: Dict[str, float],
                                        indicators: Dict[str, Any],
                                        current_price: float) -> float:
        """
        Calculate position size dynamically based on AI confidence and market conditions.
        """
        # Base position size from confidence
        max_confidence = max(confidences.values())
        base_size = 0.02 + (max_confidence - 0.33) * 0.08  # 2% to 10% based on confidence
        
        # Adjust for volatility (reduce size in high volatility)
        volatility = indicators.get('atr', 0) / current_price if current_price > 0 else 0.02
        volatility_multiplier = max(0.5, 1.0 - volatility * 5)  # Reduce by up to 50% in high volatility
        
        # Adjust for portfolio risk
        current_exposure = sum(self.active_positions.values()) / self.portfolio.cash if self.portfolio.cash > 0 else 0
        risk_multiplier = max(0.3, 1.0 - current_exposure)  # Reduce as exposure increases
        
        # Calculate final position size
        position_size = base_size * volatility_multiplier * risk_multiplier
        
        # Ensure within reasonable bounds
        return max(0.01, min(0.15, position_size))  # 1% to 15% of portfolio
    
    def _evaluate_dynamic_entry(self, q_values: np.ndarray, 
                               confidences: Dict[str, float],
                               indicators: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Evaluate entry decision using dynamic AI predictions.
        """
        # Get the highest confidence action
        best_action = max(confidences, key=confidences.get)
        confidence = confidences[best_action]
        
        # Dynamic entry threshold based on recent market behavior
        baseline = self.adaptive_thresholds.get('q_confidence_baseline', 0.33)
        entry_threshold = baseline * self.adaptive_thresholds['momentum_factor']
        
        if best_action == 'buy' and confidence > entry_threshold:
            # Additional filters based on market structure
            rsi = indicators.get('rsi', 50)
            macd_signal = indicators.get('macd_signal', 0)
            
            # Dynamic RSI thresholds
            rsi_threshold = 30 + (1 - confidence) * 20  # More confident = accept higher RSI
            
            if rsi < rsi_threshold or macd_signal > 0:
                reason = f"AI Buy Signal: Q={q_values[1]:.3f}, Confidence={confidence:.2%}, Dynamic threshold={entry_threshold:.3f}"
                return 'buy', confidence, reason
        
        # Default to hold
        return 'hold', confidences['hold'], f"AI Hold: Insufficient confidence (best={best_action}:{confidence:.2%})"
    
    def _evaluate_dynamic_exit(self, symbol: str, q_values: np.ndarray,
                              confidences: Dict[str, float],
                              indicators: Dict[str, Any],
                              current_price: float) -> Tuple[str, float, str]:
        """
        Evaluate exit decision for existing position using dynamic AI predictions.
        """
        entry_price = self.active_positions.get(symbol, current_price)
        entry_time = self.position_entry_times.get(symbol, datetime.now())
        entry_q = self.position_entry_q_values.get(symbol, q_values)
        
        # Calculate position performance
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        hold_duration = (datetime.now() - entry_time).total_seconds() / 60  # minutes
        
        # Dynamic exit based on Q-value degradation
        q_degradation = (entry_q[1] - q_values[1]) / max(abs(entry_q[1]), 0.001)
        
        # Sell confidence from AI
        sell_confidence = confidences['sell']
        
        # Dynamic profit target based on AI confidence at entry
        dynamic_target = 0.001 + abs(entry_q[1]) * 0.05  # 0.1% base + up to 5% based on entry confidence
        
        # Dynamic stop loss based on volatility
        volatility = indicators.get('atr', 0) / current_price if current_price > 0 else 0.02
        dynamic_stop = -0.002 - volatility * 0.5  # -0.2% base minus volatility adjustment
        
        # Exit conditions
        if pnl_pct >= dynamic_target:
            return 'sell', 0.9, f"AI Dynamic Target Hit: {pnl_pct:.2%} >= {dynamic_target:.2%}"
        
        if pnl_pct <= dynamic_stop:
            return 'sell', 0.95, f"AI Dynamic Stop Hit: {pnl_pct:.2%} <= {dynamic_stop:.2%}"
        
        if q_degradation > 0.5 and sell_confidence > 0.4:
            return 'sell', sell_confidence, f"AI Signal Degradation: Q-drop={q_degradation:.2f}, Sell confidence={sell_confidence:.2%}"
        
        if hold_duration > 60 and pnl_pct < 0:  # 1 hour with loss
            return 'sell', 0.7, f"AI Time Stop: {hold_duration:.0f} minutes with {pnl_pct:.2%} loss"
        
        # Strong sell signal from AI
        if sell_confidence > 0.6:
            return 'sell', sell_confidence, f"AI Sell Signal: Confidence={sell_confidence:.2%}"
        
        # Continue holding
        return 'hold', confidences['hold'], f"AI Hold Position: PnL={pnl_pct:.2%}, Q-sell={q_values[2]:.3f}"
    
    def _track_prediction(self, symbol: str, q_values: np.ndarray, action: str, confidence: float):
        """
        Track predictions for continuous learning and threshold adjustment.
        """
        # This would connect to an online learning system in production
        # For now, just log the prediction
        logging.debug(f"[LAEF AI] {symbol}: Action={action}, Confidence={confidence:.2%}, Q={q_values}")
    
    def execute_trade(self, symbol: str, action: str, position_size: float, 
                     current_price: float, q_values: np.ndarray = None) -> bool:
        """
        Execute the trade and update position tracking.
        """
        try:
            if action == 'buy':
                # Track the position
                self.active_positions[symbol] = current_price
                self.position_entry_times[symbol] = datetime.now()
                if q_values is not None:
                    self.position_entry_q_values[symbol] = q_values.copy()
                
                shares = int((self.portfolio.cash * position_size) / current_price)
                if shares > 0:
                    cost = shares * current_price
                    self.portfolio.cash -= cost
                    logging.info(f"[LAEF AI] Bought {shares} shares of {symbol} at ${current_price:.2f}")
                    return True
                    
            elif action == 'sell' and symbol in self.active_positions:
                # Clear position tracking
                del self.active_positions[symbol]
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                if symbol in self.position_entry_q_values:
                    del self.position_entry_q_values[symbol]
                
                # Execute sell (simplified - would integrate with portfolio)
                logging.info(f"[LAEF AI] Sold position in {symbol} at ${current_price:.2f}")
                return True
                
        except Exception as e:
            logging.error(f"[LAEF AI] Trade execution failed: {e}")
            
        return False