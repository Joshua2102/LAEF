# === momentum_scalping_engine.py - Momentum-Based Micro-Scalping Engine ===

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from config import TRADING_THRESHOLDS
from core.fifo_portfolio import FIFOPortfolio

class MomentumScalpingEngine:
    """
    Momentum-Based Micro-Scalping Trading Engine
    
    Strategy:
    - Ultra-fast momentum detection (1-5 minute timeframes)
    - Quick entry/exit on momentum shifts
    - Small profit targets (0.1-0.5%)
    - Tight stop losses (0.05-0.2%)
    - High frequency trading with small position sizes
    
    Key Features:
    1. Momentum indicators: Rate of Change, Velocity, Acceleration
    2. Volume-based confirmation
    3. Micro profit targets with quick exits
    4. Dynamic position sizing based on volatility
    5. Time-based exits (max hold time: 15-60 minutes)
    """
    
    def __init__(self, portfolio: FIFOPortfolio, thresholds: dict = None, custom_config: dict = None):
        self.portfolio = portfolio
        self.thresholds = thresholds or self._get_default_scalping_thresholds()
        
        # Apply custom config overrides if provided
        if custom_config:
            for key, value in custom_config.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
        
        # Scalping-specific parameters
        self.MOMENTUM_BUY_THRESHOLD = self.thresholds['momentum_buy_threshold']
        self.MOMENTUM_SELL_THRESHOLD = self.thresholds['momentum_sell_threshold']
        self.MICRO_PROFIT_TARGET = self.thresholds['micro_profit_target']
        self.MICRO_STOP_LOSS = self.thresholds['micro_stop_loss']
        self.MAX_HOLD_MINUTES = self.thresholds['max_hold_minutes']
        self.VOLUME_CONFIRMATION_MULTIPLIER = self.thresholds['volume_confirmation']
        
        # Momentum tracking
        self.momentum_history = {}  # symbol -> list of momentum values
        self.last_trade_time = {}   # symbol -> datetime of last trade
        
        logging.info("[MOMENTUM SCALPING] Trading engine initialized")
        logging.info(f"[MOMENTUM SCALPING] Micro profit target: {self.MICRO_PROFIT_TARGET:.3f}%")
        logging.info(f"[MOMENTUM SCALPING] Micro stop loss: {self.MICRO_STOP_LOSS:.3f}%")
        logging.info(f"[MOMENTUM SCALPING] Max hold time: {self.MAX_HOLD_MINUTES} minutes")
    
    def _get_default_scalping_thresholds(self) -> dict:
        """Default thresholds optimized for micro-scalping"""
        return {
            # Momentum thresholds
            'momentum_buy_threshold': 2.0,      # 2% momentum required for buy
            'momentum_sell_threshold': -1.0,    # -1% momentum triggers sell
            'momentum_acceleration_min': 0.5,   # Minimum acceleration for entry
            
            # Micro profit/loss targets
            'micro_profit_target': 0.15,        # 0.15% profit target
            'micro_stop_loss': 0.08,            # 0.08% stop loss
            'trailing_micro_stop': 0.05,        # 0.05% trailing stop
            
            # Volume confirmation
            'volume_confirmation': 1.5,         # Volume must be 1.5x average
            'volume_spike_threshold': 2.0,      # 2x volume = strong signal
            
            # Time-based limits
            'max_hold_minutes': 30,             # Maximum hold time
            'cooldown_seconds': 60,             # Cooldown between trades
            
            # Position sizing (smaller for scalping)
            'scalp_position_pct': 0.02,         # 2% of portfolio per scalp
            'max_scalp_positions': 3,           # Max concurrent scalp positions
            
            # Technical indicators
            'roc_period': 5,                    # Rate of change period
            'velocity_period': 3,               # Price velocity period
            'volume_sma_period': 10,            # Volume moving average
        }
    
    def evaluate_trade_decision(self, symbol: str, current_price: float, 
                              indicators: dict, timestamp: datetime = None) -> tuple:
        """
        Main scalping decision function
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            indicators: Dict with technical indicators and recent price data
            timestamp: Current timestamp
            
        Returns:
            tuple: (decision, confidence, detailed_reason, action_data)
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Check cooldown period
            if not self._check_cooldown(symbol, timestamp):
                return 'hold', 0.0, f"HOLD: {symbol} in cooldown period", {}
            
            # Check if we currently hold this position
            has_position = self.portfolio.has_position(symbol)
            
            if not has_position:
                return self._evaluate_scalp_entry(symbol, current_price, indicators, timestamp)
            else:
                return self._evaluate_scalp_exit(symbol, current_price, indicators, timestamp)
                
        except Exception as e:
            logging.error(f"[MOMENTUM SCALPING] Decision evaluation failed for {symbol}: {e}")
            return 'hold', 0.0, f"Error in scalping evaluation: {e}", {}
    
    def _evaluate_scalp_entry(self, symbol: str, current_price: float, 
                             indicators: dict, timestamp: datetime) -> tuple:
        """Evaluate momentum-based scalping entry"""
        
        # Calculate momentum indicators
        momentum_score = self._calculate_momentum_score(symbol, indicators)
        volume_confirmation = self._check_volume_confirmation(indicators)
        acceleration = self._calculate_price_acceleration(indicators)
        
        # Check maximum concurrent scalp positions
        current_scalp_positions = len([s for s in self.portfolio.positions.keys()])
        if current_scalp_positions >= self.thresholds['max_scalp_positions']:
            return 'hold', 0.2, f"HOLD: Max scalp positions reached ({current_scalp_positions})", {}
        
        # PRIMARY: Strong momentum with volume confirmation
        if (momentum_score >= self.MOMENTUM_BUY_THRESHOLD and 
            volume_confirmation and 
            acceleration >= self.thresholds['momentum_acceleration_min']):
            
            # Calculate micro position size
            position_data = self._calculate_scalp_position_size(symbol, current_price)
            
            if position_data['can_buy']:
                confidence = min(0.9, momentum_score / 5.0 + 0.3)  # Scale confidence
                
                reason = (f"BUY: Strong momentum ({momentum_score:.2f}), "
                         f"volume {volume_confirmation:.1f}x, "
                         f"acceleration {acceleration:.2f}%")
                
                # Record entry for tracking
                self.last_trade_time[symbol] = timestamp
                
                return 'buy', confidence, reason, position_data
            else:
                return 'hold', 0.3, f"HOLD: Momentum signal but {position_data['block_reason']}", {}
        
        # SECONDARY: Moderate momentum (lower confidence)
        elif momentum_score >= self.MOMENTUM_BUY_THRESHOLD * 0.7 and volume_confirmation:
            position_data = self._calculate_scalp_position_size(symbol, current_price)
            
            if position_data['can_buy']:
                confidence = min(0.7, momentum_score / 6.0 + 0.2)
                
                reason = (f"BUY: Moderate momentum ({momentum_score:.2f}), "
                         f"volume {volume_confirmation:.1f}x (speculative)")
                
                self.last_trade_time[symbol] = timestamp
                return 'buy', confidence, reason, position_data
            else:
                return 'hold', 0.3, f"HOLD: Weak momentum signal, {position_data['block_reason']}", {}
        
        else:
            reason = (f"HOLD: Insufficient momentum ({momentum_score:.2f} < {self.MOMENTUM_BUY_THRESHOLD}), "
                     f"volume {volume_confirmation:.1f}x")
            return 'hold', momentum_score / 10.0, reason, {}
    
    def _evaluate_scalp_exit(self, symbol: str, current_price: float, 
                            indicators: dict, timestamp: datetime) -> tuple:
        """Evaluate momentum-based scalping exit"""
        
        # Get position details
        position_info = self.portfolio.get_position_details(symbol, current_price)
        if not position_info:
            return 'hold', 0.0, "No position found", {}
        
        current_pnl_pct = position_info['unrealized_pnl_pct']
        entry_time = self._get_position_entry_time(symbol)
        hold_time_minutes = (timestamp - entry_time).total_seconds() / 60 if entry_time else 0
        
        # Calculate current momentum
        momentum_score = self._calculate_momentum_score(symbol, indicators)
        
        # PRIMARY: Micro profit target hit
        if current_pnl_pct >= self.MICRO_PROFIT_TARGET:
            sell_data = self._calculate_sell_position(symbol, current_price, position_info)
            reason = f"SELL: Micro profit target hit ({current_pnl_pct:.3f}% >= {self.MICRO_PROFIT_TARGET:.3f}%)"
            return 'sell', 0.95, reason, sell_data
        
        # SECONDARY: Momentum reversal
        if momentum_score <= self.MOMENTUM_SELL_THRESHOLD:
            sell_data = self._calculate_sell_position(symbol, current_price, position_info)
            reason = f"SELL: Momentum reversal ({momentum_score:.2f} <= {self.MOMENTUM_SELL_THRESHOLD})"
            return 'sell', 0.9, reason, sell_data
        
        # TERTIARY: Micro stop loss
        if current_pnl_pct <= -self.MICRO_STOP_LOSS:
            sell_data = self._calculate_sell_position(symbol, current_price, position_info)
            reason = f"SELL: Micro stop loss ({current_pnl_pct:.3f}% <= -{self.MICRO_STOP_LOSS:.3f}%)"
            return 'sell', 0.95, reason, sell_data
        
        # QUATERNARY: Time-based exit
        if hold_time_minutes >= self.MAX_HOLD_MINUTES:
            sell_data = self._calculate_sell_position(symbol, current_price, position_info)
            reason = f"SELL: Max hold time reached ({hold_time_minutes:.1f} >= {self.MAX_HOLD_MINUTES} min)"
            return 'sell', 0.8, reason, sell_data
        
        # TRAILING MICRO STOP (for profitable positions)
        if current_pnl_pct > 0:
            peak_profit = position_info.get('peak_unrealized_pct', current_pnl_pct)
            trailing_stop = self.thresholds['trailing_micro_stop']
            
            if current_pnl_pct < peak_profit - trailing_stop:
                sell_data = self._calculate_sell_position(symbol, current_price, position_info)
                reason = f"SELL: Trailing micro stop (peak: {peak_profit:.3f}% - {trailing_stop:.3f}%)"
                return 'sell', 0.85, reason, sell_data
        
        # DEFAULT: HOLD
        reason = (f"HOLD: P&L {current_pnl_pct:.3f}%, momentum {momentum_score:.2f}, "
                 f"hold time {hold_time_minutes:.1f}min")
        confidence = 0.5 + (momentum_score / 10.0)
        return 'hold', min(confidence, 0.8), reason, {}
    
    def _calculate_momentum_score(self, symbol: str, indicators: dict) -> float:
        """Calculate comprehensive momentum score"""
        try:
            # Get recent price data
            close_prices = indicators.get('recent_closes', [])
            if len(close_prices) < 5:
                return 0.0
            
            # Rate of Change (ROC)
            roc_period = self.thresholds['roc_period']
            if len(close_prices) >= roc_period:
                roc = ((close_prices[-1] / close_prices[-roc_period]) - 1) * 100
            else:
                roc = 0.0
            
            # Price velocity (rate of price change)
            velocity_period = self.thresholds['velocity_period']
            if len(close_prices) >= velocity_period:
                velocity = sum([(close_prices[i] - close_prices[i-1]) / close_prices[i-1] 
                               for i in range(-velocity_period, 0)]) * 100
            else:
                velocity = 0.0
            
            # Price acceleration (change in velocity)
            acceleration = self._calculate_price_acceleration(indicators)
            
            # Combined momentum score
            momentum_score = (roc * 0.5) + (velocity * 0.3) + (acceleration * 0.2)
            
            # Update momentum history
            if symbol not in self.momentum_history:
                self.momentum_history[symbol] = []
            
            self.momentum_history[symbol].append(momentum_score)
            # Keep only last 20 momentum readings
            if len(self.momentum_history[symbol]) > 20:
                self.momentum_history[symbol] = self.momentum_history[symbol][-20:]
            
            return momentum_score
            
        except Exception as e:
            logging.error(f"[MOMENTUM SCALPING] Momentum calculation failed for {symbol}: {e}")
            return 0.0
    
    def _calculate_price_acceleration(self, indicators: dict) -> float:
        """Calculate price acceleration (change in velocity)"""
        try:
            close_prices = indicators.get('recent_closes', [])
            if len(close_prices) < 6:
                return 0.0
            
            # Calculate two velocity readings
            velocity1 = ((close_prices[-2] - close_prices[-4]) / close_prices[-4]) * 100
            velocity2 = ((close_prices[-1] - close_prices[-3]) / close_prices[-3]) * 100
            
            acceleration = velocity2 - velocity1
            return acceleration
            
        except Exception as e:
            logging.error(f"[MOMENTUM SCALPING] Acceleration calculation failed: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, indicators: dict) -> float:
        """Check if volume confirms the momentum signal"""
        try:
            current_volume = indicators.get('volume', 0)
            volume_sma = indicators.get('volume_sma', 0)
            
            if volume_sma <= 0:
                return 1.0  # Default if no volume data
            
            volume_ratio = current_volume / volume_sma
            return volume_ratio
            
        except Exception as e:
            logging.error(f"[MOMENTUM SCALPING] Volume confirmation failed: {e}")
            return 1.0
    
    def _check_cooldown(self, symbol: str, timestamp: datetime) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.last_trade_time:
            return True
        
        last_trade = self.last_trade_time[symbol]
        cooldown_seconds = self.thresholds['cooldown_seconds']
        time_since_last = (timestamp - last_trade).total_seconds()
        
        return time_since_last >= cooldown_seconds
    
    def _get_position_entry_time(self, symbol: str) -> Optional[datetime]:
        """Get the entry time of the current position"""
        if not self.portfolio.has_position(symbol):
            return None
        
        # Get the oldest position's entry time
        if symbol in self.portfolio.positions:
            oldest_position = self.portfolio.positions[symbol][0]
            return oldest_position.entry_time
        
        return None
    
    def _calculate_scalp_position_size(self, symbol: str, current_price: float) -> dict:
        """Calculate position size for scalping (smaller sizes)"""
        try:
            # Use smaller position sizes for scalping
            portfolio_value = self.portfolio.get_available_cash()
            scalp_amount = portfolio_value * self.thresholds['scalp_position_pct']
            
            shares = int(scalp_amount / current_price)
            cost = shares * current_price
            
            if shares <= 0 or cost > self.portfolio.get_available_cash():
                return {
                    'can_buy': False,
                    'block_reason': f"Insufficient funds for scalp position (need ${cost:.2f})",
                    'shares': 0,
                    'cost': 0
                }
            
            return {
                'can_buy': True,
                'shares': shares,
                'cost': cost,
                'price': current_price
            }
            
        except Exception as e:
            logging.error(f"[MOMENTUM SCALPING] Position size calculation failed: {e}")
            return {'can_buy': False, 'block_reason': f"Calculation error: {e}", 'shares': 0, 'cost': 0}
    
    def _calculate_sell_position(self, symbol: str, current_price: float, position_info: dict) -> dict:
        """Calculate sell position data"""
        shares_to_sell = position_info['total_shares']
        proceeds = shares_to_sell * current_price
        
        return {
            'shares': shares_to_sell,
            'price': current_price,
            'proceeds': proceeds,
            'expected_pnl': position_info['unrealized_pnl']
        }


# Compatibility function for existing system
def create_momentum_scalping_engine(portfolio: FIFOPortfolio, custom_config: dict = None):
    """Create momentum scalping engine instance"""
    return MomentumScalpingEngine(portfolio, custom_config=custom_config)