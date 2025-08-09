# === HYBRID DAY TRADING + SWING TRADING ENGINE ===

import logging
import numpy as np
from datetime import datetime, timedelta
from core.fifo_portfolio import FIFOPortfolio

class HybridTradingEngine:
    """
    AGGRESSIVE ENGINE: Day Trading + Swing Trading for High Volatility
    
    Strategy:
    1. Quick day trades (1.5% profit, same day)
    2. Swing trades (4% profit, 1-7 days)
    3. Momentum-based entries
    4. Higher risk tolerance
    """
    
    def __init__(self, portfolio: FIFOPortfolio, custom_config: dict = None):
        self.portfolio = portfolio
        
        # AGGRESSIVE DEFAULT SETTINGS
        self.thresholds = {
            'q_buy': 0.40,                    # Aggressive entry
            'q_sell': 0.20,                   # Quick exit
            'ml_profit_peak': 0.40,           # Fast profit detection
            'day_trade_profit': 0.015,        # 1.5% day trade target
            'swing_trade_profit': 0.04,       # 4% swing trade target
            'stop_loss_pct': 0.93,            # 7% stop loss
            'momentum_threshold': 0.02,        # 2% momentum
            'rsi_oversold': 20,               # Aggressive oversold
            'rsi_overbought': 80,             # Less restrictive
            'max_risk_per_trade': 0.08,       # 8% risk
            'max_position_size': 0.25,        # 25% position
            'max_hold_hours': 168,            # 7 days max
        }
        
        # Apply custom config
        if custom_config:
            self.thresholds.update(custom_config)
        
        # Track trade timing
        self.position_entry_times = {}
        
        logging.info("[HYBRID ENGINE] Initialized for day + swing trading")
        logging.info(f"[HYBRID ENGINE] Aggressive settings - Q-buy: {self.thresholds['q_buy']}")
        logging.info(f"[HYBRID ENGINE] Risk per trade: {self.thresholds['max_risk_per_trade']*100}%")
    
    def evaluate_trade_decision(self, symbol: str, q_value: float, ml_confidence: float, 
                              indicators: dict, current_price: float, 
                              current_time: datetime = None, force_sell: bool = False) -> tuple:
        """
        HYBRID DECISION: Combines day trading + swing trading logic
        """
        try:
            if current_time is None:
                current_time = datetime.now()
            
            has_position = self.portfolio.has_position(symbol)
            
            # Get technical indicators
            rsi = indicators.get('rsi', 50.0)
            macd = indicators.get('macd', 0.0)
            volume = indicators.get('volume', 0)
            
            if not has_position:
                return self._evaluate_hybrid_entry(symbol, q_value, ml_confidence, rsi, macd, current_price, current_time)
            else:
                return self._evaluate_hybrid_exit(symbol, q_value, ml_confidence, rsi, macd, current_price, current_time, force_sell)
                
        except Exception as e:
            logging.error(f"[HYBRID ENGINE] Decision error for {symbol}: {e}")
            return 'hold', 0.0, f"Error: {e}", {}
    
    def _evaluate_hybrid_entry(self, symbol: str, q_value: float, ml_confidence: float,
                             rsi: float, macd: float, current_price: float, current_time: datetime) -> tuple:
        """AGGRESSIVE ENTRY: Multiple entry conditions for higher volatility"""
        
        # ENTRY CONDITION 1: Q-Value Momentum (Day Trading)
        if q_value >= self.thresholds['q_buy']:
            position_data = self._calculate_aggressive_position_size(symbol, current_price, 'momentum')
            
            if position_data['can_buy']:
                confidence = min(0.95, q_value * 0.7 + ml_confidence * 0.3)
                reason = f"BUY: Momentum entry Q={q_value:.3f} >= {self.thresholds['q_buy']}"
                
                # Mark as day trade candidate
                position_data['trade_type'] = 'day_trade'
                self.position_entry_times[symbol] = current_time
                
                return 'buy', confidence, reason, position_data
        
        # ENTRY CONDITION 2: Oversold Bounce (Swing Trading)
        elif rsi <= self.thresholds['rsi_oversold'] and q_value >= 0.30:
            position_data = self._calculate_aggressive_position_size(symbol, current_price, 'oversold')
            
            if position_data['can_buy']:
                confidence = 0.8
                reason = f"BUY: Oversold bounce RSI={rsi:.1f}, Q={q_value:.3f}"
                
                # Mark as swing trade candidate
                position_data['trade_type'] = 'swing_trade'
                self.position_entry_times[symbol] = current_time
                
                return 'buy', confidence, reason, position_data
        
        # ENTRY CONDITION 3: Strong ML Signal (Aggressive)
        elif ml_confidence >= 0.65 and q_value >= 0.35:
            position_data = self._calculate_aggressive_position_size(symbol, current_price, 'ml_signal')
            
            if position_data['can_buy']:
                confidence = 0.85
                reason = f"BUY: Strong ML signal ML={ml_confidence:.3f}, Q={q_value:.3f}"
                
                position_data['trade_type'] = 'hybrid'
                self.position_entry_times[symbol] = current_time
                
                return 'buy', confidence, reason, position_data
        
        # ENTRY CONDITION 4: MACD Momentum
        elif macd > 0.1 and q_value >= 0.35 and rsi < 70:
            position_data = self._calculate_aggressive_position_size(symbol, current_price, 'macd')
            
            if position_data['can_buy']:
                confidence = 0.75
                reason = f"BUY: MACD momentum MACD={macd:.3f}, Q={q_value:.3f}"
                
                position_data['trade_type'] = 'momentum'
                self.position_entry_times[symbol] = current_time
                
                return 'buy', confidence, reason, position_data
        
        # DEFAULT: HOLD
        reason = f"HOLD: No entry signal (Q={q_value:.3f}, RSI={rsi:.1f}, ML={ml_confidence:.3f})"
        return 'hold', q_value * 0.5, reason, {}
    
    def _evaluate_hybrid_exit(self, symbol: str, q_value: float, ml_confidence: float,
                            rsi: float, macd: float, current_price: float, 
                            current_time: datetime, force_sell: bool) -> tuple:
        """HYBRID EXIT: Different logic for day trades vs swing trades"""
        
        # Get position details
        position_info = self.portfolio.get_position_details(symbol, current_price)
        if not position_info:
            return 'hold', 0.0, "No position found", {}
        
        # Calculate hold time
        entry_time = self.position_entry_times.get(symbol, current_time)
        hold_time = current_time - entry_time
        hold_hours = hold_time.total_seconds() / 3600
        is_same_day = hold_time.days == 0
        
        # FORCE SELL
        if force_sell:
            sell_data = self._calculate_sell_data(symbol, current_price, position_info)
            return 'sell', 0.9, "FORCE SELL", sell_data
        
        # PRIORITY 1: Stop Loss (Universal)
        loss_pct = abs(position_info['unrealized_pnl_pct'])
        stop_loss_threshold = (1 - self.thresholds['stop_loss_pct']) * 100
        
        if position_info['unrealized_pnl'] < 0 and loss_pct >= stop_loss_threshold:
            sell_data = self._calculate_sell_data(symbol, current_price, position_info)
            reason = f"SELL: Stop loss {loss_pct:.1f}% >= {stop_loss_threshold:.1f}%"
            
            # Clean up tracking
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]
            
            return 'sell', 0.95, reason, sell_data
        
        # PRIORITY 2: Day Trade Quick Profit (Same Day)
        if is_same_day and position_info['unrealized_pnl'] > 0:
            profit_pct = position_info['unrealized_pnl_pct']
            day_target = self.thresholds['day_trade_profit'] * 100
            
            if profit_pct >= day_target:
                sell_data = self._calculate_sell_data(symbol, current_price, position_info)
                reason = f"SELL: Day trade profit {profit_pct:.1f}% >= {day_target:.1f}% (same day)"
                
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                
                return 'sell', 0.9, reason, sell_data
        
        # PRIORITY 3: Swing Trade Profit (Multi-day)
        if not is_same_day and position_info['unrealized_pnl'] > 0:
            profit_pct = position_info['unrealized_pnl_pct']
            swing_target = self.thresholds['swing_trade_profit'] * 100
            
            if profit_pct >= swing_target:
                sell_data = self._calculate_sell_data(symbol, current_price, position_info)
                reason = f"SELL: Swing trade profit {profit_pct:.1f}% >= {swing_target:.1f}% ({hold_hours:.1f}h)"
                
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                
                return 'sell', 0.88, reason, sell_data
        
        # PRIORITY 4: ML Profit Peak Detection
        if ml_confidence >= self.thresholds['ml_profit_peak'] and position_info['unrealized_pnl'] > 0:
            sell_data = self._calculate_sell_data(symbol, current_price, position_info)
            reason = f"SELL: ML profit peak ML={ml_confidence:.3f}, P&L=${position_info['unrealized_pnl']:.2f}"
            
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]
            
            return 'sell', 0.85, reason, sell_data
        
        # PRIORITY 5: Q-Value Conviction Loss
        if q_value <= self.thresholds['q_sell']:
            sell_data = self._calculate_sell_data(symbol, current_price, position_info)
            reason = f"SELL: Conviction lost Q={q_value:.3f} <= {self.thresholds['q_sell']}"
            
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]
            
            return 'sell', 0.8, reason, sell_data
        
        # PRIORITY 6: Maximum Hold Time (Risk Management)
        max_hours = self.thresholds['max_hold_hours']
        if hold_hours >= max_hours:
            sell_data = self._calculate_sell_data(symbol, current_price, position_info)
            reason = f"SELL: Max hold time {hold_hours:.1f}h >= {max_hours}h"
            
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]
            
            return 'sell', 0.75, reason, sell_data
        
        # PRIORITY 7: Overbought with Profit (Technical Exit)
        if rsi >= self.thresholds['rsi_overbought'] and position_info['unrealized_pnl'] > 0:
            # Only sell if we have decent profit
            if position_info['unrealized_pnl_pct'] >= 1.0:  # At least 1% profit
                sell_data = self._calculate_sell_data(symbol, current_price, position_info)
                reason = f"SELL: Overbought exit RSI={rsi:.1f} with {position_info['unrealized_pnl_pct']:.1f}% profit"
                
                if symbol in self.position_entry_times:
                    del self.position_entry_times[symbol]
                
                return 'sell', 0.7, reason, sell_data
        
        # DEFAULT: HOLD
        trade_type = 'day' if is_same_day else 'swing'
        reason = f"HOLD: {trade_type} trade, Q={q_value:.3f}, P&L=${position_info['unrealized_pnl']:.2f} ({position_info['unrealized_pnl_pct']:.1f}%), {hold_hours:.1f}h"
        
        return 'hold', (q_value + ml_confidence) / 2, reason, {}
    
    def _calculate_aggressive_position_size(self, symbol: str, price: float, entry_type: str) -> dict:
        """AGGRESSIVE: Larger position sizes for higher volatility"""
        try:
            available_cash = self.portfolio.cash
            
            if available_cash < 100:
                return {'can_buy': False, 'block_reason': 'insufficient cash'}
            
            # Adjust position size based on entry type
            risk_multipliers = {
                'momentum': 1.0,      # Full risk for momentum
                'oversold': 0.8,      # Slightly less for oversold
                'ml_signal': 1.2,     # More for strong ML signals
                'macd': 0.9           # Moderate for MACD
            }
            
            multiplier = risk_multipliers.get(entry_type, 1.0)
            
            # Calculate aggressive position size
            max_risk = available_cash * self.thresholds['max_risk_per_trade'] * multiplier
            max_position = available_cash * self.thresholds['max_position_size']
            
            # Use larger position for aggressive strategy
            target_amount = min(max_risk * 3, max_position)  # 3x risk for position
            target_amount = max(target_amount, 200)  # Minimum $200 trade
            
            if target_amount > available_cash * 0.95:
                target_amount = available_cash * 0.95
            
            shares = max(1, int(target_amount / price))
            actual_cost = shares * price
            
            if actual_cost > available_cash:
                return {'can_buy': False, 'block_reason': f'cost ${actual_cost:.2f} > cash ${available_cash:.2f}'}
            
            return {
                'can_buy': True,
                'shares': shares,
                'cost': actual_cost,
                'target_amount': target_amount,
                'entry_type': entry_type,
                'risk_multiplier': multiplier
            }
            
        except Exception as e:
            return {'can_buy': False, 'block_reason': f'calculation error: {e}'}
    
    def _calculate_sell_data(self, symbol: str, price: float, position_info: dict) -> dict:
        """Calculate sell data"""
        total_shares = position_info.get('total_shares', 0)
        return {
            'shares': total_shares,
            'proceeds': total_shares * price,
            'pnl': position_info.get('unrealized_pnl', 0)
        }

# Factory function
def create_hybrid_trading_engine(portfolio: FIFOPortfolio, custom_config: dict = None):
    """Create the hybrid day+swing trading engine"""
    return HybridTradingEngine(portfolio, custom_config)
