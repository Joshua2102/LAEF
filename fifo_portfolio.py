# === fifo_portfolio.py - FIFO Portfolio Management (FIXED) ===

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
from config import INITIAL_CASH, MAX_RISK_PER_TRADE, MAX_POSITION_SIZE, TRADE_LOG_FILE

@dataclass
class Position:
    """Individual position tracking for FIFO."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_id: str  # Unique identifier for this specific purchase

class FIFOPortfolio:
    """
    FIFO Portfolio Management System
    
    Key Features:
    - Tracks each individual purchase separately
    - Implements "oldest first" selling (FIFO)
    - Calculates profit/loss for each individual trade
    - Prevents selling at a loss unless stop-loss triggered
    - Provides detailed trade tracking and reporting
    """
    
    def __init__(self, initial_cash=None):
        self.cash = initial_cash or INITIAL_CASH
        self.initial_cash = self.cash
        
        # FIFO queues for each symbol - oldest positions first
        self.positions: Dict[str, deque] = {}  # symbol -> deque of Position objects
        
        # Trade logging
        self.trade_log = []
        self.trade_counter = 0
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Peak profit tracking for trailing stops
        self.peak_profits: Dict[str, float] = {}  # symbol -> peak unrealized P&L %
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        logging.info(f"[PORTFOLIO] Initialized FIFO portfolio with ${self.cash:,.2f}")
    
    def get_available_cash(self) -> float:
        """Get current available cash."""
        return self.cash
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get total market value of all positions in a symbol."""
        if symbol not in self.positions:
            return 0.0
        
        total_value = 0.0
        for position in self.positions[symbol]:
            total_value += position.quantity * current_price
        
        return total_value
    
    def get_total_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Get total portfolio value (cash + positions)."""
        total_value = self.cash
        
        for symbol in self.positions:
            if symbol in current_prices:
                total_value += self.get_position_value(symbol, current_prices[symbol])
        
        return total_value
    
    def get_position_quantity(self, symbol: str) -> int:
        """Get total quantity of shares owned for a symbol."""
        if symbol not in self.positions:
            return 0
        
        return sum(position.quantity for position in self.positions[symbol])
    
    def get_average_cost_basis(self, symbol: str) -> float:
        """Get weighted average cost basis for all positions in a symbol."""
        if symbol not in self.positions or len(self.positions[symbol]) == 0:
            return 0.0
        
        total_cost = 0.0
        total_quantity = 0
        
        for position in self.positions[symbol]:
            total_cost += position.quantity * position.entry_price
            total_quantity += position.quantity
        
        return total_cost / total_quantity if total_quantity > 0 else 0.0
    
    def get_position_cost_basis(self, symbol: str) -> float:
        """Alias for get_average_cost_basis for compatibility."""
        return self.get_average_cost_basis(symbol)
    
    def get_oldest_position_cost(self, symbol: str) -> Optional[float]:
        """Get the cost basis of the oldest (FIFO) position for a symbol."""
        if symbol not in self.positions or len(self.positions[symbol]) == 0:
            return None
        
        return self.positions[symbol][0].entry_price
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have any position in the given symbol."""
        return symbol in self.positions and len(self.positions[symbol]) > 0
    
    def get_position_details(self, symbol: str, current_price: float) -> dict:
        """Get detailed position information for a symbol."""
        if not self.has_position(symbol):
            return {}
        
        total_shares = self.get_position_quantity(symbol)
        avg_cost = self.get_average_cost_basis(symbol)
        current_value = total_shares * current_price
        cost_basis = total_shares * avg_cost
        unrealized_pnl = current_value - cost_basis
        pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        # Update peak profit tracking
        if symbol not in self.peak_profits:
            self.peak_profits[symbol] = pnl_pct
        else:
            self.peak_profits[symbol] = max(self.peak_profits[symbol], pnl_pct)
        
        return {
            'total_shares': total_shares,
            'avg_cost': avg_cost,
            'current_value': current_value,
            'cost_basis': cost_basis,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': pnl_pct,  # Adding this for consistency
            'pnl_pct': pnl_pct,
            'peak_unrealized_pct': self.peak_profits[symbol]
        }
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """
        Calculate appropriate position size based on risk management rules.
        
        Rules:
        1. Don't risk more than MAX_RISK_PER_TRADE of portfolio per trade
        2. Don't let any single symbol exceed MAX_POSITION_SIZE of portfolio
        """
        try:
            # Calculate risk budget per trade
            portfolio_value = self.cash  # Conservative estimate using only cash
            risk_budget = portfolio_value * MAX_RISK_PER_TRADE
            
            # Maximum shares based on risk budget
            max_shares_by_risk = int(risk_budget / current_price)
            
            # Check position size limit
            current_position_value = self.get_position_value(symbol, current_price)
            max_position_value = portfolio_value * MAX_POSITION_SIZE
            remaining_position_budget = max_position_value - current_position_value
            
            if remaining_position_budget <= 0:
                logging.info(f"[PORTFOLIO] {symbol} position size limit reached")
                return 0
            
            max_shares_by_position = int(remaining_position_budget / current_price)
            
            # Take the smaller of the two limits
            position_size = min(max_shares_by_risk, max_shares_by_position)
            
            # Make sure we have enough cash
            total_cost = position_size * current_price
            if total_cost > self.cash:
                position_size = int(self.cash / current_price)
            
            return max(0, position_size)
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Position size calculation failed: {e}")
            return 0
    
    def buy(self, symbol: str, price: float, quantity: int = None, timestamp: datetime = None) -> bool:
        """
        Buy shares using FIFO tracking.
        
        Args:
            symbol: Stock symbol
            price: Price per share
            quantity: Number of shares (if None, calculate automatically)
            timestamp: Transaction timestamp
            
        Returns:
            bool: True if purchase successful
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self.calculate_position_size(symbol, price)
            
            if quantity <= 0:
                logging.warning(f"[PORTFOLIO] Cannot buy {symbol}: quantity={quantity}")
                return False
            
            total_cost = quantity * price
            
            # Check if we have enough cash
            if total_cost > self.cash:
                # Adjust quantity to what we can afford
                quantity = int(self.cash / price)
                total_cost = quantity * price
                
                if quantity <= 0:
                    logging.warning(f"[PORTFOLIO] Insufficient cash to buy {symbol}")
                    return False
            
            # Create new position
            self.trade_counter += 1
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                entry_id=f"{symbol}_{self.trade_counter}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Add to FIFO queue
            if symbol not in self.positions:
                self.positions[symbol] = deque()
            
            self.positions[symbol].append(position)
            
            # Update cash
            self.cash -= total_cost
            
            # Log the trade
            self._log_trade({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'total_cost': total_cost,
                'entry_id': position.entry_id,
                'remaining_cash': self.cash
            })
            
            logging.info(f"[PORTFOLIO] BUY: {quantity} shares of {symbol} @ ${price:.2f} (Cost: ${total_cost:.2f})")
            return True
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Buy failed for {symbol}: {e}")
            return False
    
    def sell(self, symbol: str, price: float, quantity: int = None, 
             force_sell: bool = False, timestamp: datetime = None) -> Tuple[bool, float, str]:
        """
        Sell shares using FIFO (oldest first).
        
        Args:
            symbol: Stock symbol
            price: Current price per share
            quantity: Number of shares to sell (if None, sell oldest position)
            force_sell: If True, sell even at a loss (for stop-loss)
            timestamp: Transaction timestamp
            
        Returns:
            Tuple[bool, float, str]: (success, realized_pnl, reason)
        """
        try:
            timestamp = timestamp or datetime.now()
            
            if symbol not in self.positions or len(self.positions[symbol]) == 0:
                return False, 0.0, f"No positions in {symbol}"
            
            # If quantity not specified, sell the entire oldest position
            if quantity is None:
                quantity = self.positions[symbol][0].quantity
            
            total_quantity_available = self.get_position_quantity(symbol)
            if quantity > total_quantity_available:
                quantity = total_quantity_available
            
            if quantity <= 0:
                return False, 0.0, "No shares to sell"
            
            # FIFO selling: start with oldest positions
            remaining_to_sell = quantity
            total_realized_pnl = 0.0
            total_proceeds = 0.0
            positions_sold = []
            
            while remaining_to_sell > 0 and len(self.positions[symbol]) > 0:
                oldest_position = self.positions[symbol][0]
                
                # Note: Loss protection removed for dual model trading logic
                # The dual model already includes sophisticated loss management:
                # - Q-Value conviction deterioration for cutting losses
                # - Built-in stop loss thresholds  
                # - ML confidence for optimal exit timing
                
                # Determine how many shares to sell from this position
                shares_from_this_position = min(remaining_to_sell, oldest_position.quantity)
                
                # Calculate P&L for this portion
                proceeds_from_position = shares_from_this_position * price
                cost_basis_for_position = shares_from_this_position * oldest_position.entry_price
                pnl_from_position = proceeds_from_position - cost_basis_for_position
                
                total_proceeds += proceeds_from_position
                total_realized_pnl += pnl_from_position
                
                # Record this sale
                positions_sold.append({
                    'entry_id': oldest_position.entry_id,
                    'entry_price': oldest_position.entry_price,
                    'entry_time': oldest_position.entry_time,
                    'shares_sold': shares_from_this_position,
                    'sale_price': price,
                    'pnl': pnl_from_position
                })
                
                # Update the position
                if shares_from_this_position == oldest_position.quantity:
                    # Sold entire position
                    self.positions[symbol].popleft()
                else:
                    # Partial sale - reduce quantity
                    oldest_position.quantity -= shares_from_this_position
                
                remaining_to_sell -= shares_from_this_position
            
            # Update cash with proceeds
            self.cash += total_proceeds
            
            # Update statistics
            self.total_realized_pnl += total_realized_pnl
            self.total_trades += len(positions_sold)
            if total_realized_pnl > 0:
                self.winning_trades += len(positions_sold)
            
            # Log each individual sale
            for sale in positions_sold:
                self._log_trade({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': sale['shares_sold'],
                    'price': price,
                    'total_proceeds': sale['shares_sold'] * price,
                    'entry_id': sale['entry_id'],
                    'entry_price': sale['entry_price'],
                    'entry_time': sale['entry_time'],
                    'pnl': sale['pnl'],
                    'remaining_cash': self.cash
                })
            
            # Clean up empty position queue
            if symbol in self.positions and len(self.positions[symbol]) == 0:
                del self.positions[symbol]
                # Reset peak profit tracking when position is closed
                if symbol in self.peak_profits:
                    del self.peak_profits[symbol]
            
            # FIXED: Safe profit percentage calculation
            avg_cost = self.get_average_cost_basis(symbol)
            if avg_cost > 0:
                total_cost = quantity * avg_cost
                profit_pct = (total_realized_pnl / total_cost) * 100 if total_cost > 0 else 0.0
            else:
                profit_pct = 0.0
                
            reason = f"FIFO sale: {quantity} shares, P&L: ${total_realized_pnl:.2f} ({profit_pct:.1f}%)"
            
            logging.info(f"[PORTFOLIO] SELL: {quantity} shares of {symbol} @ ${price:.2f} | P&L: ${total_realized_pnl:.2f}")
            return True, total_realized_pnl, reason
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Sell failed for {symbol}: {e}")
            return False, 0.0, f"Error: {e}"
    
    def should_sell_for_profit(self, symbol: str, current_price: float, 
                              min_profit_pct: float = 2.0) -> Tuple[bool, str]:
        """
        Check if we should sell based on FIFO profit potential.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            min_profit_pct: Minimum profit percentage required
            
        Returns:
            Tuple[bool, str]: (should_sell, reason)
        """
        try:
            oldest_cost = self.get_oldest_position_cost(symbol)
            if oldest_cost is None:
                return False, "No positions"
            
            profit_pct = ((current_price - oldest_cost) / oldest_cost) * 100
            
            if profit_pct >= min_profit_pct:
                return True, f"Profit: {profit_pct:.1f}% on oldest position (${oldest_cost:.2f} → ${current_price:.2f})"
            else:
                return False, f"Insufficient profit: {profit_pct:.1f}% < {min_profit_pct}%"
                
        except Exception as e:
            return False, f"Error checking profit: {e}"
    
    def _log_trade(self, trade_data: dict):
        """Log trade data for analysis and reporting."""
        self.trade_log.append(trade_data)
    
    def save_trade_log(self, filename: str = None):
        """Save trade log to CSV file."""
        try:
            filename = filename or TRADE_LOG_FILE
            
            if not self.trade_log:
                logging.info("[PORTFOLIO] No trades to save")
                return
            
            df = pd.DataFrame(self.trade_log)
            df.to_csv(filename, index=False)
            logging.info(f"[PORTFOLIO] Saved {len(self.trade_log)} trades to {filename}")
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Failed to save trade log: {e}")
    
    def get_performance_summary(self) -> dict:
        """Get portfolio performance summary."""
        try:
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                'initial_cash': self.initial_cash,
                'current_cash': self.cash,
                'total_realized_pnl': self.total_realized_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'active_positions': {symbol: len(positions) for symbol, positions in self.positions.items()},
                'cash_remaining_pct': (self.cash / self.initial_cash) * 100
            }
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Performance summary failed: {e}")
            return {}
    
    def print_positions(self):
        """Print current positions for debugging."""
        print("\n=== CURRENT POSITIONS ===")
        for symbol, positions in self.positions.items():
            print(f"\n{symbol}:")
            for i, pos in enumerate(positions):
                print(f"  {i+1}. {pos.quantity} shares @ ${pos.entry_price:.2f} (ID: {pos.entry_id})")
        print(f"\nCash: ${self.cash:,.2f}")
        print("=" * 25)
    
    # =============================================================================
    # NEW AGGRESSIVE STRATEGY METHODS
    # =============================================================================
    
    def get_all_positions_with_performance(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Get all positions with current performance metrics for ranking."""
        all_positions = []
        
        for symbol, positions in self.positions.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            for position in positions:
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                
                # Calculate days held
                time_held = datetime.now() - position.entry_time
                days_held = time_held.total_seconds() / (24 * 3600)
                
                all_positions.append({
                    'symbol': symbol,
                    'position': position,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'minutes_held': time_held.total_seconds() / 60,
                    'market_value': current_price * position.quantity
                })
        
        return all_positions
    
    def get_worst_performers(self, current_prices: Dict[str, float], percentage: float = 0.10) -> List[Dict]:
        """Get the worst performing positions for liquidation."""
        all_positions = self.get_all_positions_with_performance(current_prices)
        
        if not all_positions:
            return []
        
        # Sort by performance (worst first)
        all_positions.sort(key=lambda x: x['pnl_pct'])
        
        # Calculate how many positions to liquidate
        num_to_liquidate = max(1, int(len(all_positions) * percentage))
        
        return all_positions[:num_to_liquidate]
    
    def force_liquidate_worst_performers(self, current_prices: Dict[str, float], 
                                       percentage: float = 0.10) -> List[Dict]:
        """Force sell the worst performing positions (overrides everything)."""
        from config import DAILY_WORST_PERFORMER_LIQUIDATION, WORST_PERFORMER_PERCENTAGE
        
        if not DAILY_WORST_PERFORMER_LIQUIDATION:
            return []
        
        worst_performers = self.get_worst_performers(current_prices, percentage)
        liquidated = []
        
        logging.info(f"[PORTFOLIO] FORCE LIQUIDATION: Selling {len(worst_performers)} worst performers")
        
        for perf_data in worst_performers:
            symbol = perf_data['symbol']
            position = perf_data['position']
            current_price = perf_data['current_price']
            
            # Force sell this specific position (bypass profit protection)
            success, pnl, reason = self._force_sell_position(symbol, position, current_price)
            
            if success:
                liquidated.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': perf_data['pnl_pct'],
                    'days_held': perf_data['days_held'],
                    'reason': 'WORST_PERFORMER_LIQUIDATION'
                })
                
                logging.info(f"[PORTFOLIO] LIQUIDATED: {symbol} - {position.quantity} shares @ ${current_price:.2f} "
                           f"| P&L: ${pnl:.2f} ({perf_data['pnl_pct']:.1f}%) | Held: {perf_data['days_held']:.1f} days")
        
        return liquidated
    
    def check_mandatory_exits(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check for positions that must be exited after 5 days (overrides everything)."""
        from config import MANDATORY_EXIT_MINUTES
        
        all_positions = self.get_all_positions_with_performance(current_prices)
        mandatory_exits = []
        
        for perf_data in all_positions:
            if perf_data['minutes_held'] >= MANDATORY_EXIT_MINUTES:
                symbol = perf_data['symbol']
                position = perf_data['position']
                current_price = perf_data['current_price']
                
                # Force sell (overrides everything)
                success, pnl, reason = self._force_sell_position(symbol, position, current_price)
                
                if success:
                    mandatory_exits.append({
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': perf_data['pnl_pct'],
                        'days_held': perf_data['days_held'],
                        'reason': 'MANDATORY_5DAY_EXIT'
                    })
                    
                    logging.info(f"[PORTFOLIO] MANDATORY EXIT: {symbol} - {position.quantity} shares @ ${current_price:.2f} "
                               f"| P&L: ${pnl:.2f} ({perf_data['pnl_pct']:.1f}%) | Held: {perf_data['days_held']:.1f} days")
        
        return mandatory_exits
    
    def check_stop_loss_overrides(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check for stop loss conditions that override FIFO protection."""
        from config import OVERRIDE_FIFO_FOR_STOP_LOSS, TRADING_THRESHOLDS
        
        if not OVERRIDE_FIFO_FOR_STOP_LOSS:
            return []
        
        stop_loss_threshold = TRADING_THRESHOLDS['stop_loss_pct']  # 0.970 for -3%
        all_positions = self.get_all_positions_with_performance(current_prices)
        stop_loss_exits = []
        
        for perf_data in all_positions:
            current_price = perf_data['current_price']
            position = perf_data['position']
            
            # Check if stop loss triggered
            if current_price / position.entry_price <= stop_loss_threshold:
                symbol = perf_data['symbol']
                
                # Force sell (overrides FIFO protection)
                success, pnl, reason = self._force_sell_position(symbol, position, current_price)
                
                if success:
                    stop_loss_exits.append({
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': perf_data['pnl_pct'],
                        'days_held': perf_data['days_held'],
                        'reason': 'STOP_LOSS_OVERRIDE'
                    })
                    
                    logging.info(f"[PORTFOLIO] STOP LOSS OVERRIDE: {symbol} - {position.quantity} shares @ ${current_price:.2f} "
                               f"| P&L: ${pnl:.2f} ({perf_data['pnl_pct']:.1f}%) | Loss: {perf_data['pnl_pct']:.1f}%")
        
        return stop_loss_exits
    
    def _force_sell_position(self, symbol: str, position: Position, current_price: float) -> Tuple[bool, float, str]:
        """Force sell a specific position, bypassing all protections."""
        try:
            if symbol not in self.positions:
                return False, 0.0, "No positions found"
            
            # Find and remove the specific position
            position_queue = self.positions[symbol]
            position_found = False
            
            for i, pos in enumerate(position_queue):
                if pos.entry_id == position.entry_id:
                    # Remove this specific position
                    position_queue.remove(pos)
                    position_found = True
                    break
            
            if not position_found:
                return False, 0.0, "Position not found"
            
            # Calculate proceeds and P&L
            proceeds = position.quantity * current_price
            cost = position.quantity * position.entry_price
            pnl = proceeds - cost
            
            # Update cash
            self.cash += proceeds
            
            # Update statistics
            self.total_realized_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            
            # Log the forced sale
            self._log_trade({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'FORCE_SELL',
                'quantity': position.quantity,
                'price': current_price,
                'total_proceeds': proceeds,
                'entry_id': position.entry_id,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'pnl': pnl,
                'remaining_cash': self.cash
            })
            
            # Clean up empty position queue
            if len(position_queue) == 0:
                del self.positions[symbol]
            
            return True, pnl, f"Force sold {position.quantity} shares"
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Force sell failed for {symbol}: {e}")
            return False, 0.0, f"Error: {e}"
    
    def get_position_details(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Get detailed position information for dual model trading logic.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            Dict with position details or None if no position exists
        """
        if not self.has_position(symbol):
            return None
        
        try:
            # Get position metrics
            total_quantity = self.get_position_quantity(symbol)
            avg_cost = self.get_average_cost_basis(symbol)
            current_value = total_quantity * current_price
            total_cost = total_quantity * avg_cost
            unrealized_pnl = current_value - total_cost
            unrealized_pnl_pct = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0.0
            
            return {
                'symbol': symbol,
                'quantity': total_quantity,
                'avg_cost_basis': avg_cost,
                'current_price': current_price,
                'current_value': current_value,
                'total_cost': total_cost,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct
            }
            
        except Exception as e:
            logging.error(f"[PORTFOLIO] Error getting position details for {symbol}: {e}")
            return None

    def apply_aggressive_strategy(self, current_prices: Dict[str, float]) -> Dict:
        """Apply all new aggressive strategy rules."""
        results = {
            'mandatory_exits': [],
            'worst_performer_liquidations': [],
            'stop_loss_overrides': [],
            'total_forced_sales': 0,
            'total_pnl_realized': 0.0
        }
        
        # 1. Check mandatory 5-day exits first (highest priority)
        mandatory_exits = self.check_mandatory_exits(current_prices)
        results['mandatory_exits'] = mandatory_exits
        
        # 2. Force liquidate worst 10% of remaining positions
        worst_performers = self.force_liquidate_worst_performers(current_prices)
        results['worst_performer_liquidations'] = worst_performers
        
        # 3. Apply stop loss overrides to remaining positions  
        stop_loss_exits = self.check_stop_loss_overrides(current_prices)
        results['stop_loss_overrides'] = stop_loss_exits
        
        # Calculate totals
        all_exits = mandatory_exits + worst_performers + stop_loss_exits
        results['total_forced_sales'] = len(all_exits)
        results['total_pnl_realized'] = sum(exit_data['pnl'] for exit_data in all_exits)
        
        if all_exits:
            logging.info(f"[PORTFOLIO] AGGRESSIVE STRATEGY: {len(all_exits)} positions force-closed, "
                        f"P&L: ${results['total_pnl_realized']:.2f}")
        
        return results
