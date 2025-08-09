# === live_trader_alpaca.py - FIXED VERSION WITH DUAL MODEL ENGINE ===

import os
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from typing import List, Dict

from config import TICKER_FILE, COOLDOWN_MINUTES
from data.data_fetcher_unified import fetch_stock_data, get_latest_price
from core.state_utils_unified import create_state_vector, extract_raw_indicators
from core.agent_unified import LAEFAgent
from core.fifo_portfolio import FIFOPortfolio
from core.dual_model_trading_logic import DualModelTradingEngine, create_dual_model_trading_engine  # FIXED: Using Dual Model
from trading.alpaca_integration import AlpacaTrader
from utils import setup_logging, load_symbols_from_csv

# Set up logging using centralized function
logger = setup_logging('logs/live_trader.log')

class LAEFLiveTrader:
    """
    FIXED: Live trading system now uses DualModelTradingEngine.
    This should provide realistic win rates and proper risk management.
    """
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.is_running = False
        self.last_action_time = {}
        
        # Initialize Alpaca API
        self.alpaca = AlpacaTrader(paper_trading=paper_trading)
        
        # Initialize ML components
        self.agent = LAEFAgent(pretrained=True)
        
        # For paper trading, we still use our FIFO portfolio
        # For live trading, we sync with Alpaca positions
        account_info = self.alpaca.get_account_info()
        initial_cash = account_info.get('cash', 100000)
        
        self.portfolio = FIFOPortfolio(initial_cash)
        
        # FIXED: Use DualModelTradingEngine instead of TradingDecisionEngine
        self.trading_engine = create_dual_model_trading_engine(self.portfolio)
        
        # Market hours (Eastern Time)
        self.eastern = pytz.timezone('US/Eastern')
        
        mode = 'PAPER' if paper_trading else 'LIVE'
        logger.info(f"[LIVE] FIXED: Initialized {mode} trader with ${initial_cash:,.2f} using DualModelEngine")
        
        # Sync positions if live trading
        if not paper_trading:
            self._sync_positions_with_alpaca()
    
    def _sync_positions_with_alpaca(self):
        """Sync our portfolio with actual Alpaca positions (for live trading)."""
        try:
            logger.info("[LIVE] Syncing positions with Alpaca...")
            alpaca_positions = self.alpaca.get_all_positions()
            
            for pos in alpaca_positions:
                symbol = pos['symbol']
                qty = pos['qty']
                avg_price = pos['avg_entry_price']
                
                logger.info(f"[LIVE] Found Alpaca position: {qty} shares of {symbol} @ ${avg_price:.2f}")
                
                # Add to our FIFO tracker (approximate - we don't have exact entry times)
                # This is a simplification - in production you'd want to track this better
                self.portfolio.buy(symbol, avg_price, qty, datetime.now())
            
            logger.info(f"[LIVE] Synced {len(alpaca_positions)} positions")
            
        except Exception as e:
            logger.error(f"[LIVE] Failed to sync positions: {e}")
    
    def load_trading_symbols(self) -> List[str]:
        """Load symbols approved for live trading using centralized utility."""
        return load_symbols_from_csv(TICKER_FILE)
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open using Alpaca API."""
        try:
            return self.alpaca.is_market_open()
        except Exception as e:
            logger.error(f"[LIVE] Error checking market status: {e}")
            return False
    
    def wait_for_market_open(self):
        """Wait until the market opens."""
        logger.info("[LIVE] Waiting for market to open...")
        
        while not self.is_market_open():
            now = datetime.now(self.eastern)
            
            if now.weekday() >= 5:  # Weekend
                logger.info("[LIVE] Market closed for weekend. Sleeping 4 hours...")
                time.sleep(14400)  # 4 hours
            else:
                # Weekday - check how long until market opens
                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                
                if now.time() < market_open.time():
                    # Before market open
                    wait_seconds = (market_open - now).total_seconds()
                    wait_minutes = wait_seconds / 60
                    logger.info(f"[LIVE] Market opens in {wait_minutes:.1f} minutes. Waiting...")
                    time.sleep(min(1800, wait_seconds))  # Sleep max 30 minutes at a time
                else:
                    # After market close
                    logger.info("[LIVE] Market closed for the day. Sleeping 4 hours...")
                    time.sleep(14400)  # 4 hours
        
        logger.info("[LIVE] ✅ Market is open! Starting trading...")
    
    def start_trading(self, symbols: List[str] = None, cycle_delay: int = 60):
        """
        FIXED: Start live trading with dual-model engine.
        """
        try:
            if symbols is None:
                symbols = self.load_trading_symbols()
            
            if not symbols:
                logger.error("[LIVE] No symbols to trade")
                return
            
            mode = 'PAPER' if self.paper_trading else 'LIVE'
            logger.info(f"[LIVE] FIXED: Starting {mode} trading on {len(symbols)} symbols with DualModelEngine")
            self.is_running = True
            
            while self.is_running:
                try:
                    # Wait for market to be open
                    if not self.is_market_open():
                        self.wait_for_market_open()
                        continue
                    
                    # Process all symbols in this cycle
                    self._run_trading_cycle(symbols)
                    
                    # Print portfolio status every 10 cycles
                    if hasattr(self, '_cycle_count'):
                        self._cycle_count += 1
                    else:
                        self._cycle_count = 1
                    
                    if self._cycle_count % 10 == 0:
                        self._print_portfolio_status()
                    
                    # Wait before next cycle
                    logger.info(f"[LIVE] Cycle complete. Sleeping {cycle_delay} seconds...")
                    time.sleep(cycle_delay)
                    
                except KeyboardInterrupt:
                    logger.info("[LIVE] Received interrupt signal. Stopping trading...")
                    self.stop_trading()
                    break
                    
                except Exception as e:
                    logger.error(f"[LIVE] Error in trading loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
            
            logger.info("[LIVE] Trading stopped.")
            
        except Exception as e:
            logger.error(f"[LIVE] Failed to start trading: {e}")
    
    def stop_trading(self):
        """Stop the trading loop gracefully."""
        self.is_running = False
        
        # Save final state
        self.portfolio.save_trade_log()
        
        # Print final portfolio summary
        self._print_final_summary()
    
    def _run_trading_cycle(self, symbols: List[str]):
        """Run one complete trading cycle on all symbols."""
        try:
            logger.info(f"[LIVE] Starting trading cycle on {len(symbols)} symbols...")
            
            decisions_made = 0
            actions_taken = 0
            
            for symbol in symbols:
                try:
                    # Check cooldown
                    if self._is_in_cooldown(symbol):
                        continue
                    
                    # Process this symbol
                    action_taken = self._process_symbol_live(symbol)
                    
                    decisions_made += 1
                    if action_taken:
                        actions_taken += 1
                    
                    # Small delay between symbols to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"[LIVE] Error processing {symbol}: {e}")
                    continue
            
            # Calculate execution rate
            execution_rate = (actions_taken / max(1, decisions_made)) * 100
            logger.info(f"[LIVE] FIXED: Cycle complete - {decisions_made} decisions, {actions_taken} actions ({execution_rate:.1f}% execution rate)")
            
        except Exception as e:
            logger.error(f"[LIVE] Trading cycle failed: {e}")
    
    def _process_symbol_live(self, symbol: str) -> bool:
        """
        FIXED: Process a single symbol using dual-model logic.
        """
        try:
            # Fetch recent data for this symbol
            df = fetch_stock_data(symbol, interval="5m", period="2d")
            
            if df is None or len(df) < 50:
                logger.warning(f"[LIVE] Insufficient data for {symbol}")
                return False
            
            # Get the latest window for state creation
            window = df.tail(50)
            current_row = df.iloc[-1]
            current_price = current_row['close']
            timestamp = pd.Timestamp.now()
            
            # Also get real-time price from Alpaca
            real_time_price = self.alpaca.get_current_price(symbol)
            if real_time_price:
                current_price = real_time_price
            
            # Create state vector for ML prediction
            state = create_state_vector(window)
            if state is None:
                logger.warning(f"[LIVE] Failed to create state for {symbol}")
                return False
            
            # Get ML prediction - both Q-values and action prediction
            q_values = self.agent.predict_q_values(state)  # [hold, buy, sell]
            best_action = self.agent.predict_action(state)   # 0=hold, 1=buy, 2=sell
            max_q_value = float(np.max(q_values))
            
            # Convert action to ML confidence score
            # If model prefers buy (action=1), use buy Q-value as confidence
            # If model prefers sell (action=2), use sell Q-value as confidence  
            # If model prefers hold (action=0), use a lower confidence
            if best_action == 1:  # Buy signal
                ml_confidence = min(1.0, max(0.5, q_values[1]))  # Buy Q-value
            elif best_action == 2:  # Sell signal
                ml_confidence = min(1.0, max(0.3, q_values[2]))  # Sell Q-value
            else:  # Hold signal
                ml_confidence = min(0.6, max(0.1, q_values[0]))  # Hold Q-value
            
            # Extract indicators for trading logic
            indicators = extract_raw_indicators(window)
            
            # FIXED: Make dual-model decision with proper ML interpretation
            decision, confidence, reason, action_data = self.trading_engine.evaluate_trade_decision(
                symbol=symbol,
                q_value=max_q_value,
                ml_confidence=ml_confidence,
                indicators=indicators,
                current_price=current_price
            )
            
            # Log the decision with additional context
            action_names = ['hold', 'buy', 'sell']
            logger.info(f"[LIVE] {symbol} ${current_price:.2f} | ML: {action_names[best_action]} ({ml_confidence:.3f}) | Q-values: [{q_values[0]:.3f}, {q_values[1]:.3f}, {q_values[2]:.3f}] | Decision: {decision.upper()} ({confidence:.2f})")
            logger.info(f"[LIVE] {symbol}: {reason}")
            
            # Execute decision
            action_taken = False
            if decision == 'buy' and action_data:
                action_taken = self._execute_buy(symbol, current_price, action_data, timestamp)
            elif decision == 'sell' and action_data:
                action_taken = self._execute_sell(symbol, current_price, action_data, timestamp)
            
            # Update cooldown if action was taken
            if action_taken:
                self.last_action_time[symbol] = timestamp
            
            return action_taken
            
        except Exception as e:
            logger.error(f"[LIVE] Failed to process {symbol}: {e}")
            return False
    
    def _execute_buy(self, symbol: str, price: float, action_data: dict, timestamp: pd.Timestamp) -> bool:
        """Execute a buy order using Alpaca API."""
        try:
            # Handle different key names from dual model engine
            quantity = action_data.get('shares', action_data.get('quantity', 0))
            
            if quantity <= 0:
                return False
            
            if self.paper_trading:
                # Paper trading - use our portfolio system
                success = self.portfolio.buy(symbol, price, quantity, timestamp)
                if success:
                    logger.info(f"[LIVE] 📝 PAPER BUY: {quantity} {symbol} @ ${price:.2f} (DualModel)")
                return success
            else:
                # Live trading - use Alpaca API
                success, message, order_id = self.alpaca.submit_buy_order(symbol, quantity)
                
                if success:
                    logger.info(f"[LIVE] 🚀 LIVE BUY: {message} (Order ID: {order_id}) (DualModel)")
                    # Also update our local portfolio for tracking
                    self.portfolio.buy(symbol, price, quantity, timestamp)
                    return True
                else:
                    logger.error(f"[LIVE] BUY FAILED: {message}")
                    return False
            
        except Exception as e:
            logger.error(f"[LIVE] Buy execution failed for {symbol}: {e}")
            return False
    
    def _execute_sell(self, symbol: str, price: float, action_data: dict, timestamp: pd.Timestamp) -> bool:
        """Execute a sell order using Alpaca API."""
        try:
            # Handle different key names from dual model engine
            quantity = action_data.get('shares', action_data.get('quantity'))
            force_sell = action_data.get('force_sell', False)
            
            if self.paper_trading:
                # Paper trading - use our portfolio system
                success, pnl, reason = self.portfolio.sell(symbol, price, quantity, force_sell, timestamp)
                if success:
                    logger.info(f"[LIVE] 📝 PAPER SELL: {symbol} @ ${price:.2f} | P&L: ${pnl:.2f} (DualModel)")
                return success
            else:
                # Live trading - use Alpaca API
                success, message, order_id = self.alpaca.submit_sell_order(symbol, quantity)
                
                if success:
                    logger.info(f"[LIVE] 🚀 LIVE SELL: {message} (Order ID: {order_id}) (DualModel)")
                    # Also update our local portfolio for tracking
                    self.portfolio.sell(symbol, price, quantity, force_sell, timestamp)
                    return True
                else:
                    logger.error(f"[LIVE] SELL FAILED: {message}")
                    return False
            
        except Exception as e:
            logger.error(f"[LIVE] Sell execution failed for {symbol}: {e}")
            return False
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.last_action_time:
            return False
        
        last_time = self.last_action_time[symbol]
        cooldown_duration = timedelta(minutes=COOLDOWN_MINUTES)
        
        return (pd.Timestamp.now() - last_time) < cooldown_duration
    
    def _print_portfolio_status(self):
        """Print current portfolio status."""
        try:
            account_info = self.alpaca.get_account_info()
            
            print(f"\n{'='*50}")
            print(f"PORTFOLIO STATUS (DUAL MODEL ENGINE) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            print(f"Cash: ${account_info.get('cash', 0):,.2f}")
            print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            
            # Show positions
            positions = self.alpaca.get_all_positions()
            if positions:
                print(f"\nActive Positions ({len(positions)}) - DualModel Engine:")
                for pos in positions:
                    pnl_pct = pos['unrealized_plpc'] * 100
                    print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f} | "
                          f"P&L: ${pos['unrealized_pl']:+.2f} ({pnl_pct:+.1f}%)")
            else:
                print("\nNo active positions")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"[LIVE] Failed to print portfolio status: {e}")
    
    def _print_final_summary(self):
        """Print final trading session summary."""
        try:
            account_info = self.alpaca.get_account_info()
            
            print(f"\n{'='*60}")
            print(f"FINAL TRADING SESSION SUMMARY (DUAL MODEL ENGINE)")
            print(f"{'='*60}")
            print(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'} Trading")
            print(f"Engine: DualModelTradingEngine (FIXED)")
            print(f"Final Cash: ${account_info.get('cash', 0):,.2f}")
            print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            
            # Portfolio history
            try:
                history = self.alpaca.get_portfolio_history('1D')
                if history and history['profit_loss_pct']:
                    daily_return = history['profit_loss_pct'][-1] * 100
                    print(f"Daily Return: {daily_return:+.2f}%")
            except:
                pass
            
            positions = self.alpaca.get_all_positions()
            if positions:
                total_unrealized = sum(pos['unrealized_pl'] for pos in positions)
                print(f"Unrealized P&L: ${total_unrealized:+.2f}")
                print(f"Active Positions: {len(positions)}")
                
                for pos in positions:
                    pnl_pct = pos['unrealized_plpc'] * 100
                    print(f"  {pos['symbol']}: ${pos['unrealized_pl']:+.2f} ({pnl_pct:+.1f}%)")
            
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"[LIVE] Failed to print final summary: {e}")
    
    def get_portfolio_status(self) -> dict:
        """Get current portfolio status."""
        try:
            account_info = self.alpaca.get_account_info()
            positions = self.alpaca.get_all_positions()
            
            return {
                'account_info': account_info,
                'positions': positions,
                'mode': 'paper' if self.paper_trading else 'live',
                'engine': 'DualModelTradingEngine'
            }
        except Exception as e:
            logger.error(f"[LIVE] Failed to get portfolio status: {e}")
            return {}

def main():
    """Main entry point for live trading."""
    import sys
    
    # Parse command line arguments
    paper_trading = '--real' not in sys.argv
    
    if paper_trading:
        print("🚀 Starting LAEF Live Trader (PAPER TRADING MODE)")
        print("   - Using Alpaca Paper Trading API")
        print("   - DualModelTradingEngine (FIXED)")
        print("   - No real money at risk")
    else:
        print("⚠️  Starting LAEF Live Trader (LIVE TRADING MODE)")
        print("   - Using Alpaca Live Trading API")
        print("   - DualModelTradingEngine (FIXED)")
        print("   - REAL MONEY WILL BE USED")
        confirm = input("Are you sure you want to trade with real money? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Switching to paper trading mode for safety.")
            paper_trading = True
    
    try:
        # Create and start trader
        trader = LAEFLiveTrader(paper_trading=paper_trading)
        trader.start_trading()
    except KeyboardInterrupt:
        print("\n🛑 Trading interrupted by user")
        if 'trader' in locals():
            trader.stop_trading()
    except Exception as e:
        print(f"❌ Trading failed: {e}")
        logger.error(f"Trading failed: {e}")

if __name__ == "__main__":
    main()
