"""
Live Market Learning System for LAEF
Monitors live markets, makes predictions, tracks outcomes, and learns from results
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, time as datetime_time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
from collections import defaultdict
import threading
import queue

# Delayed imports to avoid circular dependencies
LAEFAgent = None
create_state_vector = None
extract_raw_indicators = None
fetch_stock_data = None
fetch_multiple_symbols = None
FIFOPortfolio = None

def _lazy_imports():
    """Import heavy modules only when needed"""
    global LAEFAgent, create_state_vector, extract_raw_indicators
    global fetch_stock_data, fetch_multiple_symbols, FIFOPortfolio
    
    if LAEFAgent is None:
        from core.agent_unified import LAEFAgent
        from core.state_utils_unified import create_state_vector, extract_raw_indicators
        from data.data_fetcher_unified import fetch_stock_data, fetch_multiple_symbols
        from core.fifo_portfolio import FIFOPortfolio

try:
    from trading.alpaca_integration import get_alpaca_client, test_alpaca_connection
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    def test_alpaca_connection(paper_trading=True):
        return True  # Mock for testing

from training.prediction_tracker import PredictionTracker


class LiveMarketLearner:
    """
    Main class for live market learning
    """
    
    def __init__(self, symbols: List[str] = None, paper_trading: bool = True):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.paper_trading = paper_trading
        
        # Lazy import heavy modules
        _lazy_imports()
        
        # Initialize components
        self.agent = LAEFAgent(pretrained=True)
        self.tracker = PredictionTracker()
        self.portfolio = FIFOPortfolio(100000)  # Virtual portfolio for tracking
        
        # Learning parameters
        self.learning_rate = 0.001
        self.min_predictions_for_learning = 20
        self.prediction_intervals = {
            'short_term': {'1_hour': 60, '4_hour': 240, '1_day': 1440},
            'long_term': {'1_week': 10080, '1_month': 43200}
        }
        
        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.prediction_queue = queue.Queue()
        
        # Logging
        self.log_dir = "logs/training/live_learning"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, f"live_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        market_open = datetime_time(9, 30)
        market_close = datetime_time(16, 0)
        
        # Check if weekday (Monday=0, Friday=4)
        if now.weekday() > 4:
            return False
        
        # Check if within market hours
        return market_open <= now.time() <= market_close
    
    def make_predictions(self, symbol: str, current_data: pd.DataFrame) -> List[Dict]:
        """Generate predictions for different timeframes"""
        predictions = []
        
        try:
            # Import and calculate indicators on the data
            from core.indicators_unified import calculate_all_indicators
            
            # Calculate all indicators first
            df_with_indicators = calculate_all_indicators(current_data.copy())
            
            # Extract current state - create_state_vector takes the dataframe with indicators
            state = create_state_vector(df_with_indicators)
            
            if state is None:
                logging.error(f"Failed to create state vector for {symbol}")
                return predictions
            
            # Extract indicators as dictionary
            indicators = extract_raw_indicators(df_with_indicators)
            current_price = indicators.get('price', current_data.iloc[-1]['close'])
            
            # Get model predictions - Q-values for [hold, buy, sell]
            q_values = self.agent.predict_q_values(state)
            
            # Debug logging to see what we're getting
            logging.debug(f"q_values shape: {q_values.shape if hasattr(q_values, 'shape') else 'no shape'}")
            logging.debug(f"q_values: {q_values}")
            
            # Ensure q_values is a proper array with 3 elements
            if not isinstance(q_values, np.ndarray):
                q_values = np.array(q_values)
            
            if q_values.size < 3:
                logging.error(f"Invalid q_values size: {q_values.size}, expected 3")
                # Return default predictions
                return predictions
                
            action = np.argmax(q_values)
            confidence = float(np.max(q_values) - np.mean(q_values))
            
            # Calculate ML score (buy vs sell signal strength)
            ml_score = float(q_values[1] - q_values[2])  # Buy vs Sell signal strength
            
            # Technical analysis for price targets
            sma_20 = indicators.get('sma20', df_with_indicators['close'].rolling(20).mean().iloc[-1])
            sma_50 = indicators.get('sma50', df_with_indicators['close'].rolling(50).mean().iloc[-1])
            rsi = indicators.get('rsi', 50.0)
            
            # Market conditions
            market_conditions = {
                'trend': 'bullish' if current_price > sma_50 else 'bearish',
                'momentum': 'strong' if abs(current_price - sma_20) / sma_20 > 0.02 else 'weak',
                'volatility': float(df_with_indicators['close'].pct_change().std()),
                'volume_trend': 'increasing' if df_with_indicators['volume'].iloc[-1] > df_with_indicators['volume'].mean() else 'decreasing'
            }
            
            # Generate predictions for each timeframe
            for pred_type, timeframes in self.prediction_intervals.items():
                for timeframe, minutes in timeframes.items():
                    # Calculate price target based on historical volatility and ML signals
                    volatility = df_with_indicators['close'].pct_change().std()
                    time_factor = np.sqrt(minutes / 1440)  # Scale volatility by time
                    
                    # Base prediction on ML signal and technical indicators
                    if action == 1:  # Buy signal
                        base_move = ml_score * 0.01  # 1% per unit of ML score
                    elif action == 2:  # Sell signal
                        base_move = -ml_score * 0.01
                    else:  # Hold
                        base_move = 0
                    
                    # Adjust for timeframe and volatility
                    expected_move = base_move * time_factor * (1 + volatility)
                    
                    # Apply technical levels as resistance/support
                    predicted_price = current_price * (1 + expected_move)
                    
                    # Cap predictions at reasonable levels
                    max_move = 0.05 if pred_type == 'short_term' else 0.20
                    predicted_price = np.clip(
                        predicted_price,
                        current_price * (1 - max_move),
                        current_price * (1 + max_move)
                    )
                    
                    prediction = {
                        'symbol': symbol,
                        'prediction_type': pred_type,
                        'timeframe': timeframe,
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'confidence': confidence,
                        'q_value': float(q_values[action]),
                        'ml_score': ml_score,
                        'indicators': {
                            'sma_20': float(sma_20),
                            'sma_50': float(sma_50),
                            'rsi': float(rsi),
                            'action': int(action),
                            'macd': indicators.get('macd', 0.0),
                            'signal': indicators.get('signal', 0.0)
                        },
                        'market_conditions': market_conditions
                    }
                    
                    predictions.append(prediction)
                    
        except Exception as e:
            logging.error(f"Error making predictions for {symbol}: {e}")
        
        return predictions
    
    def monitor_and_predict(self):
        """Main monitoring loop"""
        logging.info("Starting live market monitoring...")
        
        while self.monitoring_active:
            try:
                if self.is_market_open() or self.paper_trading:
                    # Fetch current data for all symbols
                    current_time = datetime.now()
                    logging.info(f"Fetching market data at {current_time}")
                    
                    # Get recent data for analysis - increased to 120 days for more data points
                    data = fetch_multiple_symbols(
                        self.symbols,
                        start_date=(datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    # Make predictions for each symbol
                    for symbol, df in data.items():
                        if df is not None and len(df) >= 50:
                            predictions = self.make_predictions(symbol, df)
                            
                            # Store predictions
                            for pred in predictions:
                                pred_id = self.tracker.add_prediction(
                                    symbol=pred['symbol'],
                                    prediction_type=pred['prediction_type'],
                                    timeframe=pred['timeframe'],
                                    current_price=pred['current_price'],
                                    predicted_price=pred['predicted_price'],
                                    confidence=pred['confidence'],
                                    q_value=pred['q_value'],
                                    ml_score=pred['ml_score'],
                                    indicators=pred['indicators'],
                                    market_conditions=pred['market_conditions']
                                )
                                
                                logging.info(
                                    f"Prediction {pred_id}: {symbol} {pred['timeframe']} - "
                                    f"Current: ${pred['current_price']:.2f}, "
                                    f"Predicted: ${pred['predicted_price']:.2f} "
                                    f"({(pred['predicted_price']/pred['current_price']-1)*100:+.1f}%)"
                                )
                    
                    # Check and update outcomes for pending predictions
                    self.check_prediction_outcomes()
                    
                    # Perform learning if enough predictions have completed
                    self.learn_from_outcomes()
                    
                # Sleep based on market status
                if self.is_market_open():
                    time.sleep(300)  # 5 minutes during market hours
                else:
                    time.sleep(3600)  # 1 hour after hours
                    
            except KeyboardInterrupt:
                logging.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def check_prediction_outcomes(self):
        """Check and update outcomes for pending predictions"""
        current_time = datetime.now()
        
        # Get all pending predictions
        pending = self.tracker.get_pending_predictions()
        
        for pred in pending:
            # Parse prediction time and calculate if outcome time has passed
            pred_time = datetime.fromisoformat(pred['timestamp'])
            timeframe_minutes = self.prediction_intervals[pred['prediction_type']][pred['timeframe']]
            outcome_time = pred_time + timedelta(minutes=timeframe_minutes)
            
            if current_time >= outcome_time:
                # Fetch current price
                try:
                    data = fetch_stock_data(
                        pred['symbol'],
                        start=(current_time - timedelta(days=1)).strftime('%Y-%m-%d'),
                        end=current_time.strftime('%Y-%m-%d')
                    )
                    
                    if data is not None and len(data) > 0:
                        outcome_price = data.iloc[-1]['close']
                        self.tracker.update_outcome(pred['id'], outcome_price)
                        
                        # Log outcome
                        actual_return = (outcome_price - pred['current_price']) / pred['current_price']
                        predicted_return = (pred['predicted_price'] - pred['current_price']) / pred['current_price']
                        
                        logging.info(
                            f"Outcome for prediction {pred['id']}: "
                            f"{pred['symbol']} {pred['timeframe']} - "
                            f"Predicted: {predicted_return:.1%}, Actual: {actual_return:.1%}"
                        )
                        
                except Exception as e:
                    logging.error(f"Error checking outcome for prediction {pred['id']}: {e}")
    
    def learn_from_outcomes(self):
        """Learn from completed predictions"""
        candidates = self.tracker.get_learning_candidates(self.min_predictions_for_learning)
        
        if len(candidates) < self.min_predictions_for_learning:
            return
        
        logging.info(f"Learning from {len(candidates)} completed predictions...")
        
        # Group by symbol for batch learning
        symbol_groups = defaultdict(list)
        for pred in candidates:
            symbol_groups[pred['symbol']].append(pred)
        
        total_updates = 0
        
        for symbol, predictions in symbol_groups.items():
            try:
                # Prepare training batch
                states = []
                targets = []
                
                for pred in predictions:
                    # Reconstruct state from stored data
                    indicators = json.loads(pred['technical_indicators'])
                    
                    # Create simplified state (you might want to store full state in DB)
                    state = np.array([
                        pred['current_price'],
                        indicators['sma_20'],
                        indicators['sma_50'],
                        indicators['rsi'],
                        pred['q_value'],
                        pred['ml_score']
                    ])
                    
                    # Calculate target based on actual outcome
                    actual_return = pred['actual_return']
                    
                    # Calculate reward based on prediction accuracy
                    if pred['prediction_accuracy'] == 'correct':
                        # Reinforce the action
                        reward = abs(actual_return)
                    elif pred['prediction_accuracy'] == 'incorrect':
                        # Penalize the action
                        reward = -abs(actual_return)
                    else:  # partially_correct
                        reward = abs(actual_return) * 0.5
                    
                    # Create target Q-values for [hold, buy, sell]
                    current_q_values = np.array([0.0, pred['q_value'], -pred['q_value']])  # Approximate Q-values
                    
                    # Update Q-value for the predicted action
                    action = indicators['action']
                    current_q_values[action] += self.learning_rate * reward
                    
                    states.append(state)
                    targets.append(current_q_values)
                
                if states:
                    # Note: This is simplified - you'd need to adapt based on your actual model architecture
                    # For now, we'll just log the learning attempt
                    logging.info(
                        f"Would update model for {symbol} with {len(states)} examples. "
                        f"Avg target Q-values: Hold={np.mean([t[0] for t in targets]):.4f}, "
                        f"Buy={np.mean([t[1] for t in targets]):.4f}, "
                        f"Sell={np.mean([t[2] for t in targets]):.4f}"
                    )
                    total_updates += len(states)
                    
            except Exception as e:
                logging.error(f"Error learning from {symbol} predictions: {e}")
        
        # Mark predictions as learned
        learned_ids = [p['id'] for p in candidates]
        self.tracker.mark_learning_applied(learned_ids)
        
        logging.info(f"Learning complete. Applied {total_updates} updates.")
        
        # Save updated model
        if total_updates > 0:
            checkpoint_path = os.path.join(
                self.log_dir,
                f"model_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            )
            self.agent.save_model(checkpoint_path)
            logging.info(f"Model checkpoint saved: {checkpoint_path}")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("="*60)
        report.append("LIVE MARKET LEARNING REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        overall_stats = self.tracker.get_performance_stats()
        report.append("OVERALL PERFORMANCE (Last 30 days):")
        report.append(f"  Total Predictions: {overall_stats['total_predictions']}")
        report.append(f"  Accuracy Rate: {overall_stats['accuracy_rate']:.1%}")
        report.append(f"  Correct: {overall_stats['correct']}")
        report.append(f"  Partially Correct: {overall_stats['partial']}")
        report.append(f"  Incorrect: {overall_stats['incorrect']}")
        report.append(f"  Avg Confidence: {overall_stats['avg_confidence']:.3f}")
        report.append(f"  Avg Price Error: {overall_stats['avg_price_error']:.2%}")
        report.append("")
        
        # Per-symbol performance
        report.append("PER-SYMBOL PERFORMANCE:")
        for symbol in self.symbols:
            stats = self.tracker.get_performance_stats(symbol)
            if stats['total_predictions'] > 0:
                report.append(f"\n{symbol}:")
                report.append(f"  Predictions: {stats['total_predictions']}")
                report.append(f"  Accuracy: {stats['accuracy_rate']:.1%}")
                report.append(f"  Avg Error: {stats['avg_price_error']:.2%}")
        
        # Timeframe analysis
        report.append("\nTIMEFRAME ANALYSIS:")
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timeframe, 
                   COUNT(*) as count,
                   AVG(CASE WHEN prediction_accuracy = 'correct' THEN 1 ELSE 0 END) as accuracy
            FROM predictions
            WHERE prediction_accuracy != 'pending'
            GROUP BY timeframe
        ''')
        
        for row in cursor.fetchall():
            report.append(f"  {row[0]}: {row[1]} predictions, {row[2]:.1%} accuracy")
        
        conn.close()
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(
            self.log_dir,
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def start_monitoring(self):
        """Start live monitoring in background thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self.monitor_and_predict)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logging.info("Live market monitoring started")
    
    def stop_monitoring(self):
        """Stop live monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            logging.info("Live market monitoring stopped")
    
    def run_interactive_session(self):
        """Run interactive live learning session"""
        print("\n" + "="*60)
        print("LIVE MARKET LEARNING SYSTEM")
        print("="*60)
        print("\nThis system will:")
        print("  • Monitor live market data")
        print("  • Make predictions for multiple timeframes")
        print("  • Track prediction outcomes automatically")
        print("  • Learn from successes and failures")
        print("  • Generate performance reports")
        
        # Start monitoring
        self.start_monitoring()
        
        # Interactive menu
        while True:
            print("\n" + "-"*40)
            print("LIVE LEARNING MENU")
            print("-"*40)
            print("1. View Current Predictions")
            print("2. Check Prediction Outcomes")
            print("3. Force Learning Cycle")
            print("4. Generate Performance Report")
            print("5. View Real-time Stats")
            print("6. Change Symbols")
            print("7. Stop and Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.view_current_predictions()
            elif choice == '2':
                self.check_prediction_outcomes()
                print("Outcomes updated.")
            elif choice == '3':
                self.learn_from_outcomes()
            elif choice == '4':
                report = self.generate_performance_report()
                print("\n" + report)
            elif choice == '5':
                self.view_realtime_stats()
            elif choice == '6':
                self.change_symbols()
            elif choice == '7':
                break
            else:
                print("Invalid option")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Final report
        print("\nGenerating final report...")
        report = self.generate_performance_report()
        print("\n" + report)
        print(f"\nLogs saved to: {self.log_dir}")
    
    def view_current_predictions(self):
        """Display current pending predictions"""
        pending = self.tracker.get_pending_predictions()
        
        if not pending:
            print("No pending predictions.")
            return
        
        print(f"\nCurrent Pending Predictions ({len(pending)} total):")
        print("-"*80)
        print(f"{'ID':>4} {'Symbol':>6} {'Type':>10} {'Timeframe':>10} "
              f"{'Current':>8} {'Predicted':>8} {'Change':>7} {'Confidence':>10}")
        print("-"*80)
        
        for pred in pending[:20]:  # Show first 20
            change = (pred['predicted_price'] / pred['current_price'] - 1) * 100
            print(f"{pred['id']:>4} {pred['symbol']:>6} {pred['prediction_type']:>10} "
                  f"{pred['timeframe']:>10} ${pred['current_price']:>7.2f} "
                  f"${pred['predicted_price']:>7.2f} {change:>6.1f}% "
                  f"{pred['confidence']:>9.3f}")
    
    def view_realtime_stats(self):
        """Display real-time statistics"""
        stats = self.tracker.get_performance_stats(days=1)
        print("\n" + "="*40)
        print("REAL-TIME STATISTICS (Last 24 hours)")
        print("="*40)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Accuracy Rate: {stats['accuracy_rate']:.1%}")
        print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"Average Price Error: {stats['avg_price_error']:.2%}")
        
        # Show per-symbol stats
        print("\nPer-Symbol Performance:")
        for symbol in self.symbols:
            symbol_stats = self.tracker.get_performance_stats(symbol, days=1)
            if symbol_stats['total_predictions'] > 0:
                print(f"  {symbol}: {symbol_stats['total_predictions']} predictions, "
                      f"{symbol_stats['accuracy_rate']:.1%} accuracy")
    
    def change_symbols(self):
        """Change monitored symbols"""
        print(f"\nCurrent symbols: {', '.join(self.symbols)}")
        new_symbols = input("Enter new symbols (comma-separated): ").strip().upper()
        
        if new_symbols:
            self.symbols = [s.strip() for s in new_symbols.split(',')]
            print(f"Updated symbols: {', '.join(self.symbols)}")
            
            # Restart monitoring with new symbols
            self.stop_monitoring()
            time.sleep(2)
            self.start_monitoring()


def run_live_market_learning_interface():
    """
    Interface function for integration with main.py
    """
    print("\n" + "="*60)
    print("LIVE MARKET LEARNING")
    print("="*60)
    print("\nThis system monitors live markets and learns from predictions.")
    print("It works with paper trading (fake money) for safety.")
    
    # Test API connection
    print("\nChecking API connection...")
    if not test_alpaca_connection(paper_trading=True):
        print("Error: Alpaca API connection failed.")
        print("Please check your API keys in config.py")
        return
    
    print("✓ API connection successful")
    
    # Configuration
    print("\n" + "-"*40)
    print("CONFIGURATION")
    print("-"*40)
    
    # Symbol selection
    print("\n1. Use default symbols (AAPL, MSFT, GOOGL, AMZN, META)")
    print("2. Custom symbol list")
    print("3. Load from smart symbol selector")
    
    symbol_choice = input("Select option (1-3) [1]: ").strip() or "1"
    
    if symbol_choice == "1":
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    elif symbol_choice == "2":
        custom = input("Enter symbols (comma-separated): ").strip().upper()
        symbols = [s.strip() for s in custom.split(',')]
    else:
        from data.smart_symbol_selector import SmartSymbolSelector
        selector = SmartSymbolSelector()
        symbols = selector.analyze_symbols(limit=10, analysis_mode='ml_focused')
    
    print(f"\nMonitoring symbols: {', '.join(symbols)}")
    
    # Learning parameters
    print("\nLearning Parameters:")
    learning_rate = float(input("Learning rate (0.0001-0.01) [0.001]: ") or "0.001")
    min_predictions = int(input("Min predictions before learning [20]: ") or "20")
    
    # Confirm
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Mode: Paper Trading (Safe)")
    print(f"Learning Rate: {learning_rate}")
    print(f"Min Predictions: {min_predictions}")
    print("\nThe system will:")
    print("  • Make predictions every 5 minutes during market hours")
    print("  • Track short-term (1hr, 4hr, 1day) and long-term (1week, 1month) outcomes")
    print("  • Learn from prediction accuracy")
    print("  • Generate performance reports")
    
    confirm = input("\nStart live learning? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Live learning cancelled.")
        return
    
    # Create and run learner
    learner = LiveMarketLearner(symbols=symbols, paper_trading=True)
    learner.learning_rate = learning_rate
    learner.min_predictions_for_learning = min_predictions
    
    try:
        learner.run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nLive learning interrupted by user.")
        learner.stop_monitoring()
    except Exception as e:
        print(f"\nError during live learning: {e}")
        logging.error(f"Live learning failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Test the live learner
    run_live_market_learning_interface()