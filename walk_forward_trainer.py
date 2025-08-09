"""
Walk-Forward Training System for LAEF
Implements best practices for training Q-learning agents on financial data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Tuple, Any
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features disabled.")
from collections import defaultdict
import time

from core.agent_unified import LAEFAgent
from core.state_utils_unified import create_state_vector, extract_raw_indicators
from data.data_fetcher_unified import fetch_stock_data, fetch_multiple_symbols
from core.fifo_portfolio import FIFOPortfolio
from core.hybrid_trading_engine import create_hybrid_trading_engine


class WalkForwardTrainer:
    """
    Implements Walk-Forward Analysis for training LAEF Q-learning agent
    with visualization and progress tracking
    """
    
    def __init__(self, 
                 train_window_days: int = 252,  # 1 year
                 test_window_days: int = 63,    # 3 months
                 step_days: int = 21,           # 1 month step
                 initial_cash: float = 100000):
        """
        Initialize Walk-Forward Trainer
        
        Args:
            train_window_days: Number of days for training window
            test_window_days: Number of days for testing window
            step_days: Number of days to step forward each iteration
            initial_cash: Starting capital for each training episode
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.initial_cash = initial_cash
        
        # Training components
        self.agent = LAEFAgent(pretrained=True)
        self.training_history = []
        self.weight_history = defaultdict(list)
        self.performance_metrics = []
        
        # Initialize experience buffer for training
        self.experience_buffer = []  # Simple list-based buffer
        self.max_buffer_size = 10000
        
        # Create training logs directory
        self.log_dir = "logs/training/walk_forward"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, f"wf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def prepare_data_windows(self, 
                           symbols: List[str], 
                           start_date: str, 
                           end_date: str) -> List[Dict]:
        """
        Prepare walk-forward data windows
        
        Returns list of dictionaries containing train/test date ranges
        """
        windows = []
        current_start = pd.to_datetime(start_date)
        final_end = pd.to_datetime(end_date)
        
        while True:
            train_end = current_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)
            
            if test_end > final_end:
                break
                
            windows.append({
                'train_start': current_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'window_id': len(windows) + 1
            })
            
            current_start += timedelta(days=self.step_days)
            
        return windows
    
    def train_on_window(self, 
                       symbols: List[str], 
                       window: Dict,
                       learning_rate: float = 0.001,
                       episodes: int = 5) -> Dict:
        """
        Train agent on a single window of data
        """
        print(f"\n{'='*60}")
        print(f"Training Window {window['window_id']}")
        print(f"Train: {window['train_start']} to {window['train_end']}")
        print(f"Test: {window['test_start']} to {window['test_end']}")
        print(f"{'='*60}")
        
        # Training phase
        train_metrics = {
            'total_rewards': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'q_value_changes': []
        }
        
        # Fetch all training data upfront
        print("Fetching training data...")
        all_train_data = fetch_multiple_symbols(
            symbols, 
            start_date=window['train_start'], 
            end_date=window['train_end']
        )
        
        valid_symbols = [s for s, df in all_train_data.items() if df is not None and len(df) >= 50]
        if not valid_symbols:
            logging.error("No valid symbols with sufficient data")
            return {}
        
        print(f"Training on {len(valid_symbols)} symbols with sufficient data")
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            episode_start = time.time()
            
            # Reset portfolio for each episode
            portfolio = FIFOPortfolio(self.initial_cash)
            
            # Track Q-values before training
            initial_q_values = self._sample_q_values(valid_symbols[:5])
            
            # Train on each symbol
            for symbol in valid_symbols:
                try:
                    # Get pre-fetched data
                    train_data = all_train_data[symbol]
                    
                    # Extract indicators
                    indicators = extract_raw_indicators(train_data)
                    
                    # Simulate trading and learning
                    rewards = self._train_on_symbol(
                        symbol, 
                        train_data, 
                        indicators, 
                        portfolio,
                        learning_rate
                    )
                    
                    if rewards:
                        train_metrics['total_rewards'].extend(rewards)
                        
                except Exception as e:
                    logging.error(f"Error training on {symbol}: {e}")
                    continue
            
            # Track Q-values after training
            final_q_values = self._sample_q_values(symbols[:5])
            q_change = self._calculate_q_value_change(initial_q_values, final_q_values)
            train_metrics['q_value_changes'].append(q_change)
            
            # Calculate episode metrics
            if train_metrics['total_rewards']:
                win_rate = sum(1 for r in train_metrics['total_rewards'] if r > 0) / len(train_metrics['total_rewards'])
                train_metrics['win_rates'].append(win_rate)
                
                # Simple Sharpe ratio approximation
                returns = np.array(train_metrics['total_rewards'])
                if len(returns) > 1 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                    train_metrics['sharpe_ratios'].append(sharpe)
            
            print(f"Episode completed in {time.time() - episode_start:.1f}s")
            print(f"Q-value change: {q_change:.4f}")
            
            # Update progress bar
            self._display_progress(episode + 1, episodes, prefix='Training Progress:')
        
        # Testing phase
        print(f"\n{'Testing Phase':^60}")
        test_results = self._test_on_window(valid_symbols, window)
        
        # Save window results
        window_results = {
            'window': window,
            'train_metrics': train_metrics,
            'test_results': test_results,
            'agent_weights': self._get_current_weights()
        }
        
        self._save_window_results(window_results)
        
        return window_results
    
    def _train_on_symbol(self, 
                        symbol: str, 
                        data: pd.DataFrame, 
                        indicators: np.ndarray,
                        portfolio: FIFOPortfolio,
                        learning_rate: float) -> List[float]:
        """
        Train agent on single symbol data
        """
        rewards = []
        
        for i in range(20, len(data) - 1):
            # Create state vector
            state = create_state_vector(
                indicators[i],
                data.iloc[i]['close'],
                portfolio.cash,
                portfolio.get_position_value(symbol, data.iloc[i]['close'])
            )
            
            # Get action from agent
            action = self.agent.get_action(state, epsilon=0.1)  # Exploration
            
            # Calculate reward based on next price movement
            current_price = data.iloc[i]['close']
            next_price = data.iloc[i + 1]['close']
            
            if action == 1:  # Buy
                reward = (next_price - current_price) / current_price
            elif action == 2:  # Sell
                reward = (current_price - next_price) / current_price
            else:  # Hold
                reward = 0
            
            # Create next state
            next_state = create_state_vector(
                indicators[i + 1],
                next_price,
                portfolio.cash,
                portfolio.get_position_value(symbol, next_price)
            )
            
            # Store experience
            if len(self.experience_buffer) >= self.max_buffer_size:
                self.experience_buffer.pop(0)  # Remove oldest
            self.experience_buffer.append((state, action, reward, next_state))
            rewards.append(reward)
            
            # Periodic training on experience batch
            if i % 32 == 0 and len(self.experience_buffer) >= 32:
                # Sample batch of experiences
                import random
                batch = random.sample(self.experience_buffer, 32)
                
                # Prepare training data
                states = np.array([exp[0] for exp in batch])
                actions = np.array([exp[1] for exp in batch])
                rewards_batch = np.array([exp[2] for exp in batch])
                next_states = np.array([exp[3] for exp in batch])
                
                # Calculate target Q-values
                current_q_values = self.agent.model.predict(states, verbose=0)
                next_q_values = self.agent.model.predict(next_states, verbose=0)
                
                # Update Q-values using Bellman equation
                for idx, (action, reward) in enumerate(zip(actions, rewards_batch)):
                    target = reward + 0.95 * np.max(next_q_values[idx])  # Gamma = 0.95
                    current_q_values[idx][action] = current_q_values[idx][action] + learning_rate * (target - current_q_values[idx][action])
                
                # Train model on batch
                self.agent.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        return rewards
    
    def _test_on_window(self, symbols: List[str], window: Dict) -> Dict:
        """
        Test trained agent on test window
        """
        portfolio = FIFOPortfolio(self.initial_cash)
        trading_engine = create_hybrid_trading_engine(portfolio)
        
        total_trades = 0
        winning_trades = 0
        total_return = 0
        
        # Fetch all test data upfront
        print("Fetching test data...")
        all_test_data = fetch_multiple_symbols(
            symbols,
            start_date=window['test_start'],
            end_date=window['test_end']
        )
        
        for symbol, test_data in all_test_data.items():
            try:
                if test_data is None or len(test_data) < 20:
                    continue
                
                initial_value = portfolio.get_total_value(test_data.iloc[0]['close'])
                
                # Simulate trading
                for i in range(20, len(test_data)):
                    indicators = extract_raw_indicators(test_data[:i+1])
                    state = create_state_vector(
                        indicators[-1],
                        test_data.iloc[i]['close'],
                        portfolio.cash,
                        portfolio.get_position_value(symbol, test_data.iloc[i]['close'])
                    )
                    
                    # Get action (no exploration during testing)
                    action = self.agent.get_action(state, epsilon=0)
                    
                    # Execute trade if action suggests
                    if action == 1 and portfolio.cash > 1000:  # Buy
                        shares = int(portfolio.cash * 0.95 / test_data.iloc[i]['close'])
                        if shares > 0:
                            portfolio.buy(symbol, shares, test_data.iloc[i]['close'])
                            total_trades += 1
                    elif action == 2:  # Sell
                        position = portfolio.get_position(symbol)
                        if position and position['shares'] > 0:
                            sell_result = portfolio.sell(
                                symbol, 
                                position['shares'], 
                                test_data.iloc[i]['close']
                            )
                            if sell_result['profit'] > 0:
                                winning_trades += 1
                            total_trades += 1
                
                final_value = portfolio.get_total_value(test_data.iloc[-1]['close'])
                symbol_return = (final_value - initial_value) / initial_value
                total_return += symbol_return
                
            except Exception as e:
                logging.error(f"Error testing {symbol}: {e}")
                continue
        
        return {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_return': total_return / len(symbols) if symbols else 0,
            'portfolio_value': portfolio.get_total_value(100)  # Dummy price
        }
    
    def _sample_q_values(self, symbols: List[str]) -> np.ndarray:
        """
        Sample Q-values for tracking learning progress
        """
        q_samples = []
        
        for symbol in symbols:
            try:
                # Get recent data
                data = fetch_stock_data(
                    symbol,
                    start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d')
                )
                
                if data is not None and len(data) > 20:
                    indicators = extract_raw_indicators(data)
                    state = create_state_vector(
                        indicators[-1],
                        data.iloc[-1]['close'],
                        50000,  # Dummy cash
                        0       # No position
                    )
                    
                    # Get Q-values predictions from neural network
                    q_values = self.agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                    q_samples.extend(q_values)
                    
            except:
                continue
        
        return np.array(q_samples) if q_samples else np.array([0])
    
    def _calculate_q_value_change(self, initial: np.ndarray, final: np.ndarray) -> float:
        """
        Calculate average change in Q-values
        """
        if len(initial) == len(final) and len(initial) > 0:
            return np.mean(np.abs(final - initial))
        return 0.0
    
    def _get_current_weights(self) -> Dict:
        """
        Extract current model weights for visualization
        """
        # For neural network, track weight statistics
        weights = {
            'total_parameters': self.agent.model.count_params(),
            'experience_buffer_size': len(self.experience_buffer),
            'avg_weight_magnitude': 0,
            'max_weight_magnitude': 0,
            'min_weight_magnitude': 0
        }
        
        # Calculate weight statistics
        all_weights = []
        for layer in self.agent.model.layers:
            if layer.get_weights():
                layer_weights = layer.get_weights()[0].flatten()
                all_weights.extend(layer_weights)
        
        if all_weights:
            weights['avg_weight_magnitude'] = float(np.mean(np.abs(all_weights)))
            weights['max_weight_magnitude'] = float(np.max(np.abs(all_weights)))
            weights['min_weight_magnitude'] = float(np.min(np.abs(all_weights)))
        
        return weights
    
    def _display_progress(self, current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
        """
        Display training progress bar
        """
        percent = current / total
        filled = int(length * percent)
        bar = '█' * filled + '-' * (length - filled)
        print(f'\r{prefix} |{bar}| {percent:.1%} {suffix}', end='', flush=True)
        if current == total:
            print()
    
    def _save_window_results(self, results: Dict):
        """
        Save results for a training window
        """
        window_id = results['window']['window_id']
        filename = os.path.join(self.log_dir, f"window_{window_id}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'window': results['window'],
            'train_metrics': {
                'avg_reward': float(np.mean(results['train_metrics']['total_rewards'])) if results['train_metrics']['total_rewards'] else 0,
                'avg_win_rate': float(np.mean(results['train_metrics']['win_rates'])) if results['train_metrics']['win_rates'] else 0,
                'avg_sharpe': float(np.mean(results['train_metrics']['sharpe_ratios'])) if results['train_metrics']['sharpe_ratios'] else 0,
                'avg_q_change': float(np.mean(results['train_metrics']['q_value_changes'])) if results['train_metrics']['q_value_changes'] else 0
            },
            'test_results': results['test_results'],
            'agent_weights': results['agent_weights']
        }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def visualize_training_progress(self):
        """
        Create visualization of training progress
        """
        if not self.performance_metrics:
            print("No training data to visualize yet.")
            return
        
        if not HAS_MATPLOTLIB:
            print("Visualization skipped: matplotlib not installed")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Training Progress', fontsize=16)
        
        # Extract metrics
        windows = [m['window']['window_id'] for m in self.performance_metrics]
        train_rewards = [m['train_metrics']['avg_reward'] for m in self.performance_metrics]
        test_returns = [m['test_results']['avg_return'] for m in self.performance_metrics]
        win_rates = [m['test_results']['win_rate'] for m in self.performance_metrics]
        q_changes = [m['train_metrics']['avg_q_change'] for m in self.performance_metrics]
        
        # Plot 1: Training Rewards vs Test Returns
        ax1 = axes[0, 0]
        ax1.plot(windows, train_rewards, 'b-', label='Train Rewards', marker='o')
        ax1.plot(windows, test_returns, 'r-', label='Test Returns', marker='s')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Performance')
        ax1.set_title('Training vs Testing Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win Rate Evolution
        ax2 = axes[0, 1]
        ax2.plot(windows, win_rates, 'g-', marker='o')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate Over Time')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Q-Value Changes
        ax3 = axes[1, 0]
        ax3.bar(windows, q_changes, color='purple', alpha=0.7)
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Avg Q-Value Change')
        ax3.set_title('Learning Progress (Q-Value Updates)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Experience Buffer Growth
        ax4 = axes[1, 1]
        buffer_sizes = [m['agent_weights']['experience_buffer_size'] for m in self.performance_metrics]
        ax4.plot(windows, buffer_sizes, 'orange', marker='o')
        ax4.set_xlabel('Window')
        ax4.set_ylabel('Experience Buffer Size')
        ax4.set_title('Training Experience Accumulation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        viz_path = os.path.join(self.log_dir, f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(viz_path, dpi=300)
        print(f"\nVisualization saved to: {viz_path}")
        
        # Also display if possible
        try:
            plt.show()
        except:
            pass
    
    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report
        """
        if not self.performance_metrics:
            return "No training completed yet."
        
        report = []
        report.append("="*60)
        report.append("WALK-FORWARD TRAINING REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Windows: {len(self.performance_metrics)}")
        report.append("")
        
        # Overall statistics
        all_test_returns = [m['test_results']['avg_return'] for m in self.performance_metrics]
        all_win_rates = [m['test_results']['win_rate'] for m in self.performance_metrics]
        
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Average Test Return: {np.mean(all_test_returns):.2%}")
        report.append(f"  Best Test Return: {max(all_test_returns):.2%}")
        report.append(f"  Worst Test Return: {min(all_test_returns):.2%}")
        report.append(f"  Average Win Rate: {np.mean(all_win_rates):.2%}")
        report.append("")
        
        # Learning progress
        q_changes = [m['train_metrics']['avg_q_change'] for m in self.performance_metrics]
        report.append("LEARNING PROGRESS:")
        report.append(f"  Total Q-Value Change: {sum(q_changes):.4f}")
        report.append(f"  Average Q-Value Change per Window: {np.mean(q_changes):.4f}")
        report.append(f"  Final Experience Buffer: {self.performance_metrics[-1]['agent_weights']['experience_buffer_size']} samples")
        report.append("")
        
        # Window-by-window summary
        report.append("WINDOW-BY-WINDOW SUMMARY:")
        for metric in self.performance_metrics:
            window = metric['window']
            report.append(f"\nWindow {window['window_id']}:")
            report.append(f"  Train Period: {window['train_start']} to {window['train_end']}")
            report.append(f"  Test Period: {window['test_start']} to {window['test_end']}")
            report.append(f"  Test Return: {metric['test_results']['avg_return']:.2%}")
            report.append(f"  Win Rate: {metric['test_results']['win_rate']:.2%}")
            report.append(f"  Total Trades: {metric['test_results']['total_trades']}")
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.log_dir, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {report_path}")
        return report_text
    
    def run_walk_forward_training(self,
                                 symbols: List[str],
                                 start_date: str,
                                 end_date: str,
                                 learning_rate: float = 0.001,
                                 episodes_per_window: int = 5):
        """
        Main training loop
        """
        print(f"\n{'='*60}")
        print("WALK-FORWARD TRAINING SYSTEM")
        print(f"{'='*60}")
        print(f"Training Period: {start_date} to {end_date}")
        print(f"Symbols: {len(symbols)} stocks")
        print(f"Train Window: {self.train_window_days} days")
        print(f"Test Window: {self.test_window_days} days")
        print(f"Step Size: {self.step_days} days")
        print(f"Learning Rate: {learning_rate}")
        print(f"Episodes per Window: {episodes_per_window}")
        print(f"{'='*60}")
        
        # Prepare windows
        windows = self.prepare_data_windows(symbols, start_date, end_date)
        print(f"\nTotal Training Windows: {len(windows)}")
        
        if not windows:
            print("Error: No valid training windows found.")
            return
        
        # Confirm training
        response = input("\nProceed with training? (y/n): ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            return
        
        # Save initial model state
        self.agent.save_model()
        print("\nInitial model state saved.")
        
        # Train on each window
        for i, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"WINDOW {i+1}/{len(windows)}")
            print(f"{'='*60}")
            
            try:
                results = self.train_on_window(
                    symbols, 
                    window,
                    learning_rate,
                    episodes_per_window
                )
                
                self.performance_metrics.append(results)
                
                # Save model checkpoint
                checkpoint_path = os.path.join(
                    self.log_dir, 
                    f"model_checkpoint_window_{window['window_id']}.keras"
                )
                self.agent.model.save(checkpoint_path)
                print(f"Model checkpoint saved: {checkpoint_path}")
                
                # Show interim results
                print(f"\nWindow {window['window_id']} Results:")
                print(f"  Test Return: {results['test_results']['avg_return']:.2%}")
                print(f"  Win Rate: {results['test_results']['win_rate']:.2%}")
                print(f"  Experience Buffer: {results['agent_weights']['experience_buffer_size']} samples")
                
            except Exception as e:
                logging.error(f"Error in window {window['window_id']}: {e}")
                print(f"Error training window {window['window_id']}: {e}")
                continue
        
        # Generate final report
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        report = self.generate_training_report()
        print("\n" + report)
        
        # Create visualizations
        self.visualize_training_progress()
        
        # Save final model
        final_model_path = os.path.join(self.log_dir, "final_trained_model.keras")
        self.agent.save_model(final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
        
        return self.performance_metrics


# Integration function for main.py
def run_walk_forward_interface():
    """
    Interactive interface for Walk-Forward Training
    """
    print("\n" + "="*60)
    print("WALK-FORWARD TRAINING SYSTEM")
    print("="*60)
    print("\nThis system will train your LAEF bot using Walk-Forward Analysis,")
    print("the gold standard for developing robust trading strategies.")
    print("\nKey Features:")
    print("  • Rolling window training with out-of-sample testing")
    print("  • Real-time progress tracking and weight visualization")
    print("  • Automatic performance metrics and reporting")
    print("  • Model checkpointing for safety")
    
    # Get configuration
    print("\n" + "-"*60)
    print("TRAINING CONFIGURATION")
    print("-"*60)
    
    # Date range
    print("\nStep 1: Select Date Range")
    print("Recommended: At least 18 months of data for meaningful results (free tier compatible)")
    start_date = input("Start date (YYYY-MM-DD) [2023-06-01]: ").strip() or "2023-06-01"
    end_date = input("End date (YYYY-MM-DD) [2024-12-31]: ").strip() or "2024-12-31"
    
    # Window sizes
    print("\nStep 2: Configure Windows")
    print("Default: 1 year train, 3 months test, 1 month step")
    use_defaults = input("Use default window sizes? (Y/n): ").strip().lower()
    
    if use_defaults == 'n':
        train_days = int(input("Training window (days) [252]: ") or "252")
        test_days = int(input("Testing window (days) [63]: ") or "63")
        step_days = int(input("Step size (days) [21]: ") or "21")
    else:
        train_days, test_days, step_days = 252, 63, 21
    
    # Symbol selection
    print("\nStep 3: Select Symbols")
    print("1. Top 10 S&P 500 stocks (recommended for testing)")
    print("2. Custom list")
    print("3. Load from smart symbol selector")
    
    symbol_choice = input("Select option (1-3) [1]: ").strip() or "1"
    
    if symbol_choice == "1":
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B', 'JPM', 'JNJ']
    elif symbol_choice == "2":
        custom_symbols = input("Enter symbols (comma-separated): ").strip().upper()
        symbols = [s.strip() for s in custom_symbols.split(',')]
    else:
        from data.smart_symbol_selector import SmartSymbolSelector
        selector = SmartSymbolSelector()
        symbols = selector.analyze_symbols(limit=10)
    
    # Training parameters
    print("\nStep 4: Training Parameters")
    learning_rate = float(input("Learning rate (0.0001-0.01) [0.001]: ") or "0.001")
    episodes = int(input("Episodes per window (1-10) [5]: ") or "5")
    
    # Confirm configuration
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Windows: {train_days}d train / {test_days}d test / {step_days}d step")
    print(f"Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Episodes: {episodes}")
    
    # Calculate estimated time
    from datetime import datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    num_windows = (total_days - train_days - test_days) // step_days + 1
    estimated_time = num_windows * episodes * len(symbols) * 0.5 / 60  # rough estimate
    
    print(f"\nEstimated Windows: {num_windows}")
    print(f"Estimated Time: {estimated_time:.1f} hours")
    
    confirm = input("\nStart training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Create trainer and run
    trainer = WalkForwardTrainer(
        train_window_days=train_days,
        test_window_days=test_days,
        step_days=step_days
    )
    
    try:
        results = trainer.run_walk_forward_training(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            learning_rate=learning_rate,
            episodes_per_window=episodes
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Check logs/training/walk_forward/ for detailed results")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results saved in logs/training/walk_forward/")
    except Exception as e:
        print(f"\nError during training: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Test the trainer
    run_walk_forward_interface()