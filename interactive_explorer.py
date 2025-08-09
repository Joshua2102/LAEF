#!/usr/bin/env python3
"""
LAEF Interactive Result Explorer
Provides interactive exploration of backtest results with detailed analysis
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class InteractiveResultExplorer:
    """Interactive exploration and analysis of LAEF backtest results"""
    
    def __init__(self):
        self.current_results = None
        self.trade_history = pd.DataFrame()
        self.decision_history = pd.DataFrame()
        self.performance_metrics = {}
        self.loaded_files = {}
        
    def load_backtest_results(self, timestamp: str = None) -> bool:
        """Load backtest results from files"""
        try:
            if timestamp:
                # Load specific backtest
                decisions_file = f'logs/backtest_decisions_{timestamp}.csv'
                trades_file = f'logs/backtest_trades_{timestamp}.csv'
                summary_file = f'logs/backtest_summary_{timestamp}.txt'
            else:
                # Load most recent backtest
                import glob
                decision_files = glob.glob('logs/backtest_decisions_*.csv')
                if not decision_files:
                    print("No backtest results found")
                    return False
                    
                # Get most recent file
                decisions_file = max(decision_files, key=os.path.getctime)
                timestamp = decisions_file.split('_')[-1].replace('.csv', '')
                trades_file = f'logs/backtest_trades_{timestamp}.csv'
                summary_file = f'logs/backtest_summary_{timestamp}.txt'
            
            # Load data
            if os.path.exists(decisions_file):
                self.decision_history = pd.read_csv(decisions_file)
                self.loaded_files['decisions'] = decisions_file
                
            if os.path.exists(trades_file):
                self.trade_history = pd.read_csv(trades_file)
                self.loaded_files['trades'] = trades_file
                
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    self.loaded_files['summary'] = summary_file
                    self.summary_text = f.read()
            
            print(f"Loaded backtest results from {timestamp}")
            return True
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def explore_interactive(self):
        """Main interactive exploration interface"""
        if not self.loaded_files:
            print("No results loaded. Loading most recent backtest...")
            if not self.load_backtest_results():
                return
        
        while True:
            print("\n" + "="*60)
            print("LAEF INTERACTIVE RESULT EXPLORER")
            print("="*60)
            print("1. Overall Performance Summary")
            print("2. Trade Analysis")
            print("3. P&L Analysis with Contributing Factors")
            print("4. Symbol Performance Comparison")
            print("5. Decision Pattern Analysis")
            print("6. Win/Loss Distribution")
            print("7. Time-based Performance")
            print("8. Risk Analysis")
            print("9. Export Detailed Report")
            print("10. Load Different Backtest")
            print("0. Exit")
            print("="*60)
            
            choice = input("\nSelect option (0-10): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_performance_summary()
            elif choice == '2':
                self.analyze_trades()
            elif choice == '3':
                self.analyze_pnl_factors()
            elif choice == '4':
                self.compare_symbol_performance()
            elif choice == '5':
                self.analyze_decision_patterns()
            elif choice == '6':
                self.analyze_win_loss_distribution()
            elif choice == '7':
                self.analyze_time_performance()
            elif choice == '8':
                self.analyze_risk_metrics()
            elif choice == '9':
                self.export_detailed_report()
            elif choice == '10':
                self.load_different_backtest()
            else:
                print("Invalid option")
    
    def show_performance_summary(self):
        """Display overall performance summary"""
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*60)
        
        if hasattr(self, 'summary_text'):
            print(self.summary_text)
        
        if not self.trade_history.empty:
            # Calculate additional metrics
            total_trades = len(self.trade_history)
            buy_trades = self.trade_history[self.trade_history['action'] == 'BUY']
            sell_trades = self.trade_history[self.trade_history['action'] == 'SELL']
            
            if not sell_trades.empty and 'profit_pct' in sell_trades.columns:
                winning_trades = sell_trades[sell_trades['profit_pct'] > 0]
                losing_trades = sell_trades[sell_trades['profit_pct'] <= 0]
                
                avg_win = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0
                avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
                
                print(f"\nDETAILED METRICS:")
                print(f"Total Trades: {total_trades}")
                print(f"Buy Orders: {len(buy_trades)}")
                print(f"Sell Orders: {len(sell_trades)}")
                print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(sell_trades)*100:.1f}%)")
                print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(sell_trades)*100:.1f}%)")
                print(f"Average Win: {avg_win:.2%}")
                print(f"Average Loss: {avg_loss:.2%}")
                print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        
        input("\nPress Enter to continue...")
    
    def analyze_trades(self):
        """Detailed trade analysis"""
        if self.trade_history.empty:
            print("No trade history available")
            return
            
        print("\n" + "="*60)
        print("TRADE ANALYSIS")
        print("="*60)
        
        while True:
            print("\n1. Show All Trades")
            print("2. Show Winning Trades Only")
            print("3. Show Losing Trades Only")
            print("4. Show Trades by Symbol")
            print("5. Show Largest Wins/Losses")
            print("6. Back to Main Menu")
            
            sub_choice = input("\nSelect option (1-6): ").strip()
            
            if sub_choice == '6':
                break
            elif sub_choice == '1':
                self._show_all_trades()
            elif sub_choice == '2':
                self._show_winning_trades()
            elif sub_choice == '3':
                self._show_losing_trades()
            elif sub_choice == '4':
                self._show_trades_by_symbol()
            elif sub_choice == '5':
                self._show_largest_trades()
    
    def analyze_pnl_factors(self):
        """Analyze P&L with contributing factors"""
        print("\n" + "="*60)
        print("P&L ANALYSIS WITH CONTRIBUTING FACTORS")
        print("="*60)
        
        if self.trade_history.empty:
            print("No trade history available")
            return
        
        # Merge trade and decision data for comprehensive analysis
        sell_trades = self.trade_history[self.trade_history['action'] == 'SELL'].copy()
        
        if sell_trades.empty or 'profit_pct' not in sell_trades.columns:
            print("No completed trades with P&L data")
            return
        
        # Categorize P&L
        sell_trades['pnl_category'] = sell_trades['profit_pct'].apply(
            lambda x: 'Big Win' if x > 0.01 else
                     'Small Win' if x > 0 else
                     'Small Loss' if x > -0.01 else
                     'Big Loss'
        )
        
        # Analyze factors for each category
        for category in ['Big Win', 'Small Win', 'Small Loss', 'Big Loss']:
            category_trades = sell_trades[sell_trades['pnl_category'] == category]
            
            if not category_trades.empty:
                print(f"\n{category.upper()} ({len(category_trades)} trades):")
                print("-" * 40)
                
                # Average metrics
                avg_profit = category_trades['profit_pct'].mean()
                print(f"Average P&L: {avg_profit:.2%}")
                
                # Find corresponding buy decisions
                contributing_factors = self._analyze_contributing_factors(category_trades)
                
                print("Contributing Factors:")
                for factor, value in contributing_factors.items():
                    print(f"  • {factor}: {value}")
        
        # Overall P&L distribution
        print("\n" + "="*40)
        print("P&L DISTRIBUTION:")
        print(f"Total P&L: {sell_trades['profit_pct'].sum():.2%}")
        print(f"Average P&L per trade: {sell_trades['profit_pct'].mean():.2%}")
        print(f"Best Trade: {sell_trades['profit_pct'].max():.2%}")
        print(f"Worst Trade: {sell_trades['profit_pct'].min():.2%}")
        print(f"P&L Std Dev: {sell_trades['profit_pct'].std():.2%}")
        
        input("\nPress Enter to continue...")
    
    def _analyze_contributing_factors(self, trades_df):
        """Analyze factors contributing to trade outcomes"""
        factors = {}
        
        # Symbol distribution
        symbol_counts = trades_df['symbol'].value_counts()
        top_symbol = symbol_counts.index[0] if not symbol_counts.empty else "N/A"
        factors['Most Common Symbol'] = f"{top_symbol} ({symbol_counts.iloc[0]} trades)"
        
        # Time-based factors
        if 'timestamp' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
            avg_hour = trades_df['hour'].mean()
            factors['Average Trade Hour'] = f"{int(avg_hour)}:00"
        
        # Entry/exit reasoning (from decisions if available)
        if not self.decision_history.empty:
            # Match trades with decisions
            sell_reasons = []
            for _, trade in trades_df.iterrows():
                # Find corresponding sell decision
                symbol = trade['symbol']
                timestamp = trade['timestamp']
                
                decision = self.decision_history[
                    (self.decision_history['symbol'] == symbol) &
                    (self.decision_history['action'] == 'SELL') &
                    (self.decision_history['timestamp'] == timestamp)
                ]
                
                if not decision.empty and 'decision_reason' in decision.columns:
                    sell_reasons.append(decision.iloc[0]['decision_reason'])
            
            if sell_reasons:
                # Most common sell reason
                from collections import Counter
                reason_counts = Counter(sell_reasons)
                top_reason = reason_counts.most_common(1)[0]
                factors['Primary Exit Reason'] = f"{top_reason[0]} ({top_reason[1]} times)"
        
        return factors
    
    def compare_symbol_performance(self):
        """Compare performance across symbols"""
        print("\n" + "="*60)
        print("SYMBOL PERFORMANCE COMPARISON")
        print("="*60)
        
        if self.trade_history.empty:
            print("No trade history available")
            return
        
        # Group by symbol
        symbols = self.trade_history['symbol'].unique()
        
        symbol_stats = []
        for symbol in symbols:
            symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
            buy_trades = symbol_trades[symbol_trades['action'] == 'BUY']
            sell_trades = symbol_trades[symbol_trades['action'] == 'SELL']
            
            if not sell_trades.empty and 'profit_pct' in sell_trades.columns:
                total_pnl = sell_trades['profit_pct'].sum()
                avg_pnl = sell_trades['profit_pct'].mean()
                win_rate = len(sell_trades[sell_trades['profit_pct'] > 0]) / len(sell_trades) * 100
                
                symbol_stats.append({
                    'Symbol': symbol,
                    'Total Trades': len(buy_trades),
                    'Completed Trades': len(sell_trades),
                    'Total P&L': f"{total_pnl:.2%}",
                    'Avg P&L': f"{avg_pnl:.2%}",
                    'Win Rate': f"{win_rate:.1f}%",
                    'Best Trade': f"{sell_trades['profit_pct'].max():.2%}",
                    'Worst Trade': f"{sell_trades['profit_pct'].min():.2%}"
                })
        
        if symbol_stats:
            df = pd.DataFrame(symbol_stats)
            print(df.to_string(index=False))
        else:
            print("No completed trades to analyze")
        
        input("\nPress Enter to continue...")
    
    def analyze_decision_patterns(self):
        """Analyze trading decision patterns"""
        print("\n" + "="*60)
        print("DECISION PATTERN ANALYSIS")
        print("="*60)
        
        if self.decision_history.empty:
            print("No decision history available")
            return
        
        # Analyze ML scores
        if 'ml_score' in self.decision_history.columns:
            buy_decisions = self.decision_history[self.decision_history['action'] == 'BUY']
            sell_decisions = self.decision_history[self.decision_history['action'] == 'SELL']
            
            print("\nML SCORE ANALYSIS:")
            print(f"Average Buy ML Score: {buy_decisions['ml_score'].mean():.3f}")
            print(f"Average Sell ML Score: {sell_decisions['ml_score'].mean():.3f}")
            
            # Score distribution
            print("\nML Score Distribution (Buy Decisions):")
            print(f"  High Confidence (>0.7): {len(buy_decisions[buy_decisions['ml_score'] > 0.7])} trades")
            print(f"  Medium Confidence (0.5-0.7): {len(buy_decisions[(buy_decisions['ml_score'] >= 0.5) & (buy_decisions['ml_score'] <= 0.7)])} trades")
            print(f"  Low Confidence (<0.5): {len(buy_decisions[buy_decisions['ml_score'] < 0.5])} trades")
        
        # Decision frequency by symbol
        print("\nDECISION FREQUENCY BY SYMBOL:")
        symbol_decisions = self.decision_history.groupby(['symbol', 'action']).size().unstack(fill_value=0)
        print(symbol_decisions)
        
        # Time-based patterns
        if 'timestamp' in self.decision_history.columns:
            self.decision_history['date'] = pd.to_datetime(self.decision_history['timestamp']).dt.date
            daily_decisions = self.decision_history.groupby(['date', 'action']).size().unstack(fill_value=0)
            
            print(f"\nDAILY DECISION STATISTICS:")
            print(f"Average Buy Decisions per Day: {daily_decisions.get('BUY', 0).mean():.1f}")
            print(f"Average Sell Decisions per Day: {daily_decisions.get('SELL', 0).mean():.1f}")
            print(f"Most Active Day: {daily_decisions.sum(axis=1).idxmax()} ({daily_decisions.sum(axis=1).max()} decisions)")
        
        input("\nPress Enter to continue...")
    
    def analyze_win_loss_distribution(self):
        """Analyze win/loss distribution"""
        print("\n" + "="*60)
        print("WIN/LOSS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        if self.trade_history.empty:
            print("No trade history available")
            return
        
        sell_trades = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'].notna())
        ]
        
        if sell_trades.empty:
            print("No completed trades to analyze")
            return
        
        # Calculate distribution
        wins = sell_trades[sell_trades['profit_pct'] > 0]
        losses = sell_trades[sell_trades['profit_pct'] <= 0]
        
        print(f"Total Completed Trades: {len(sell_trades)}")
        print(f"Wins: {len(wins)} ({len(wins)/len(sell_trades)*100:.1f}%)")
        print(f"Losses: {len(losses)} ({len(losses)/len(sell_trades)*100:.1f}%)")
        
        # Profit ranges
        print("\nPROFIT DISTRIBUTION:")
        ranges = [
            (0.02, float('inf'), 'Large Wins (>2%)'),
            (0.01, 0.02, 'Medium Wins (1-2%)'),
            (0.005, 0.01, 'Small Wins (0.5-1%)'),
            (0, 0.005, 'Tiny Wins (<0.5%)'),
            (-0.005, 0, 'Tiny Losses (<0.5%)'),
            (-0.01, -0.005, 'Small Losses (0.5-1%)'),
            (-0.02, -0.01, 'Medium Losses (1-2%)'),
            (-float('inf'), -0.02, 'Large Losses (>2%)')
        ]
        
        for min_val, max_val, label in ranges:
            count = len(sell_trades[
                (sell_trades['profit_pct'] > min_val) & 
                (sell_trades['profit_pct'] <= max_val)
            ])
            if count > 0:
                avg_pnl = sell_trades[
                    (sell_trades['profit_pct'] > min_val) & 
                    (sell_trades['profit_pct'] <= max_val)
                ]['profit_pct'].mean()
                print(f"  {label}: {count} trades (avg: {avg_pnl:.2%})")
        
        # Consecutive wins/losses
        print("\nCONSECUTIVE PATTERNS:")
        self._analyze_consecutive_patterns(sell_trades)
        
        input("\nPress Enter to continue...")
    
    def _analyze_consecutive_patterns(self, trades_df):
        """Analyze consecutive win/loss patterns"""
        if trades_df.empty:
            return
            
        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp')
        
        # Track consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for _, trade in trades_df.iterrows():
            if trade['profit_pct'] > 0:
                if current_streak >= 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 1
            else:
                if current_streak <= 0:
                    current_streak -= 1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))
                else:
                    current_streak = -1
        
        print(f"Max Consecutive Wins: {max_win_streak}")
        print(f"Max Consecutive Losses: {max_loss_streak}")
    
    def analyze_time_performance(self):
        """Analyze performance over time"""
        print("\n" + "="*60)
        print("TIME-BASED PERFORMANCE ANALYSIS")
        print("="*60)
        
        if self.trade_history.empty or 'timestamp' not in self.trade_history.columns:
            print("No time-based data available")
            return
        
        # Convert timestamps
        self.trade_history['datetime'] = pd.to_datetime(self.trade_history['timestamp'])
        self.trade_history['date'] = self.trade_history['datetime'].dt.date
        self.trade_history['hour'] = self.trade_history['datetime'].dt.hour
        self.trade_history['weekday'] = self.trade_history['datetime'].dt.day_name()
        
        # Daily performance
        print("\nDAILY PERFORMANCE:")
        daily_sells = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'].notna())
        ].groupby('date')['profit_pct'].agg(['sum', 'mean', 'count'])
        
        if not daily_sells.empty:
            best_day = daily_sells['sum'].idxmax()
            worst_day = daily_sells['sum'].idxmin()
            
            print(f"Best Day: {best_day} (P&L: {daily_sells.loc[best_day, 'sum']:.2%})")
            print(f"Worst Day: {worst_day} (P&L: {daily_sells.loc[worst_day, 'sum']:.2%})")
            print(f"Average Daily P&L: {daily_sells['sum'].mean():.2%}")
        
        # Hour of day analysis
        print("\nHOUR OF DAY ANALYSIS:")
        hourly_trades = self.trade_history.groupby(['hour', 'action']).size().unstack(fill_value=0)
        if not hourly_trades.empty:
            print("Trading Activity by Hour:")
            for hour in range(9, 17):  # Market hours
                buys = hourly_trades.get('BUY', {}).get(hour, 0)
                sells = hourly_trades.get('SELL', {}).get(hour, 0)
                if buys + sells > 0:
                    print(f"  {hour}:00 - Buys: {buys}, Sells: {sells}")
        
        # Weekday analysis
        print("\nWEEKDAY PERFORMANCE:")
        weekday_sells = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'].notna())
        ].groupby('weekday')['profit_pct'].agg(['mean', 'count'])
        
        if not weekday_sells.empty:
            print(weekday_sells.to_string())
        
        input("\nPress Enter to continue...")
    
    def analyze_risk_metrics(self):
        """Analyze risk-related metrics"""
        print("\n" + "="*60)
        print("RISK ANALYSIS")
        print("="*60)
        
        if self.trade_history.empty:
            print("No trade history available")
            return
        
        sell_trades = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'].notna())
        ]
        
        if sell_trades.empty:
            print("No completed trades to analyze")
            return
        
        # Risk metrics
        returns = sell_trades['profit_pct'].values
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        print(f"VALUE AT RISK:")
        print(f"  95% VaR: {var_95:.2%} (95% of trades have P&L above this)")
        print(f"  99% VaR: {var_99:.2%} (99% of trades have P&L above this)")
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        print(f"\nMAXIMUM DRAWDOWN: {max_drawdown:.2%}")
        
        # Risk/Reward ratios
        avg_win = returns[returns > 0].mean() if any(returns > 0) else 0
        avg_loss = returns[returns < 0].mean() if any(returns < 0) else 0
        
        print(f"\nRISK/REWARD METRICS:")
        print(f"  Average Win: {avg_win:.2%}")
        print(f"  Average Loss: {avg_loss:.2%}")
        print(f"  Risk/Reward Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        
        # Sharpe-like ratio (simplified)
        if len(returns) > 1:
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            print(f"  Sharpe-like Ratio: {sharpe:.2f}")
        
        # Recovery analysis
        losing_trades = returns[returns < 0]
        if len(losing_trades) > 0:
            print(f"\nLOSS RECOVERY ANALYSIS:")
            print(f"  Total Losses: {losing_trades.sum():.2%}")
            print(f"  Trades Needed to Recover (at avg win): {abs(losing_trades.sum()/avg_win):.1f}" if avg_win > 0 else "N/A")
        
        input("\nPress Enter to continue...")
    
    def export_detailed_report(self):
        """Export comprehensive analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'logs/interactive_analysis_{timestamp}.txt'
        
        print(f"\nExporting detailed report to {report_file}...")
        
        with open(report_file, 'w') as f:
            f.write("LAEF INTERACTIVE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            # Include all analysis sections
            # This would include all the analysis from above methods
            # For brevity, showing structure
            
            f.write("1. PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n")
            if hasattr(self, 'summary_text'):
                f.write(self.summary_text + "\n\n")
            
            f.write("2. TRADE STATISTICS\n")
            f.write("-"*40 + "\n")
            # Add trade statistics
            
            f.write("3. P&L ANALYSIS\n")
            f.write("-"*40 + "\n")
            # Add P&L analysis
            
            # ... continue for all sections
        
        print(f"Report exported successfully!")
        input("\nPress Enter to continue...")
    
    def load_different_backtest(self):
        """Load a different backtest result"""
        import glob
        
        print("\nAvailable Backtest Results:")
        decision_files = glob.glob('logs/backtest_decisions_*.csv')
        
        if not decision_files:
            print("No backtest results found")
            return
        
        # Sort by modification time
        decision_files.sort(key=os.path.getmtime, reverse=True)
        
        for i, file in enumerate(decision_files[:10]):  # Show last 10
            timestamp = file.split('_')[-1].replace('.csv', '')
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"{i+1}. {timestamp} - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        choice = input("\nSelect backtest to load (1-10) or 0 to cancel: ").strip()
        
        if choice == '0':
            return
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(decision_files):
                timestamp = decision_files[idx].split('_')[-1].replace('.csv', '')
                if self.load_backtest_results(timestamp):
                    print("Backtest loaded successfully!")
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")
    
    # Helper methods for showing trades
    def _show_all_trades(self):
        """Show all trades with pagination"""
        print("\nALL TRADES:")
        print("-"*80)
        self._paginate_trades(self.trade_history)
    
    def _show_winning_trades(self):
        """Show only winning trades"""
        winning = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'] > 0)
        ]
        print(f"\nWINNING TRADES ({len(winning)} total):")
        print("-"*80)
        self._paginate_trades(winning)
    
    def _show_losing_trades(self):
        """Show only losing trades"""
        losing = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'] <= 0)
        ]
        print(f"\nLOSING TRADES ({len(losing)} total):")
        print("-"*80)
        self._paginate_trades(losing)
    
    def _show_trades_by_symbol(self):
        """Show trades filtered by symbol"""
        symbols = self.trade_history['symbol'].unique()
        print("\nAvailable symbols:", ", ".join(symbols))
        
        symbol = input("Enter symbol to filter: ").strip().upper()
        if symbol in symbols:
            filtered = self.trade_history[self.trade_history['symbol'] == symbol]
            print(f"\nTRADES FOR {symbol} ({len(filtered)} total):")
            print("-"*80)
            self._paginate_trades(filtered)
        else:
            print("Symbol not found")
    
    def _show_largest_trades(self):
        """Show largest wins and losses"""
        sells = self.trade_history[
            (self.trade_history['action'] == 'SELL') & 
            (self.trade_history['profit_pct'].notna())
        ]
        
        if sells.empty:
            print("No completed trades")
            return
        
        # Top 5 wins
        top_wins = sells.nlargest(5, 'profit_pct')
        print("\nTOP 5 WINS:")
        print("-"*80)
        for _, trade in top_wins.iterrows():
            print(f"{trade['symbol']} - {trade['profit_pct']:.2%} - "
                  f"${trade.get('value', 0):,.2f} - {trade['timestamp']}")
        
        # Top 5 losses
        top_losses = sells.nsmallest(5, 'profit_pct')
        print("\nTOP 5 LOSSES:")
        print("-"*80)
        for _, trade in top_losses.iterrows():
            print(f"{trade['symbol']} - {trade['profit_pct']:.2%} - "
                  f"${trade.get('value', 0):,.2f} - {trade['timestamp']}")
        
        input("\nPress Enter to continue...")
    
    def _paginate_trades(self, trades_df, page_size=10):
        """Display trades with pagination"""
        if trades_df.empty:
            print("No trades to display")
            return
        
        total_pages = (len(trades_df) + page_size - 1) // page_size
        current_page = 0
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(trades_df))
            
            page_trades = trades_df.iloc[start_idx:end_idx]
            
            # Display trades
            for _, trade in page_trades.iterrows():
                action = trade['action']
                symbol = trade['symbol']
                price = trade.get('price', 0)
                quantity = trade.get('quantity', 0)
                
                if action == 'BUY':
                    print(f"BUY  {symbol} - {quantity} shares @ ${price:.2f} - {trade['timestamp']}")
                else:
                    pnl = trade.get('profit_pct', 0)
                    print(f"SELL {symbol} - {quantity} shares @ ${price:.2f} "
                          f"(P&L: {pnl:.2%}) - {trade['timestamp']}")
            
            print(f"\nPage {current_page + 1} of {total_pages}")
            
            if total_pages > 1:
                nav = input("(N)ext, (P)revious, (Q)uit: ").strip().lower()
                if nav == 'n' and current_page < total_pages - 1:
                    current_page += 1
                elif nav == 'p' and current_page > 0:
                    current_page -= 1
                elif nav == 'q':
                    break
            else:
                input("\nPress Enter to continue...")
                break


    def show_trade_history(self):
        """Display trade history"""
        if self.trade_history.empty:
            print("No trade history available")
            return
        print("\nTrade History:")
        print(self.trade_history.head(20))
    
    def plot_portfolio_value(self):
        """Placeholder for portfolio value plotting"""
        print("\nPortfolio value plotting functionality")
        if self.current_results and 'portfolio_values' in self.current_results:
            values = self.current_results['portfolio_values']
            print(f"Portfolio values: {len(values)} data points")
            print(f"Initial: ${values[0]:,.2f}" if values else "No data")
            print(f"Final: ${values[-1]:,.2f}" if values else "No data")
        else:
            print("No portfolio value data available")


if __name__ == "__main__":
    explorer = InteractiveResultExplorer()
    explorer.explore_interactive()