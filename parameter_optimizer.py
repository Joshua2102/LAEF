#!/usr/bin/env python3
"""
LAEF Parameter Optimization Tool
Finds optimal trading parameters through grid search and intelligent optimization
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LAEFParameterOptimizer:
    """Optimizes LAEF trading parameters for maximum performance"""
    
    def __init__(self, base_config=None):
        self.base_config = base_config or {}
        self.optimization_results = []
        self.best_params = None
        self.best_score = -float('inf')
        
        # Define parameter search spaces
        self.param_ranges = {
            'profit_target': [0.002, 0.003, 0.004, 0.005, 0.006],  # 0.2% to 0.6%
            'stop_loss': [0.001, 0.002, 0.003, 0.004],  # 0.1% to 0.4%
            'q_buy': [0.20, 0.25, 0.30, 0.35, 0.40],
            'q_sell': [0.10, 0.15, 0.20, 0.25],
            'max_position': [0.05, 0.10, 0.15, 0.20],
            'risk_per_trade': [0.02, 0.025, 0.03, 0.035],
            'ml_profit_peak': [0.20, 0.25, 0.30, 0.35],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75]
        }
        
    def generate_param_combinations(self, param_subset=None):
        """Generate all parameter combinations for testing"""
        if param_subset:
            # Only optimize specified parameters
            params_to_test = {k: v for k, v in self.param_ranges.items() if k in param_subset}
        else:
            params_to_test = self.param_ranges
            
        # Generate all combinations
        param_names = list(params_to_test.keys())
        param_values = list(params_to_test.values())
        
        combinations = []
        for values in product(*param_values):
            param_dict = dict(zip(param_names, values))
            # Add base config values for non-optimized params
            full_config = self.base_config.copy()
            full_config.update(param_dict)
            combinations.append(full_config)
            
        return combinations
    
    def evaluate_parameters(self, params: Dict, symbols: List[str], days: int = 180) -> Dict:
        """Evaluate a single parameter combination"""
        try:
            # Import here to avoid circular imports
            from trading.backtester_unified import LAEFBacktester
            
            # Create backtester with test parameters
            backtester = LAEFBacktester(initial_cash=100000, custom_config=params)
            
            # Run backtest
            results = backtester.run_backtest(
                symbols=symbols,
                days=days,
                use_smart_selection=False
            )
            
            if not results or 'performance' not in results:
                return None
                
            perf = results['performance']
            
            # Calculate optimization score (weighted metrics)
            score = self.calculate_optimization_score(perf, results)
            
            return {
                'params': params,
                'score': score,
                'total_return': perf.get('total_return_pct', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_trades': perf.get('total_trades', 0),
                'final_value': perf.get('final_value', 100000),
                'sharpe_ratio': self.calculate_sharpe_ratio(results),
                'max_drawdown': self.calculate_max_drawdown(results),
                'profit_factor': self.calculate_profit_factor(results)
            }
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return None
    
    def calculate_optimization_score(self, performance: Dict, full_results: Dict) -> float:
        """Calculate weighted score for parameter optimization"""
        # Extract metrics
        total_return = performance.get('total_return_pct', 0)
        win_rate = performance.get('win_rate', 0)
        total_trades = performance.get('total_trades', 0)
        
        # Calculate additional metrics
        sharpe = self.calculate_sharpe_ratio(full_results)
        max_dd = self.calculate_max_drawdown(full_results)
        profit_factor = self.calculate_profit_factor(full_results)
        
        # Weighted scoring formula
        score = (
            total_return * 0.30 +  # 30% weight on returns
            win_rate * 0.20 +      # 20% weight on win rate
            sharpe * 10 * 0.20 +   # 20% weight on risk-adjusted returns
            (1 - max_dd) * 100 * 0.15 +  # 15% weight on drawdown control
            profit_factor * 10 * 0.10 +   # 10% weight on profit factor
            min(total_trades / 50, 1) * 100 * 0.05  # 5% weight on activity
        )
        
        # Penalty for extreme parameters
        if total_trades < 10:
            score *= 0.5  # Penalize low activity
        if win_rate < 30:
            score *= 0.7  # Penalize very low win rates
            
        return score
    
    def calculate_sharpe_ratio(self, results: Dict) -> float:
        """Calculate Sharpe ratio from results"""
        try:
            # Get daily returns from symbol results
            daily_returns = []
            for symbol, data in results.get('symbol_results', {}).items():
                if 'total_return' in data:
                    # Approximate daily return from total return
                    days = results.get('days', 180)
                    daily_return = (1 + data['total_return']) ** (1/days) - 1
                    daily_returns.append(daily_return)
            
            if not daily_returns:
                return 0
                
            # Calculate Sharpe ratio (annualized)
            returns_array = np.array(daily_returns)
            avg_return = np.mean(returns_array) * 252  # Annualize
            std_return = np.std(returns_array) * np.sqrt(252)  # Annualize
            
            if std_return == 0:
                return 0
                
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe = (avg_return - risk_free_rate) / std_return
            
            return max(-2, min(2, sharpe))  # Cap between -2 and 2
            
        except Exception:
            return 0
    
    def calculate_max_drawdown(self, results: Dict) -> float:
        """Calculate maximum drawdown from results"""
        try:
            # Approximate from win/loss data
            symbol_results = results.get('symbol_results', {})
            
            drawdowns = []
            for symbol, data in symbol_results.items():
                total_return = data.get('total_return', 0)
                win_rate = data.get('win_rate', 0.5)
                
                # Estimate max drawdown based on win rate and return
                if win_rate < 0.4:
                    estimated_dd = 0.15 + (0.4 - win_rate) * 0.5
                else:
                    estimated_dd = max(0.05, 0.15 * (1 - win_rate))
                    
                drawdowns.append(estimated_dd)
            
            return np.mean(drawdowns) if drawdowns else 0.10
            
        except Exception:
            return 0.10
    
    def calculate_profit_factor(self, results: Dict) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            total_profit = 0
            total_loss = 0
            
            for symbol, data in results.get('symbol_results', {}).items():
                winning_trades = data.get('winning_trades', 0)
                losing_trades = data.get('losing_trades', 0)
                total_return = data.get('total_return', 0)
                
                if winning_trades + losing_trades > 0:
                    avg_win = total_return / (winning_trades + losing_trades) * 2  # Rough estimate
                    avg_loss = avg_win * 0.7  # Assume 70% loss size vs win size
                    
                    total_profit += winning_trades * avg_win
                    total_loss += losing_trades * abs(avg_loss)
            
            if total_loss == 0:
                return 2.0 if total_profit > 0 else 0
                
            return min(3.0, total_profit / total_loss)
            
        except Exception:
            return 1.0
    
    def optimize_parameters(self, symbols: List[str], param_subset=None, 
                          days: int = 180, max_workers: int = None):
        """Run parameter optimization"""
        print("Starting LAEF Parameter Optimization...")
        print(f"Testing on symbols: {', '.join(symbols)}")
        print(f"Backtest period: {days} days")
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations(param_subset)
        total_combinations = len(param_combinations)
        print(f"Total parameter combinations to test: {total_combinations}")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count() - 1, 8)
        
        # Run parallel optimization
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self.evaluate_parameters, params, symbols, days): params
                for params in param_combinations
            }
            
            # Process results as they complete
            for future in as_completed(future_to_params):
                completed += 1
                params = future_to_params[future]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Update best if needed
                        if result['score'] > self.best_score:
                            self.best_score = result['score']
                            self.best_params = result['params']
                            print(f"\nNew best score: {self.best_score:.2f}")
                            print(f"Parameters: {self._format_params(self.best_params)}")
                        
                    # Progress update
                    if completed % 10 == 0:
                        print(f"Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error processing result: {e}")
        
        self.optimization_results = results
        print(f"\nOptimization complete! Tested {len(results)} valid combinations")
        
        return self.get_optimization_report()
    
    def _format_params(self, params: Dict) -> str:
        """Format parameters for display"""
        key_params = ['profit_target', 'stop_loss', 'q_buy', 'q_sell', 'max_position']
        formatted = []
        for key in key_params:
            if key in params:
                if key in ['profit_target', 'stop_loss', 'max_position']:
                    formatted.append(f"{key}={params[key]:.1%}")
                else:
                    formatted.append(f"{key}={params[key]:.2f}")
        return ", ".join(formatted)
    
    def get_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        # Sort results by score
        sorted_results = sorted(self.optimization_results, 
                              key=lambda x: x['score'], 
                              reverse=True)
        
        # Get top 10 configurations
        top_configs = sorted_results[:10]
        
        # Analyze parameter importance
        param_importance = self.analyze_parameter_importance()
        
        # Generate report
        report = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': {
                'total_return': sorted_results[0]['total_return'],
                'win_rate': sorted_results[0]['win_rate'],
                'sharpe_ratio': sorted_results[0]['sharpe_ratio'],
                'max_drawdown': sorted_results[0]['max_drawdown'],
                'profit_factor': sorted_results[0]['profit_factor'],
                'total_trades': sorted_results[0]['total_trades']
            },
            'top_10_configs': top_configs,
            'parameter_importance': param_importance,
            'total_tested': len(self.optimization_results),
            'optimization_summary': self.generate_summary()
        }
        
        # Save report
        self.save_optimization_report(report)
        
        return report
    
    def analyze_parameter_importance(self) -> Dict:
        """Analyze which parameters have the most impact on performance"""
        if not self.optimization_results:
            return {}
        
        importance = {}
        
        for param in self.param_ranges.keys():
            # Group results by this parameter value
            param_groups = {}
            for result in self.optimization_results:
                value = result['params'].get(param)
                if value is not None:
                    if value not in param_groups:
                        param_groups[value] = []
                    param_groups[value].append(result['score'])
            
            # Calculate variance in scores across different values
            if param_groups:
                avg_scores = [np.mean(scores) for scores in param_groups.values()]
                importance[param] = {
                    'variance': np.var(avg_scores),
                    'best_value': max(param_groups.keys(), 
                                     key=lambda x: np.mean(param_groups[x])),
                    'worst_value': min(param_groups.keys(), 
                                      key=lambda x: np.mean(param_groups[x])),
                    'impact_score': np.std(avg_scores) * 100
                }
        
        return importance
    
    def generate_summary(self) -> str:
        """Generate human-readable summary of optimization results"""
        if not self.best_params:
            return "No optimization results available"
        
        summary = []
        summary.append("OPTIMIZATION SUMMARY")
        summary.append("=" * 50)
        
        # Best configuration
        summary.append("\nBEST CONFIGURATION FOUND:")
        summary.append(f"Score: {self.best_score:.2f}")
        summary.append(f"Profit Target: {self.best_params.get('profit_target', 0.004):.2%}")
        summary.append(f"Stop Loss: {self.best_params.get('stop_loss', 0.002):.2%}")
        summary.append(f"Q-Buy Threshold: {self.best_params.get('q_buy', 0.30):.2f}")
        summary.append(f"Q-Sell Threshold: {self.best_params.get('q_sell', 0.20):.2f}")
        summary.append(f"Max Position Size: {self.best_params.get('max_position', 0.15):.1%}")
        
        # Performance metrics
        best_result = self.optimization_results[0]
        summary.append(f"\nPERFORMANCE METRICS:")
        summary.append(f"Total Return: {best_result['total_return']:.2f}%")
        summary.append(f"Win Rate: {best_result['win_rate']:.1f}%")
        summary.append(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        summary.append(f"Max Drawdown: {best_result['max_drawdown']:.1%}")
        summary.append(f"Profit Factor: {best_result['profit_factor']:.2f}")
        
        # Key insights
        summary.append("\nKEY INSIGHTS:")
        
        # Profit target insight
        if self.best_params.get('profit_target', 0) < 0.003:
            summary.append("• Lower profit targets (< 0.3%) work better for scalping")
        else:
            summary.append("• Moderate profit targets (0.3-0.5%) balance risk/reward")
        
        # Risk management insight
        if self.best_params.get('stop_loss', 0) < 0.002:
            summary.append("• Tight stop losses (< 0.2%) minimize downside risk")
        else:
            summary.append("• Wider stops (0.2-0.4%) allow trades room to breathe")
        
        # Q-value insights
        if self.best_params.get('q_buy', 0) < 0.30:
            summary.append("• Lower Q-buy thresholds increase trading opportunities")
        else:
            summary.append("• Higher Q-buy thresholds focus on quality setups")
        
        return "\n".join(summary)
    
    def save_optimization_report(self, report: Dict):
        """Save optimization report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = f'logs/optimization_report_{timestamp}.json'
        with open(json_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_report = self._convert_for_json(report)
            json.dump(json_report, f, indent=2)
        
        # Save human-readable report
        text_file = f'logs/optimization_summary_{timestamp}.txt'
        with open(text_file, 'w') as f:
            f.write(report['optimization_summary'])
            f.write("\n\nTOP 10 CONFIGURATIONS:\n")
            f.write("=" * 80 + "\n")
            
            for i, config in enumerate(report['top_10_configs'][:10]):
                f.write(f"\n{i+1}. Score: {config['score']:.2f}\n")
                f.write(f"   Return: {config['total_return']:.2f}%, ")
                f.write(f"Win Rate: {config['win_rate']:.1f}%, ")
                f.write(f"Sharpe: {config['sharpe_ratio']:.2f}\n")
                f.write(f"   Key params: {self._format_params(config['params'])}\n")
        
        print(f"\nOptimization reports saved:")
        print(f"  JSON: {json_file}")
        print(f"  Text: {text_file}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def get_recommended_config(self) -> Dict:
        """Get recommended configuration based on optimization results"""
        if not self.best_params:
            return self.base_config
        
        # Start with best params
        recommended = self.best_params.copy()
        
        # Apply safety adjustments
        # Ensure stop loss is not too tight
        if recommended.get('stop_loss', 0) < 0.001:
            recommended['stop_loss'] = 0.001
        
        # Ensure position size is reasonable
        if recommended.get('max_position', 0) > 0.20:
            recommended['max_position'] = 0.20
        
        # Add optimization metadata
        recommended['_optimization_score'] = self.best_score
        recommended['_optimization_timestamp'] = datetime.now().isoformat()
        
        return recommended