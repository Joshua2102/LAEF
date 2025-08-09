#!/usr/bin/env python3
"""
LAEF P&L Analyzer with Contributing Factors
Provides deep analysis of profit/loss patterns and their root causes
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

class PnLAnalyzer:
    """Analyzes P&L with detailed contributing factor attribution"""
    
    def __init__(self):
        self.trade_data = None
        self.decision_data = None
        self.market_data = {}
        self.factor_contributions = defaultdict(list)
        
    def analyze_pnl_factors(self, trades_df: pd.DataFrame, decisions_df: pd.DataFrame, 
                           market_data: Dict = None) -> Dict:
        """
        Comprehensive P&L analysis with factor attribution
        
        Returns detailed breakdown of what drives profits and losses
        """
        self.trade_data = trades_df
        self.decision_data = decisions_df
        self.market_data = market_data or {}
        
        # Prepare data
        self._prepare_analysis_data()
        
        # Run multi-factor analysis
        analysis_results = {
            'summary': self._generate_pnl_summary(),
            'factor_attribution': self._attribute_factors(),
            'pattern_analysis': self._analyze_patterns(),
            'market_condition_impact': self._analyze_market_conditions(),
            'timing_analysis': self._analyze_timing_factors(),
            'risk_reward_analysis': self._analyze_risk_reward(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis_results
    
    def _prepare_analysis_data(self):
        """Prepare and enrich data for analysis"""
        # Filter for completed trades (sells)
        self.completed_trades = self.trade_data[
            (self.trade_data['action'] == 'SELL') & 
            (self.trade_data['profit_pct'].notna())
        ].copy()
        
        # Match sells with corresponding buys
        self._match_buy_sell_pairs()
        
        # Categorize P&L
        self.completed_trades['pnl_category'] = self.completed_trades['profit_pct'].apply(
            self._categorize_pnl
        )
        
        # Add time-based features
        if 'timestamp' in self.completed_trades.columns:
            self.completed_trades['datetime'] = pd.to_datetime(self.completed_trades['timestamp'])
            self.completed_trades['hour'] = self.completed_trades['datetime'].dt.hour
            self.completed_trades['weekday'] = self.completed_trades['datetime'].dt.weekday
            self.completed_trades['day_of_month'] = self.completed_trades['datetime'].dt.day
    
    def _categorize_pnl(self, pnl):
        """Categorize P&L into meaningful buckets"""
        if pnl > 0.02:
            return 'Large Win'
        elif pnl > 0.01:
            return 'Medium Win'
        elif pnl > 0.005:
            return 'Small Win'
        elif pnl > 0:
            return 'Tiny Win'
        elif pnl > -0.005:
            return 'Tiny Loss'
        elif pnl > -0.01:
            return 'Small Loss'
        elif pnl > -0.02:
            return 'Medium Loss'
        else:
            return 'Large Loss'
    
    def _match_buy_sell_pairs(self):
        """Match sell trades with their corresponding buy trades"""
        # This would be more sophisticated with FIFO tracking
        # For now, simplified matching by symbol and sequence
        buy_trades = self.trade_data[self.trade_data['action'] == 'BUY'].copy()
        
        for idx, sell in self.completed_trades.iterrows():
            symbol = sell['symbol']
            sell_time = sell['timestamp']
            
            # Find the most recent buy before this sell
            symbol_buys = buy_trades[
                (buy_trades['symbol'] == symbol) & 
                (buy_trades['timestamp'] < sell_time)
            ]
            
            if not symbol_buys.empty:
                matching_buy = symbol_buys.iloc[-1]
                self.completed_trades.at[idx, 'buy_price'] = matching_buy['price']
                self.completed_trades.at[idx, 'hold_duration'] = (
                    pd.to_datetime(sell_time) - pd.to_datetime(matching_buy['timestamp'])
                ).total_seconds() / 3600  # Hours
                
                # Get buy decision data
                buy_decision = self._get_decision_data(
                    symbol, matching_buy['timestamp'], 'BUY'
                )
                if buy_decision is not None:
                    self.completed_trades.at[idx, 'buy_ml_score'] = buy_decision.get('ml_score', 0.5)
                    self.completed_trades.at[idx, 'buy_reason'] = buy_decision.get('decision_reason', '')
    
    def _get_decision_data(self, symbol, timestamp, action):
        """Get decision data for a specific trade"""
        if self.decision_data is None or self.decision_data.empty:
            return None
            
        decision = self.decision_data[
            (self.decision_data['symbol'] == symbol) &
            (self.decision_data['action'] == action) &
            (self.decision_data['timestamp'] == timestamp)
        ]
        
        if not decision.empty:
            return decision.iloc[0].to_dict()
        return None
    
    def _generate_pnl_summary(self) -> Dict:
        """Generate overall P&L summary statistics"""
        if self.completed_trades.empty:
            return {'error': 'No completed trades'}
        
        total_pnl = self.completed_trades['profit_pct'].sum()
        avg_pnl = self.completed_trades['profit_pct'].mean()
        
        wins = self.completed_trades[self.completed_trades['profit_pct'] > 0]
        losses = self.completed_trades[self.completed_trades['profit_pct'] < 0]
        
        return {
            'total_trades': len(self.completed_trades),
            'total_pnl_pct': total_pnl,
            'average_pnl_pct': avg_pnl,
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / len(self.completed_trades) * 100,
            'average_win': wins['profit_pct'].mean() if not wins.empty else 0,
            'average_loss': losses['profit_pct'].mean() if not losses.empty else 0,
            'largest_win': wins['profit_pct'].max() if not wins.empty else 0,
            'largest_loss': losses['profit_pct'].min() if not losses.empty else 0,
            'profit_factor': abs(wins['profit_pct'].sum() / losses['profit_pct'].sum()) 
                           if not losses.empty and losses['profit_pct'].sum() != 0 else 0,
            'pnl_distribution': self.completed_trades['pnl_category'].value_counts().to_dict()
        }
    
    def _attribute_factors(self) -> Dict:
        """Attribute P&L to specific factors"""
        factors = {
            'ml_score_impact': self._analyze_ml_score_impact(),
            'timing_impact': self._analyze_timing_impact(),
            'hold_duration_impact': self._analyze_hold_duration_impact(),
            'symbol_impact': self._analyze_symbol_impact(),
            'entry_quality_impact': self._analyze_entry_quality(),
            'exit_timing_impact': self._analyze_exit_timing()
        }
        
        # Calculate relative importance
        total_impact = sum(abs(f.get('impact_score', 0)) for f in factors.values())
        
        for factor_name, factor_data in factors.items():
            if total_impact > 0:
                factor_data['relative_importance'] = abs(
                    factor_data.get('impact_score', 0)
                ) / total_impact * 100
            else:
                factor_data['relative_importance'] = 0
        
        return factors
    
    def _analyze_ml_score_impact(self) -> Dict:
        """Analyze how ML scores correlate with P&L"""
        if 'buy_ml_score' not in self.completed_trades.columns:
            return {'error': 'No ML score data available'}
        
        # Group by ML score ranges
        score_ranges = [
            (0, 0.3, 'Low'),
            (0.3, 0.5, 'Medium-Low'),
            (0.5, 0.7, 'Medium-High'),
            (0.7, 1.0, 'High')
        ]
        
        results = []
        for min_score, max_score, label in score_ranges:
            mask = (
                (self.completed_trades['buy_ml_score'] >= min_score) & 
                (self.completed_trades['buy_ml_score'] < max_score)
            )
            
            group_trades = self.completed_trades[mask]
            if not group_trades.empty:
                results.append({
                    'score_range': label,
                    'avg_pnl': group_trades['profit_pct'].mean(),
                    'win_rate': len(group_trades[group_trades['profit_pct'] > 0]) / len(group_trades) * 100,
                    'trade_count': len(group_trades)
                })
        
        # Calculate impact score
        if results:
            impact_score = np.std([r['avg_pnl'] for r in results]) * 100
        else:
            impact_score = 0
        
        return {
            'score_ranges': results,
            'impact_score': impact_score,
            'conclusion': self._generate_ml_conclusion(results)
        }
    
    def _generate_ml_conclusion(self, results):
        """Generate conclusion about ML score impact"""
        if not results:
            return "Insufficient data"
        
        # Find best performing score range
        best_range = max(results, key=lambda x: x['avg_pnl'])
        worst_range = min(results, key=lambda x: x['avg_pnl'])
        
        if best_range['avg_pnl'] > worst_range['avg_pnl'] * 1.5:
            return f"ML scores strongly predict performance. {best_range['score_range']} scores yield {best_range['avg_pnl']:.2%} avg return vs {worst_range['avg_pnl']:.2%} for {worst_range['score_range']} scores."
        else:
            return "ML scores show moderate predictive power. Consider refining the model or combining with other factors."
    
    def _analyze_timing_impact(self) -> Dict:
        """Analyze how timing affects P&L"""
        if 'hour' not in self.completed_trades.columns:
            return {'error': 'No timing data available'}
        
        # Hour of day analysis
        hourly_pnl = self.completed_trades.groupby('hour')['profit_pct'].agg(['mean', 'count'])
        
        # Weekday analysis
        weekday_pnl = self.completed_trades.groupby('weekday')['profit_pct'].agg(['mean', 'count'])
        
        # Find best/worst times
        best_hour = hourly_pnl['mean'].idxmax() if not hourly_pnl.empty else None
        worst_hour = hourly_pnl['mean'].idxmin() if not hourly_pnl.empty else None
        
        best_day = weekday_pnl['mean'].idxmax() if not weekday_pnl.empty else None
        worst_day = weekday_pnl['mean'].idxmin() if not weekday_pnl.empty else None
        
        # Calculate impact score
        hour_variance = hourly_pnl['mean'].var() if not hourly_pnl.empty else 0
        day_variance = weekday_pnl['mean'].var() if not weekday_pnl.empty else 0
        impact_score = (hour_variance + day_variance) * 1000
        
        return {
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'best_weekday': best_day,
            'worst_weekday': worst_day,
            'hourly_performance': hourly_pnl.to_dict() if not hourly_pnl.empty else {},
            'weekday_performance': weekday_pnl.to_dict() if not weekday_pnl.empty else {},
            'impact_score': impact_score,
            'conclusion': f"Best trading hours: {best_hour}:00-{best_hour+1}:00. Avoid: {worst_hour}:00-{worst_hour+1}:00"
        }
    
    def _analyze_hold_duration_impact(self) -> Dict:
        """Analyze how holding period affects P&L"""
        if 'hold_duration' not in self.completed_trades.columns:
            return {'error': 'No duration data available'}
        
        # Group by duration buckets (in hours)
        duration_buckets = [
            (0, 1, 'Ultra-short (<1h)'),
            (1, 4, 'Short (1-4h)'),
            (4, 24, 'Intraday (4-24h)'),
            (24, 72, 'Multi-day (1-3d)'),
            (72, float('inf'), 'Long (>3d)')
        ]
        
        results = []
        for min_dur, max_dur, label in duration_buckets:
            mask = (
                (self.completed_trades['hold_duration'] >= min_dur) & 
                (self.completed_trades['hold_duration'] < max_dur)
            )
            
            group_trades = self.completed_trades[mask]
            if not group_trades.empty:
                results.append({
                    'duration': label,
                    'avg_pnl': group_trades['profit_pct'].mean(),
                    'win_rate': len(group_trades[group_trades['profit_pct'] > 0]) / len(group_trades) * 100,
                    'trade_count': len(group_trades)
                })
        
        # Find optimal duration
        if results:
            best_duration = max(results, key=lambda x: x['avg_pnl'])
            impact_score = np.std([r['avg_pnl'] for r in results]) * 100
        else:
            best_duration = None
            impact_score = 0
        
        return {
            'duration_analysis': results,
            'optimal_duration': best_duration['duration'] if best_duration else 'Unknown',
            'impact_score': impact_score,
            'conclusion': f"Optimal holding period: {best_duration['duration']} with {best_duration['avg_pnl']:.2%} avg return" 
                         if best_duration else "Insufficient data"
        }
    
    def _analyze_symbol_impact(self) -> Dict:
        """Analyze symbol-specific performance"""
        symbol_performance = self.completed_trades.groupby('symbol').agg({
            'profit_pct': ['mean', 'sum', 'count'],
            'quantity': 'sum'
        })
        
        symbol_performance.columns = ['avg_pnl', 'total_pnl', 'trade_count', 'total_shares']
        
        # Calculate win rates
        for symbol in symbol_performance.index:
            symbol_trades = self.completed_trades[self.completed_trades['symbol'] == symbol]
            wins = symbol_trades[symbol_trades['profit_pct'] > 0]
            symbol_performance.at[symbol, 'win_rate'] = len(wins) / len(symbol_trades) * 100
        
        # Rank symbols
        symbol_performance['score'] = (
            symbol_performance['avg_pnl'] * 0.5 + 
            symbol_performance['win_rate'] * 0.003 +
            symbol_performance['total_pnl'] * 0.2
        )
        
        top_symbols = symbol_performance.nlargest(3, 'score')
        bottom_symbols = symbol_performance.nsmallest(3, 'score')
        
        impact_score = symbol_performance['avg_pnl'].std() * 100
        
        return {
            'symbol_stats': symbol_performance.to_dict(),
            'top_performers': top_symbols.index.tolist(),
            'worst_performers': bottom_symbols.index.tolist(),
            'impact_score': impact_score,
            'conclusion': f"Focus on top performers: {', '.join(top_symbols.index[:3])}. Consider avoiding: {', '.join(bottom_symbols.index[:3])}"
        }
    
    def _analyze_entry_quality(self) -> Dict:
        """Analyze entry point quality"""
        # This would ideally use market data to check entry vs daily range
        # For now, use available data
        
        if 'buy_ml_score' not in self.completed_trades.columns:
            return {'error': 'Insufficient data for entry analysis'}
        
        # High confidence entries
        high_conf = self.completed_trades[self.completed_trades['buy_ml_score'] > 0.6]
        low_conf = self.completed_trades[self.completed_trades['buy_ml_score'] <= 0.6]
        
        results = {
            'high_confidence_entries': {
                'count': len(high_conf),
                'avg_pnl': high_conf['profit_pct'].mean() if not high_conf.empty else 0,
                'win_rate': len(high_conf[high_conf['profit_pct'] > 0]) / len(high_conf) * 100 
                           if not high_conf.empty else 0
            },
            'low_confidence_entries': {
                'count': len(low_conf),
                'avg_pnl': low_conf['profit_pct'].mean() if not low_conf.empty else 0,
                'win_rate': len(low_conf[low_conf['profit_pct'] > 0]) / len(low_conf) * 100 
                           if not low_conf.empty else 0
            }
        }
        
        # Calculate impact
        impact_score = abs(
            results['high_confidence_entries']['avg_pnl'] - 
            results['low_confidence_entries']['avg_pnl']
        ) * 100
        
        return {
            'entry_analysis': results,
            'impact_score': impact_score,
            'conclusion': "High confidence entries significantly outperform" 
                         if results['high_confidence_entries']['avg_pnl'] > 
                            results['low_confidence_entries']['avg_pnl'] * 1.2
                         else "Entry confidence shows moderate impact on returns"
        }
    
    def _analyze_exit_timing(self) -> Dict:
        """Analyze exit timing effectiveness"""
        # Categorize exits by trigger type
        exit_categories = defaultdict(list)
        
        for _, trade in self.completed_trades.iterrows():
            # Determine exit trigger (simplified)
            pnl = trade['profit_pct']
            
            if pnl > 0.004:  # Assuming 0.4% profit target
                category = 'profit_target'
            elif pnl < -0.002:  # Assuming 0.2% stop loss
                category = 'stop_loss'
            else:
                category = 'ml_signal'
            
            exit_categories[category].append(pnl)
        
        # Analyze each category
        results = {}
        for category, pnls in exit_categories.items():
            if pnls:
                results[category] = {
                    'count': len(pnls),
                    'avg_pnl': np.mean(pnls),
                    'total_pnl': sum(pnls),
                    'percentage': len(pnls) / len(self.completed_trades) * 100
                }
        
        # Calculate effectiveness score
        if 'profit_target' in results and 'stop_loss' in results:
            profit_target_ratio = results['profit_target']['count'] / (
                results['profit_target']['count'] + results['stop_loss']['count']
            )
            effectiveness_score = profit_target_ratio * 100
        else:
            effectiveness_score = 50
        
        return {
            'exit_breakdown': results,
            'effectiveness_score': effectiveness_score,
            'impact_score': np.std([r['avg_pnl'] for r in results.values()]) * 100 if results else 0,
            'conclusion': f"Exit effectiveness: {effectiveness_score:.1f}%. " +
                         ("Excellent risk management" if effectiveness_score > 60 
                          else "Consider tightening exit criteria")
        }
    
    def _analyze_patterns(self) -> Dict:
        """Identify winning and losing patterns"""
        patterns = {
            'winning_patterns': [],
            'losing_patterns': [],
            'neutral_patterns': []
        }
        
        # Pattern 1: High ML score + specific hour
        for hour in range(9, 16):
            hour_high_ml = self.completed_trades[
                (self.completed_trades.get('hour', 0) == hour) & 
                (self.completed_trades.get('buy_ml_score', 0) > 0.6)
            ]
            
            if len(hour_high_ml) >= 3:
                avg_pnl = hour_high_ml['profit_pct'].mean()
                win_rate = len(hour_high_ml[hour_high_ml['profit_pct'] > 0]) / len(hour_high_ml) * 100
                
                pattern = {
                    'description': f"High ML score (>0.6) trades at {hour}:00",
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'occurrences': len(hour_high_ml)
                }
                
                if avg_pnl > 0.002:
                    patterns['winning_patterns'].append(pattern)
                elif avg_pnl < -0.002:
                    patterns['losing_patterns'].append(pattern)
                else:
                    patterns['neutral_patterns'].append(pattern)
        
        # Pattern 2: Symbol + day of week combinations
        if 'weekday' in self.completed_trades.columns:
            for symbol in self.completed_trades['symbol'].unique():
                for day in range(5):  # Monday to Friday
                    day_symbol = self.completed_trades[
                        (self.completed_trades['symbol'] == symbol) & 
                        (self.completed_trades['weekday'] == day)
                    ]
                    
                    if len(day_symbol) >= 3:
                        avg_pnl = day_symbol['profit_pct'].mean()
                        win_rate = len(day_symbol[day_symbol['profit_pct'] > 0]) / len(day_symbol) * 100
                        
                        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                        pattern = {
                            'description': f"{symbol} trades on {days[day]}",
                            'avg_pnl': avg_pnl,
                            'win_rate': win_rate,
                            'occurrences': len(day_symbol)
                        }
                        
                        if avg_pnl > 0.003:
                            patterns['winning_patterns'].append(pattern)
                        elif avg_pnl < -0.003:
                            patterns['losing_patterns'].append(pattern)
        
        # Sort patterns by performance
        for category in patterns:
            patterns[category].sort(key=lambda x: x['avg_pnl'], reverse=True)
        
        return patterns
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze impact of market conditions on P&L"""
        # This would use market data if available
        # For now, use proxy indicators
        
        conditions = {}
        
        # Volatility proxy (using P&L variance)
        if len(self.completed_trades) > 10:
            # Split into high/low volatility periods
            pnl_rolling_std = self.completed_trades['profit_pct'].rolling(5).std()
            median_vol = pnl_rolling_std.median()
            
            high_vol_trades = self.completed_trades[pnl_rolling_std > median_vol]
            low_vol_trades = self.completed_trades[pnl_rolling_std <= median_vol]
            
            conditions['volatility_impact'] = {
                'high_volatility': {
                    'avg_pnl': high_vol_trades['profit_pct'].mean() if not high_vol_trades.empty else 0,
                    'win_rate': len(high_vol_trades[high_vol_trades['profit_pct'] > 0]) / len(high_vol_trades) * 100 
                               if not high_vol_trades.empty else 0,
                    'trade_count': len(high_vol_trades)
                },
                'low_volatility': {
                    'avg_pnl': low_vol_trades['profit_pct'].mean() if not low_vol_trades.empty else 0,
                    'win_rate': len(low_vol_trades[low_vol_trades['profit_pct'] > 0]) / len(low_vol_trades) * 100 
                               if not low_vol_trades.empty else 0,
                    'trade_count': len(low_vol_trades)
                }
            }
        
        return conditions
    
    def _analyze_risk_reward(self) -> Dict:
        """Analyze risk/reward characteristics"""
        wins = self.completed_trades[self.completed_trades['profit_pct'] > 0]
        losses = self.completed_trades[self.completed_trades['profit_pct'] < 0]
        
        if wins.empty or losses.empty:
            return {'error': 'Insufficient win/loss data'}
        
        # Calculate risk/reward metrics
        avg_win = wins['profit_pct'].mean()
        avg_loss = abs(losses['profit_pct'].mean())
        
        # Expected value per trade
        win_rate = len(wins) / len(self.completed_trades)
        expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Kelly criterion (simplified)
        if avg_loss > 0:
            kelly_fraction = (win_rate - (1 - win_rate)) / (avg_win / avg_loss)
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
        else:
            kelly_fraction = 0
        
        return {
            'average_win': avg_win,
            'average_loss': avg_loss,
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'win_rate': win_rate * 100,
            'expected_value': expected_value,
            'kelly_fraction': kelly_fraction,
            'optimal_position_size': f"{kelly_fraction * 100:.1f}%",
            'conclusion': self._generate_risk_reward_conclusion(avg_win, avg_loss, win_rate)
        }
    
    def _generate_risk_reward_conclusion(self, avg_win, avg_loss, win_rate):
        """Generate risk/reward conclusion"""
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        if rr_ratio > 2 and win_rate > 0.4:
            return "Excellent risk/reward profile. The strategy has strong positive expectancy."
        elif rr_ratio > 1.5 or win_rate > 0.6:
            return "Good risk/reward characteristics. Consider increasing position sizes on high-confidence trades."
        elif rr_ratio < 1:
            return "Poor risk/reward ratio. Focus on improving exit timing or entry selection."
        else:
            return "Moderate risk/reward profile. There's room for optimization."
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Based on P&L summary
        summary = self._generate_pnl_summary()
        if summary.get('win_rate', 0) < 40:
            recommendations.append("Low win rate detected. Consider tightening entry criteria or improving ML model accuracy.")
        
        # Based on factor attribution
        factors = self._attribute_factors()
        
        # ML score recommendations
        ml_impact = factors.get('ml_score_impact', {})
        if ml_impact.get('impact_score', 0) > 50:
            recommendations.append("ML scores are highly predictive. Consider increasing weight on high-confidence signals.")
        
        # Timing recommendations
        timing = factors.get('timing_impact', {})
        if timing.get('best_hour'):
            recommendations.append(f"Focus trading during optimal hours: {timing['best_hour']}:00-{timing['best_hour']+1}:00")
        
        # Symbol recommendations
        symbol_impact = factors.get('symbol_impact', {})
        if symbol_impact.get('top_performers'):
            recommendations.append(f"Prioritize top-performing symbols: {', '.join(symbol_impact['top_performers'][:3])}")
        
        # Risk management recommendations
        risk_reward = self._analyze_risk_reward()
        if risk_reward.get('risk_reward_ratio', 0) < 1.5:
            recommendations.append("Improve risk/reward ratio by tightening stop losses or expanding profit targets.")
        
        # Pattern-based recommendations
        patterns = self._analyze_patterns()
        if patterns.get('winning_patterns'):
            top_pattern = patterns['winning_patterns'][0] if patterns['winning_patterns'] else None
            if top_pattern:
                recommendations.append(f"Exploit winning pattern: {top_pattern['description']} (avg: {top_pattern['avg_pnl']:.2%})")
        
        return recommendations
    
    def generate_pnl_report(self, output_file: str = None) -> str:
        """Generate comprehensive P&L analysis report"""
        analysis = self.analyze_pnl_factors(self.trade_data, self.decision_data)
        
        report = []
        report.append("=" * 80)
        report.append("LAEF P&L ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now()}")
        report.append("=" * 80)
        
        # Summary section
        report.append("\n1. P&L SUMMARY")
        report.append("-" * 40)
        summary = analysis['summary']
        report.append(f"Total Trades: {summary.get('total_trades', 0)}")
        report.append(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
        report.append(f"Average P&L: {summary.get('average_pnl_pct', 0):.2%}")
        report.append(f"Total P&L: {summary.get('total_pnl_pct', 0):.2%}")
        report.append(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
        
        # Factor Attribution
        report.append("\n2. FACTOR ATTRIBUTION")
        report.append("-" * 40)
        factors = analysis['factor_attribution']
        
        # Sort factors by importance
        sorted_factors = sorted(
            factors.items(), 
            key=lambda x: x[1].get('relative_importance', 0), 
            reverse=True
        )
        
        for factor_name, factor_data in sorted_factors:
            if 'relative_importance' in factor_data:
                report.append(f"\n{factor_name.replace('_', ' ').title()}:")
                report.append(f"  Importance: {factor_data['relative_importance']:.1f}%")
                report.append(f"  Conclusion: {factor_data.get('conclusion', 'N/A')}")
        
        # Pattern Analysis
        report.append("\n3. WINNING & LOSING PATTERNS")
        report.append("-" * 40)
        patterns = analysis['pattern_analysis']
        
        if patterns.get('winning_patterns'):
            report.append("\nTop Winning Patterns:")
            for i, pattern in enumerate(patterns['winning_patterns'][:3]):
                report.append(f"  {i+1}. {pattern['description']}")
                report.append(f"     Avg P&L: {pattern['avg_pnl']:.2%}, Win Rate: {pattern['win_rate']:.1f}%")
        
        if patterns.get('losing_patterns'):
            report.append("\nTop Losing Patterns:")
            for i, pattern in enumerate(patterns['losing_patterns'][:3]):
                report.append(f"  {i+1}. {pattern['description']}")
                report.append(f"     Avg P&L: {pattern['avg_pnl']:.2%}, Win Rate: {pattern['win_rate']:.1f}%")
        
        # Risk/Reward Analysis
        report.append("\n4. RISK/REWARD ANALYSIS")
        report.append("-" * 40)
        rr = analysis['risk_reward_analysis']
        if 'error' not in rr:
            report.append(f"Average Win: {rr.get('average_win', 0):.2%}")
            report.append(f"Average Loss: {rr.get('average_loss', 0):.2%}")
            report.append(f"Risk/Reward Ratio: {rr.get('risk_reward_ratio', 0):.2f}")
            report.append(f"Expected Value per Trade: {rr.get('expected_value', 0):.3%}")
            report.append(f"Optimal Position Size (Kelly): {rr.get('optimal_position_size', 'N/A')}")
        
        # Recommendations
        report.append("\n5. RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(analysis['recommendations']):
            report.append(f"  {i+1}. {rec}")
        
        report_text = "\n".join(report)
        
        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"P&L analysis report saved to: {output_file}")
        
        return report_text