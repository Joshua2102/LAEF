"""
Backtest Tracking and Analysis System
Tracks configurations, logic, and results for each backtest run
"""

import json
import os
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any
import hashlib

class BacktestTracker:
    def __init__(self, tracking_file="backtest_tracking.json"):
        self.tracking_file = f"logs/{tracking_file}"
        self.current_run = None
        self.load_tracking_data()
        
    def load_tracking_data(self):
        """Load existing tracking data or create new"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                self.tracking_data = json.load(f)
        else:
            self.tracking_data = {
                "runs": [],
                "summary_stats": {},
                "best_configs": {}
            }
    
    def save_tracking_data(self):
        """Save tracking data to file"""
        os.makedirs("logs", exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def start_new_run(self, config: Dict[str, Any], description: str = ""):
        """Start tracking a new backtest run"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create config hash for comparison
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        self.current_run = {
            "run_id": run_id,
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "profile_info": {
                "profile_name": config.get("_profile_name", "Unknown Profile"),
                "engine_type": config.get("engine_type", "dual_model_engine"),
                "engine_name": config.get("engine_name", "LAEF Dual-Model Trading Engine"),
                "engine_description": config.get("engine_description", "Q-Learning + ML confidence system")
            },
            "configuration": {
                "thresholds": {
                    "q_buy": config.get("q_buy", "N/A"),
                    "q_sell": config.get("q_sell", "N/A"),
                    "ml_profit_peak": config.get("ml_profit_peak", "N/A"),
                    "rsi_oversold": config.get("rsi_oversold", "N/A"),
                    "rsi_overbought": config.get("rsi_overbought", "N/A"),
                    "profit_target": config.get("profit_target", "N/A"),
                    "stop_loss": config.get("stop_loss", "N/A")
                },
                "risk_management": {
                    "initial_cash": config.get("initial_cash", 100000),
                    "risk_per_trade": config.get("risk_per_trade", "N/A"),
                    "max_position": config.get("max_position", "N/A"),
                    "cooldown_minutes": config.get("cooldown_minutes", "N/A")
                },
                "time_params": {
                    "start_date": config.get("start_date", "N/A"),
                    "end_date": config.get("end_date", "N/A"),
                    "max_hold_minutes": config.get("max_hold_minutes", "N/A")
                }
            },
            "trading_logic": {
                "engine": config.get("engine_name", "LAEF Dual-Model (Q-Learning + ML)"),
                "engine_type": config.get("engine_type", "dual_model_engine"),
                "buy_logic": self._describe_buy_logic(config),
                "sell_logic": self._describe_sell_logic(config),
                "risk_logic": self._describe_risk_logic(config)
            },
            "results": {},
            "analysis": {}
        }
        
        return run_id
    
    def _describe_buy_logic(self, config):
        """Generate human-readable buy logic description"""
        q_buy = config.get("q_buy", 0.55)
        ml_threshold = config.get("ml_profit_peak", 0.55)
        rsi_oversold = config.get("rsi_oversold", 30)
        
        return {
            "summary": f"Buy when Q-value > {q_buy} AND ML confidence > {ml_threshold}",
            "conditions": [
                f"Q-Learning buy signal exceeds {q_buy}",
                f"ML model predicts profit with confidence > {ml_threshold}",
                f"RSI < {rsi_oversold} (oversold condition) enhances signal",
                "Volume confirmation required",
                "Risk management allows position"
            ],
            "logic_type": "Conservative Dual-Model" if q_buy > 0.5 else "Aggressive Dual-Model"
        }
    
    def _describe_sell_logic(self, config):
        """Generate human-readable sell logic description"""
        q_sell = config.get("q_sell", 0.35)
        profit_target = config.get("profit_target", 0.04)
        stop_loss = config.get("stop_loss", 0.96)
        rsi_overbought = config.get("rsi_overbought", 70)
        
        return {
            "summary": f"Sell on profit target {profit_target*100}% or stop loss {(1-stop_loss)*100}%",
            "conditions": [
                f"Profit target reached: {profit_target*100}%",
                f"Stop loss triggered: {(1-stop_loss)*100}% loss",
                f"Q-Learning sell signal > {q_sell}",
                f"RSI > {rsi_overbought} (overbought) suggests exit",
                "Time-based exits if configured"
            ],
            "exit_priority": "Stop Loss > Profit Target > Q-Signal > Time Exit"
        }
    
    def _describe_risk_logic(self, config):
        """Generate human-readable risk management description"""
        risk_per_trade = config.get("risk_per_trade", 0.025)
        max_position = config.get("max_position", 0.15)
        
        return {
            "summary": f"Max {risk_per_trade*100}% risk per trade, {max_position*100}% position size",
            "rules": [
                f"Maximum {risk_per_trade*100}% of capital at risk per trade",
                f"Maximum {max_position*100}% of portfolio in single position",
                "Position sizing based on volatility",
                "Cooldown period between trades",
                "Daily loss limits if configured"
            ]
        }
    
    def record_results(self, results: Dict[str, Any]):
        """Record backtest results"""
        if not self.current_run:
            print("No active run to record results")
            return
            
        self.current_run["results"] = {
            "performance": {
                "total_return": results.get("total_return_pct", 0),
                "final_value": results.get("final_value", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "win_rate": results.get("win_rate", 0)
            },
            "trading_stats": {
                "total_trades": results.get("total_trades", 0),
                "winning_trades": results.get("winning_trades", 0),
                "losing_trades": results.get("losing_trades", 0),
                "avg_win": results.get("avg_win", 0),
                "avg_loss": results.get("avg_loss", 0),
                "profit_factor": results.get("profit_factor", 0)
            },
            "execution": {
                "symbols_traded": results.get("symbols_traded", []),
                "execution_rate": results.get("execution_rate", 0),
                "avg_hold_time": results.get("avg_hold_time", "N/A")
            }
        }
        
        # Generate analysis
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze results in context of configuration"""
        if not self.current_run or not self.current_run.get("results"):
            return
            
        config = self.current_run["configuration"]["thresholds"]
        results = self.current_run["results"]["performance"]
        
        analysis = {
            "threshold_impact": self._analyze_threshold_impact(config, results),
            "performance_rating": self._rate_performance(results),
            "recommendations": self._generate_recommendations(config, results),
            "comparison": self._compare_to_previous_runs()
        }
        
        self.current_run["analysis"] = analysis
    
    def _analyze_threshold_impact(self, config, results):
        """Analyze how thresholds affected results"""
        win_rate = results.get("win_rate", 0)
        total_return = results.get("total_return", 0)
        
        impact = []
        
        # Q-value thresholds impact
        if config.get("q_buy", 0) > 0.5:
            if win_rate > 50:
                impact.append("Conservative Q-buy threshold resulted in higher win rate")
            else:
                impact.append("High Q-buy threshold may be too restrictive")
        else:
            if total_return > 10:
                impact.append("Aggressive Q-buy threshold captured more opportunities")
            else:
                impact.append("Low Q-buy threshold may need refinement")
        
        # Profit/Loss targets impact
        profit_target = config.get("profit_target", 0.04)
        if profit_target < 0.03:
            impact.append("Small profit targets led to frequent but small wins")
        elif profit_target > 0.05:
            impact.append("Large profit targets resulted in fewer but bigger wins")
            
        return impact
    
    def _rate_performance(self, results):
        """Rate the performance based on multiple metrics"""
        score = 0
        rating_details = []
        
        # Return rating
        total_return = results.get("total_return", 0)
        if total_return > 20:
            score += 3
            rating_details.append("Excellent returns (>20%)")
        elif total_return > 10:
            score += 2
            rating_details.append("Good returns (10-20%)")
        elif total_return > 0:
            score += 1
            rating_details.append("Positive returns")
        else:
            rating_details.append("Negative returns")
            
        # Win rate rating
        win_rate = results.get("win_rate", 0)
        if win_rate > 60:
            score += 2
            rating_details.append("High win rate (>60%)")
        elif win_rate > 50:
            score += 1
            rating_details.append("Positive win rate")
            
        # Risk-adjusted rating
        sharpe = results.get("sharpe_ratio", 0)
        if sharpe > 1.5:
            score += 2
            rating_details.append("Excellent risk-adjusted returns")
        elif sharpe > 1:
            score += 1
            rating_details.append("Good risk-adjusted returns")
            
        # Overall rating
        if score >= 6:
            overall = "EXCELLENT"
        elif score >= 4:
            overall = "GOOD"
        elif score >= 2:
            overall = "AVERAGE"
        else:
            overall = "POOR"
            
        return {
            "overall": overall,
            "score": f"{score}/7",
            "details": rating_details
        }
    
    def _generate_recommendations(self, config, results):
        """Generate recommendations based on results"""
        recommendations = []
        
        win_rate = results.get("win_rate", 0)
        total_return = results.get("total_return", 0)
        max_drawdown = abs(results.get("max_drawdown", 0))
        
        # Win rate recommendations
        if win_rate < 45:
            recommendations.append("Consider increasing Q-buy threshold for better entry signals")
            recommendations.append("Review ML confidence threshold - may be too low")
        elif win_rate > 65:
            recommendations.append("High win rate but check if profit targets are too small")
            
        # Return recommendations
        if total_return < 5:
            recommendations.append("Increase profit targets to capture larger moves")
            recommendations.append("Consider reducing Q-buy threshold to find more opportunities")
        
        # Risk recommendations
        if max_drawdown > 15:
            recommendations.append("Tighten stop loss to reduce drawdown")
            recommendations.append("Consider reducing position sizes")
            
        # Balance recommendations
        if win_rate > 60 and total_return < 10:
            recommendations.append("Good accuracy but small profits - increase profit targets")
        elif win_rate < 50 and total_return > 15:
            recommendations.append("High returns but low accuracy - could be luck, needs validation")
            
        return recommendations
    
    def _compare_to_previous_runs(self):
        """Compare current run to previous runs"""
        if not self.tracking_data["runs"]:
            return {"message": "First run - no comparison available"}
            
        current_return = self.current_run["results"]["performance"].get("total_return", 0)
        
        # Get returns from previous runs
        previous_returns = [
            run["results"]["performance"].get("total_return", 0) 
            for run in self.tracking_data["runs"]
            if "results" in run and "performance" in run["results"]
        ]
        
        if not previous_returns:
            return {"message": "No previous results for comparison"}
            
        avg_previous = sum(previous_returns) / len(previous_returns)
        best_previous = max(previous_returns)
        
        comparison = {
            "vs_average": f"{current_return - avg_previous:+.2f}% vs avg",
            "vs_best": f"{current_return - best_previous:+.2f}% vs best",
            "ranking": f"#{len([r for r in previous_returns if r > current_return]) + 1} of {len(previous_returns) + 1}",
            "improvement": current_return > avg_previous
        }
        
        return comparison
    
    def finalize_run(self):
        """Finalize and save the current run"""
        if self.current_run:
            self.tracking_data["runs"].append(self.current_run)
            self._update_summary_stats()
            self._update_best_configs()
            self.save_tracking_data()
            
            # Generate report
            self.generate_run_report()
            
            self.current_run = None
    
    def _update_summary_stats(self):
        """Update overall summary statistics"""
        all_returns = [
            run["results"]["performance"].get("total_return", 0)
            for run in self.tracking_data["runs"]
            if "results" in run
        ]
        
        if all_returns:
            self.tracking_data["summary_stats"] = {
                "total_runs": len(self.tracking_data["runs"]),
                "avg_return": sum(all_returns) / len(all_returns),
                "best_return": max(all_returns),
                "worst_return": min(all_returns),
                "positive_runs": len([r for r in all_returns if r > 0]),
                "success_rate": len([r for r in all_returns if r > 0]) / len(all_returns) * 100
            }
    
    def _update_best_configs(self):
        """Track best performing configurations"""
        if not self.tracking_data["runs"]:
            return
            
        # Find best by different metrics
        runs_with_results = [r for r in self.tracking_data["runs"] if "results" in r]
        
        if runs_with_results:
            # Best by return
            best_return_run = max(runs_with_results, 
                                key=lambda x: x["results"]["performance"].get("total_return", -999))
            
            # Best by win rate
            best_winrate_run = max(runs_with_results,
                                 key=lambda x: x["results"]["performance"].get("win_rate", 0))
            
            # Best by Sharpe
            best_sharpe_run = max(runs_with_results,
                                key=lambda x: x["results"]["performance"].get("sharpe_ratio", -999))
            
            self.tracking_data["best_configs"] = {
                "best_return": {
                    "run_id": best_return_run["run_id"],
                    "config_hash": best_return_run["config_hash"],
                    "return": best_return_run["results"]["performance"]["total_return"],
                    "config": best_return_run["configuration"]
                },
                "best_winrate": {
                    "run_id": best_winrate_run["run_id"],
                    "config_hash": best_winrate_run["config_hash"],
                    "win_rate": best_winrate_run["results"]["performance"]["win_rate"],
                    "config": best_winrate_run["configuration"]
                },
                "best_sharpe": {
                    "run_id": best_sharpe_run["run_id"],
                    "config_hash": best_sharpe_run["config_hash"],
                    "sharpe": best_sharpe_run["results"]["performance"]["sharpe_ratio"],
                    "config": best_sharpe_run["configuration"]
                }
            }
    
    def generate_run_report(self):
        """Generate a detailed report for the current run"""
        if not self.current_run:
            print("No run to report")
            return
            
        report_file = f"logs/backtest_report_{self.current_run['run_id']}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BACKTEST REPORT - {self.current_run['run_id']}\n")
            f.write("="*80 + "\n\n")
            
            # Profile and Engine Information
            f.write("PROFILE & ENGINE INFORMATION:\n")
            f.write("-"*40 + "\n")
            profile_info = self.current_run["profile_info"]
            f.write(f"Profile Name: {profile_info['profile_name']}\n")
            f.write(f"Engine Type: {profile_info['engine_type']}\n")
            f.write(f"Engine Name: {profile_info['engine_name']}\n")
            f.write(f"Engine Description: {profile_info['engine_description']}\n")
            
            # Configuration section
            f.write("\n\nCONFIGURATION USED:\n")
            f.write("-"*40 + "\n")
            config = self.current_run["configuration"]
            f.write(f"Thresholds:\n")
            for key, value in config["thresholds"].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nRisk Management:\n")
            for key, value in config["risk_management"].items():
                f.write(f"  {key}: {value}\n")
            
            # Trading logic section
            f.write("\n\nTRADING LOGIC:\n")
            f.write("-"*40 + "\n")
            logic = self.current_run["trading_logic"]
            f.write(f"Engine: {logic['engine']}\n")
            f.write(f"Engine Type: {logic['engine_type']}\n\n")
            
            f.write("Buy Logic:\n")
            f.write(f"  {logic['buy_logic']['summary']}\n")
            for condition in logic['buy_logic']['conditions']:
                f.write(f"  - {condition}\n")
                
            f.write("\nSell Logic:\n")
            f.write(f"  {logic['sell_logic']['summary']}\n")
            for condition in logic['sell_logic']['conditions']:
                f.write(f"  - {condition}\n")
            
            # Results section
            if "results" in self.current_run:
                f.write("\n\nRESULTS:\n")
                f.write("-"*40 + "\n")
                results = self.current_run["results"]
                
                f.write("Performance:\n")
                for key, value in results["performance"].items():
                    f.write(f"  {key}: {value}\n")
                    
                f.write("\nTrading Statistics:\n")
                for key, value in results["trading_stats"].items():
                    f.write(f"  {key}: {value}\n")
            
            # Analysis section
            if "analysis" in self.current_run:
                f.write("\n\nANALYSIS:\n")
                f.write("-"*40 + "\n")
                analysis = self.current_run["analysis"]
                
                f.write(f"Performance Rating: {analysis['performance_rating']['overall']}\n")
                f.write(f"Score: {analysis['performance_rating']['score']}\n")
                for detail in analysis['performance_rating']['details']:
                    f.write(f"  - {detail}\n")
                
                f.write("\nThreshold Impact:\n")
                for impact in analysis['threshold_impact']:
                    f.write(f"  - {impact}\n")
                    
                f.write("\nRecommendations:\n")
                for rec in analysis['recommendations']:
                    f.write(f"  - {rec}\n")
                    
                if "comparison" in analysis:
                    f.write("\nComparison to Previous Runs:\n")
                    for key, value in analysis['comparison'].items():
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Report saved to: {report_file}")
    
    def show_summary(self):
        """Display summary of all tracking data"""
        print("\n" + "="*60)
        print("BACKTEST TRACKING SUMMARY")
        print("="*60)
        
        if "summary_stats" in self.tracking_data and self.tracking_data["summary_stats"]:
            stats = self.tracking_data["summary_stats"]
            print(f"\nTotal Runs: {stats.get('total_runs', 0)}")
            print(f"Average Return: {stats.get('avg_return', 0):.2f}%")
            print(f"Best Return: {stats.get('best_return', 0):.2f}%")
            print(f"Worst Return: {stats.get('worst_return', 0):.2f}%")
            print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        
        if "best_configs" in self.tracking_data and self.tracking_data["best_configs"]:
            print("\nBEST CONFIGURATIONS:")
            print("-"*40)
            
            if "best_return" in self.tracking_data["best_configs"]:
                best = self.tracking_data["best_configs"]["best_return"]
                print(f"\nHighest Return: {best['return']:.2f}%")
                print(f"  Run ID: {best['run_id']}")
                print(f"  Q-Buy: {best['config']['thresholds']['q_buy']}")
                print(f"  Q-Sell: {best['config']['thresholds']['q_sell']}")
                
    def export_to_csv(self):
        """Export tracking data to CSV for analysis"""
        if not self.tracking_data["runs"]:
            print("No data to export")
            return
            
        rows = []
        for run in self.tracking_data["runs"]:
            if "results" not in run:
                continue
                
            row = {
                "run_id": run["run_id"],
                "timestamp": run["timestamp"],
                "config_hash": run["config_hash"],
                "q_buy": run["configuration"]["thresholds"]["q_buy"],
                "q_sell": run["configuration"]["thresholds"]["q_sell"],
                "profit_target": run["configuration"]["thresholds"]["profit_target"],
                "stop_loss": run["configuration"]["thresholds"]["stop_loss"],
                "total_return": run["results"]["performance"]["total_return"],
                "win_rate": run["results"]["performance"]["win_rate"],
                "total_trades": run["results"]["trading_stats"]["total_trades"],
                "sharpe_ratio": run["results"]["performance"].get("sharpe_ratio", 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = "logs/backtest_tracking_export.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data exported to: {csv_file}")

# Integration helper for backtester
def setup_tracking(config, description=""):
    """Setup tracking for a backtest run"""
    tracker = BacktestTracker()
    run_id = tracker.start_new_run(config, description)
    return tracker, run_id

def record_and_analyze(tracker, results):
    """Record results and perform analysis"""
    tracker.record_results(results)
    tracker.finalize_run()
    tracker.show_summary()

if __name__ == "__main__":
    # Demo/Test
    tracker = BacktestTracker()
    
    # Example configuration
    test_config = {
        "q_buy": 0.55,
        "q_sell": 0.35,
        "ml_profit_peak": 0.55,
        "profit_target": 0.04,
        "stop_loss": 0.96,
        "initial_cash": 100000,
        "risk_per_trade": 0.025,
        "max_position": 0.15
    }
    
    # Start tracking
    run_id = tracker.start_new_run(test_config, "Test run with conservative settings")
    
    # Simulate results
    test_results = {
        "total_return_pct": 15.5,
        "final_value": 115500,
        "win_rate": 58,
        "total_trades": 150,
        "winning_trades": 87,
        "losing_trades": 63
    }
    
    # Record and analyze
    tracker.record_results(test_results)
    tracker.finalize_run()
    tracker.show_summary()