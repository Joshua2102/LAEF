#!/usr/bin/env python3
"""
LAEF Main Trading System - Fixed and Simplified
Provides easy access to all trading functions
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LAEFMainSystem:
    def __init__(self):
        self.verify_environment()
        
    def verify_environment(self):
        """Verify all required components are available"""
        if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
            print("❌ ERROR: Alpaca API credentials not found!")
            print("Please ensure your .env file contains:")
            print("  ALPACA_API_KEY=your_key_here")
            print("  ALPACA_SECRET_KEY=your_secret_here")
            sys.exit(1)
            
    def print_banner(self):
        """Print system banner"""
        print("\n" + "="*70)
        print(" "*20 + "LAEF TRADING SYSTEM")
        print(" "*15 + "Learning-Augmented Equity Framework")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    def print_menu(self):
        """Print main menu"""
        print("SELECT TRADING MODE:\n")
        
        print("1. 📈 LIVE PAPER TRADING (Recommended)")
        print("   • Real-time trading with virtual money")
        print("   • Uses Alpaca Paper Trading API")
        print("   • Perfect for testing strategies")
        print()
        
        print("2. 🚀 AGGRESSIVE PAPER TRADING")
        print("   • Paper trading with relaxed thresholds")
        print("   • More trading opportunities")
        print("   • Higher risk/reward profile")
        print()
        
        print("3. 📊 BACKTESTING")
        print("   • Test strategies on historical data")
        print("   • No actual trading")
        print("   • Performance analysis and reports")
        print()
        
        print("4. 🧠 LIVE MARKET LEARNING")
        print("   • AI learns from real-time market")
        print("   • No trading, only learning")
        print("   • Improves prediction accuracy")
        print()
        
        print("5. ⚙️  SYSTEM STATUS")
        print("   • Check API connection")
        print("   • View current positions")
        print("   • Account information")
        print()
        
        print("6. 🚪 EXIT")
        print()
        
    def start_paper_trading(self):
        """Start standard paper trading"""
        print("\n🚀 Starting Paper Trading...")
        print("="*50)
        
        from trading.live_trader_alpaca import LAEFLiveTrader
        
        trader = LAEFLiveTrader(paper_trading=True)
        print("✅ Paper trader initialized")
        print("Starting trading loop (60 second cycles)...")
        print("Press Ctrl+C to stop\n")
        
        try:
            trader.start_trading(cycle_delay=60)
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            
    def start_aggressive_trading(self):
        """Start aggressive paper trading"""
        print("\n🚀 Starting Aggressive Paper Trading...")
        print("="*50)
        
        # Check if aggressive config exists
        config_path = 'config_profiles/aggressive_trading.json'
        if not os.path.exists(config_path):
            print("Creating aggressive trading configuration...")
            os.makedirs('config_profiles', exist_ok=True)
            
            config = {
                "name": "Aggressive Trading",
                "description": "Less restrictive thresholds for more opportunities",
                "thresholds": {
                    "q_buy": 0.15,
                    "q_sell": 0.10,
                    "ml_profit_peak": 0.15,
                    "momentum_multiplier": 1.2,
                    "momentum_threshold": 0.5,
                    "momentum_acceleration_min": 0.1,
                    "micro_profit_target": 0.10,
                    "micro_stop_loss": 0.10,
                    "volume_confirmation_min": 1.1,
                    "max_hold_minutes": 60,
                    "ai_scalp_position_pct": 0.05,
                    "max_concurrent_scalps": 6
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        print(f"Using profile: {config['name']}")
        print("Key settings:")
        print(f"  • Q-buy threshold: {config['thresholds']['q_buy']}")
        print(f"  • Momentum threshold: {config['thresholds']['momentum_threshold']}")
        print(f"  • Position size: {config['thresholds']['ai_scalp_position_pct']*100}%")
        print()
        
        from trading.live_trader_alpaca import LAEFLiveTrader
        
        trader = LAEFLiveTrader(paper_trading=True)
        
        # Apply aggressive thresholds
        trader.trading_engine.thresholds.update(config['thresholds'])
        trader.trading_engine.Q_BUY_THRESHOLD = config['thresholds']['q_buy']
        trader.trading_engine.Q_SELL_THRESHOLD = config['thresholds']['q_sell']
        trader.trading_engine.ML_CONFIDENCE_THRESHOLD = config['thresholds']['ml_profit_peak']
        
        print("✅ Aggressive settings applied")
        print("Starting trading loop (30 second cycles)...")
        print("Press Ctrl+C to stop\n")
        
        try:
            trader.start_trading(cycle_delay=30)
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            
    def start_backtesting(self):
        """Start backtesting"""
        print("\n📊 Starting Backtesting System...")
        
        from trading.backtester_unified import LAEFBacktester
        
        print("\nBacktest Options:")
        print("1. Quick test (3 months, default symbols)")
        print("2. Full test (1 year, all symbols)")
        print("3. Custom test")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            # Quick test
            print("\nRunning quick backtest...")
            backtester = LAEFBacktester(
                initial_cash=50000,
                custom_config={'start_date': '2024-10-01', 'end_date': '2024-12-31'}
            )
            results = backtester.run_backtest(
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                use_smart_selection=False
            )
            
        elif choice == '2':
            # Full test
            print("\nRunning full backtest (this may take a while)...")
            backtester = LAEFBacktester(initial_cash=100000)
            results = backtester.run_backtest(use_smart_selection=True)
            
        else:
            # Custom test
            symbols_input = input("Enter symbols (comma-separated): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            if not symbols:
                print("No symbols provided. Using defaults.")
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                
            start_date = input("Start date (YYYY-MM-DD) [2024-01-01]: ").strip() or '2024-01-01'
            end_date = input("End date (YYYY-MM-DD) [2024-12-31]: ").strip() or '2024-12-31'
            
            backtester = LAEFBacktester(
                initial_cash=100000,
                custom_config={'start_date': start_date, 'end_date': end_date}
            )
            results = backtester.run_backtest(symbols=symbols, use_smart_selection=False)
            
        if results:
            print("\n✅ Backtest completed!")
            print(f"Check logs folder for detailed results")
            
    def start_live_learning(self):
        """Start live market learning"""
        print("\n🧠 Starting Live Market Learning...")
        
        from training.live_market_learner import LiveMarketLearner
        
        print("\nEnter symbols to monitor (comma-separated)")
        print("Example: AAPL,MSFT,GOOGL,TSLA")
        print("Press Enter for defaults: ")
        
        user_input = input("Symbols: ").strip()
        
        if user_input:
            symbols = [s.strip().upper() for s in user_input.split(',') if s.strip()]
        else:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
            
        print(f"\n✅ Monitoring {len(symbols)} symbols: {', '.join(symbols)}")
        
        learner = LiveMarketLearner(symbols=symbols)
        
        print("\nLearning Options:")
        print("1. Start continuous monitoring")
        print("2. Single learning cycle")
        print("3. Review learning statistics")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("\n🔄 Starting continuous monitoring...")
            print("Press Ctrl+C to stop\n")
            learner.start_continuous_monitoring()
            
        elif choice == '2':
            print("\n🔄 Running single learning cycle...")
            learner.monitor_and_learn_cycle()
            
        else:
            print("\n📊 Learning Statistics:")
            stats = learner.get_learning_statistics()
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
    def check_system_status(self):
        """Check system status and connection"""
        print("\n⚙️  System Status Check")
        print("="*50)
        
        try:
            from trading.alpaca_integration import AlpacaTrader
            
            trader = AlpacaTrader(paper_trading=True)
            
            # Account info
            account = trader.get_account_info()
            print("\n📊 Account Information:")
            print(f"  • Cash: ${account.get('cash', 0):,.2f}")
            print(f"  • Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"  • Buying Power: ${account.get('buying_power', 0):,.2f}")
            
            # Market status
            if trader.is_market_open():
                print(f"\n🟢 Market Status: OPEN")
            else:
                print(f"\n🔴 Market Status: CLOSED")
                
            # Positions
            positions = trader.get_all_positions()
            print(f"\n📈 Active Positions: {len(positions)}")
            
            if positions:
                for pos in positions:
                    print(f"  • {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
                    print(f"    P&L: ${pos['unrealized_pl']:+.2f} ({pos['unrealized_plpc']*100:+.1f}%)")
                    
            print("\n✅ System check complete")
            
        except Exception as e:
            print(f"\n❌ System check failed: {e}")
            
    def run(self):
        """Main system loop"""
        while True:
            try:
                self.print_banner()
                self.print_menu()
                
                choice = input("Select option (1-6): ").strip()
                
                if choice == '1':
                    self.start_paper_trading()
                elif choice == '2':
                    self.start_aggressive_trading()
                elif choice == '3':
                    self.start_backtesting()
                elif choice == '4':
                    self.start_live_learning()
                elif choice == '5':
                    self.check_system_status()
                elif choice == '6':
                    print("\n👋 Exiting LAEF Trading System...")
                    break
                else:
                    print("\n❌ Invalid choice. Please select 1-6.")
                    
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting LAEF Trading System...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        system = LAEFMainSystem()
        system.run()
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()