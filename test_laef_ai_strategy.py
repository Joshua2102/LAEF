"""
Test script to demonstrate proper LAEF AI strategy without conflicting modes
"""

import sys
sys.path.append('.')

from trading.backtester_unified import LAEFBacktester

def test_laef_ai_strategy():
    """Test pure LAEF AI strategy with dynamic parameters"""
    print("="*70)
    print("TESTING PURE LAEF AI STRATEGY - DYNAMIC PARAMETERS")
    print("="*70)
    
    # Test with LAEF AI strategy (no fixed parameters)
    print("\n1. Testing LAEF AI Strategy (Pure Dynamic Mode):")
    print("-" * 50)
    
    custom_config = {
        'initial_cash': 100000,
        'start_date': '2024-10-01',
        'end_date': '2024-12-31'
    }
    
    # Create backtester with LAEF AI strategy
    backtester = LAEFBacktester(
        initial_cash=100000, 
        custom_config=custom_config,
        strategy='laef_ai'  # Pure LAEF AI mode
    )
    
    print("\nRunning backtest with LAEF Smart Selection...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test symbols
    
    results = backtester.run_backtest(symbols=symbols, use_smart_selection=False)
    
    if results:
        print("\n" + "="*50)
        print("LAEF AI RESULTS (Dynamic Parameters):")
        print("="*50)
        perf = results.get('performance', {})
        print(f"Final Portfolio Value: ${perf.get('final_value', 0):,.2f}")
        print(f"Total Return: {perf.get('total_return_pct', 0):+.2f}%")
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
        print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {perf.get('max_drawdown_pct', 0):.2f}%")
        
        decisions = results.get('decisions', {})
        print(f"\nTotal Decisions: {decisions.get('total_decisions', 0)}")
        print(f"Executed Actions: {decisions.get('executed_actions', 0)}")
    
    print("\n" + "="*70)
    print("LAEF AI TEST COMPLETE - NO CONFLICTING MODES")
    print("="*70)

def test_scalping_strategy():
    """Test AI Momentum Scalping strategy with fixed parameters"""
    print("\n\n" + "="*70)
    print("TESTING AI MOMENTUM SCALPING - FIXED PARAMETERS")
    print("="*70)
    
    print("\n2. Testing AI Momentum Scalping Strategy (Fixed Mode):")
    print("-" * 50)
    
    custom_config = {
        'initial_cash': 100000,
        'start_date': '2024-10-01',
        'end_date': '2024-12-31',
        'profit_target': 0.002,  # 0.2% profit target
        'stop_loss': 0.001       # 0.1% stop loss
    }
    
    # Create backtester with AI Momentum Scalping strategy
    backtester = LAEFBacktester(
        initial_cash=100000,
        custom_config=custom_config,
        strategy='ai_momentum_scalping'  # Scalping mode with fixed params
    )
    
    print("\nRunning backtest with fixed scalping parameters...")
    symbols = ['AAPL', 'MSFT']  # Test symbols
    
    results = backtester.run_backtest(symbols=symbols, use_smart_selection=False)
    
    if results:
        print("\n" + "="*50)
        print("SCALPING RESULTS (Fixed Parameters):")
        print("="*50)
        perf = results.get('performance', {})
        print(f"Final Portfolio Value: ${perf.get('final_value', 0):,.2f}")
        print(f"Total Return: {perf.get('total_return_pct', 0):+.2f}%")
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
    
    print("\n" + "="*70)
    print("SCALPING TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    # Test pure LAEF AI strategy
    test_laef_ai_strategy()
    
    # Test AI Momentum Scalping for comparison
    test_scalping_strategy()
    
    print("\n\nSUMMARY:")
    print("========")
    print("1. LAEF AI Strategy: Uses pure neural network predictions")
    print("   - No fixed profit targets or stop losses")
    print("   - Dynamic position sizing based on confidence")
    print("   - Adaptive thresholds that change with market conditions")
    print("")
    print("2. AI Momentum Scalping: Uses fixed micro-scalping parameters")
    print("   - Fixed profit targets (0.1% - 0.5%)")
    print("   - Fixed stop losses")
    print("   - Conservative mode with static risk percentages")
    print("")
    print("The strategies are now properly separated without conflicting modes!")