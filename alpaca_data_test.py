# === alpaca_data_test.py - Test Alpaca Data Integration ===

import os
import sys
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def test_alpaca_data_integration():
    """Comprehensive test of Alpaca data integration."""
    
    print("🔧 Testing LAEF Alpaca Data Integration")
    print("=" * 60)
    
    # Test 1: Environment Variables
    print("\n1. Testing Environment Variables...")
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if api_key and secret_key:
        print("   ✅ Alpaca API keys found")
        print(f"   🔑 API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
    else:
        print("   ❌ Alpaca API keys missing!")
        print("   📝 Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
        return False
    
    # Test 2: Import Dependencies
    print("\n2. Testing Dependencies...")
    try:
        import alpaca_trade_api as tradeapi
        print("   ✅ alpaca-trade-api imported successfully")
        print(f"   📦 Version: {tradeapi.__version__}")
    except ImportError:
        print("   ❌ alpaca-trade-api not installed!")
        print("   📝 Run: pip install alpaca-trade-api")
        return False
    
    try:
        from ..config import (
            DATA_SOURCE_SETTINGS, ALPACA_DATA_CONFIG, 
            MARKET_SCHEDULE, get_current_data_source
        )
        print("   ✅ Config imports successful")
    except ImportError as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    # Test 3: Alpaca API Connection
    print("\n3. Testing Alpaca API Connection...")
    try:
        # Test trading API
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )
        
        account = api.get_account()
        print("   ✅ Trading API connection successful")
        print(f"   💰 Account Status: {account.status}")
        print(f"   💵 Buying Power: ${float(account.buying_power):,.2f}")
        
    except Exception as e:
        print(f"   ❌ Trading API connection failed: {e}")
        return False
    
    # Test 4: Alpaca Data API
    print("\n4. Testing Alpaca Data API...")
    try:
        # Test data API
        data_api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=ALPACA_DATA_CONFIG["base_url"],
            api_version=ALPACA_DATA_CONFIG["api_version"]
        )
        
        # Test simple data request
        from alpaca_trade_api.rest import TimeFrame
        bars = data_api.get_bars(
            "AAPL",
            TimeFrame.Day,
            start=(datetime.now() - timedelta(days=5)),
            end=datetime.now(),
            limit=5
        ).df
        
        if not bars.empty:
            print("   ✅ Data API connection successful")
            print(f"   📊 Retrieved {len(bars)} bars for AAPL")
            print(f"   📈 Latest close: ${bars['close'].iloc[-1]:.2f}")
        else:
            print("   ⚠️  Data API connected but no data returned")
            
    except Exception as e:
        print(f"   ❌ Data API connection failed: {e}")
        print(f"   💡 This might be due to subscription limits or weekend data availability")
    
    # Test 5: Enhanced Data Fetcher
    print("\n5. Testing Enhanced Data Fetcher...")
    try:
        # Import the enhanced data fetcher
        sys.path.insert(0, '.')
        from data.data_fetcher_unified import MultiSourceDataFetcher, test_data_sources
        
        print("   ✅ Enhanced data fetcher imported")
        
        # Test data fetcher initialization
        fetcher = MultiSourceDataFetcher()
        available_sources = fetcher._get_available_sources()
        print(f"   📡 Available sources: {', '.join(available_sources)}")
        
        # Test source selection
        current_source = get_current_data_source()
        print(f"   🎯 Current optimal source: {current_source.upper()}")
        
    except Exception as e:
        print(f"   ❌ Enhanced data fetcher test failed: {e}")
        print(f"   💡 Make sure the enhanced data_fetcher_unified.py is in your directory")
    
    # Test 6: Sample Data Fetch
    print("\n6. Testing Sample Data Fetch...")
    try:
        from data.data_fetcher_unified import fetch_stock_data
        
        print("   📊 Fetching AAPL data...")
        df = fetch_stock_data("AAPL", interval="1d", period="5d")
        
        if df is not None:
            print(f"   ✅ Data fetch successful: {len(df)} rows")
            print(f"   📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            print(f"   📊 Has indicators: {set(df.columns) >= {'rsi', 'macd', 'sma20'}}")
        else:
            print("   ❌ Data fetch failed")
            
    except Exception as e:
        print(f"   ❌ Sample data fetch failed: {e}")
    
    # Test 7: Configuration Validation
    print("\n7. Testing Configuration...")
    try:
        from ..config import validate_config, print_config_summary
        
        errors, warnings = validate_config()
        
        if not errors:
            print("   ✅ Configuration validation passed")
        else:
            print("   ⚠️  Configuration errors found:")
            for error in errors:
                print(f"      - {error}")
        
        if warnings:
            print("   ⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"      - {warning}")
                
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
    
    # Test 8: Market Hours Detection
    print("\n8. Testing Market Hours Detection...")
    try:
        from datetime import datetime
        import pytz
        
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        print(f"   🕐 Current time (ET): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   📅 Weekday: {now.strftime('%A')}")
        
        # Test market status
        if now.weekday() < 5 and 9 <= now.hour <= 16:
            print("   🟢 Market should be open")
        else:
            print("   🔴 Market should be closed")
        
        print(f"   🎯 Recommended data source: {get_current_data_source().upper()}")
        
    except Exception as e:
        print(f"   ❌ Market hours detection failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Alpaca Data Integration Test Complete!")
    print("\n💡 Next Steps:")
    print("   1. Run a backtest to test data fetching during market hours")
    print("   2. Try running: python backtester_unified.py")
    print("   3. Check logs/ directory for detailed execution logs")
    
    return True

def test_weekend_backtesting():
    """Test backtesting specifically during weekend/market closed hours."""
    
    print("\n🔧 Testing Weekend/After-Hours Backtesting")
    print("=" * 50)
    
    try:
        from trading.backtester_unified import LAEFBacktester
        
        print("📊 Starting small backtest (3 symbols, 5 days)...")
        
        backtester = LAEFBacktester(initial_cash=10000)
        
        # Test with a few symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        results = backtester.run_backtest(
            symbols=test_symbols,
            use_smart_selection=False  # Disable smart selection for quick test
        )
        
        if results:
            print("   ✅ Backtest completed successfully!")
            print(f"   📈 Symbols processed: {len(results.get('symbols_processed', []))}")
            
            if 'performance' in results:
                perf = results['performance']
                print(f"   💰 Return: {perf.get('total_return_pct', 0):+.2f}%")
                print(f"   📊 Trades: {perf.get('total_trades', 0)}")
        else:
            print("   ❌ Backtest failed - check logs for details")
            
    except Exception as e:
        print(f"   ❌ Weekend backtesting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run basic integration test
    success = test_alpaca_data_integration()
    
    if success:
        # If basic test passes, try weekend backtesting
        test_weekend_backtesting()
    else:
        print("\n❌ Basic integration test failed. Please fix issues before proceeding.")
