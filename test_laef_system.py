# === test_laef_system.py - System Validation Script ===

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Test all unified components
from config import STATE_FEATURES, STATE_SIZE, TRADING_THRESHOLDS
from core.indicators_unified import calculate_all_indicators, validate_indicator_data
from core.state_utils_unified import create_state_vector, extract_raw_indicators, validate_state_pipeline
from core.agent_unified import LAEFAgent
from core.fifo_portfolio import FIFOPortfolio
from core.dual_model_trading_logic import DualModelTradingEngine
from data.data_fetcher_unified import fetch_stock_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class LAEFSystemTester:
    """
    Comprehensive system tester to validate all components work together.
    """
    
    def __init__(self):
        self.test_results = {}
        self.portfolio = FIFOPortfolio(initial_cash=10000)  # Small test portfolio
        
        print("🔧 LAEF System Validation Starting...")
        print("=" * 50)
    
    def run_all_tests(self):
        """Run comprehensive system tests."""
        try:
            # Test 1: Configuration
            self.test_configuration()
            
            # Test 2: Data Pipeline
            self.test_data_pipeline()
            
            # Test 3: Indicator Calculation
            self.test_indicator_calculation()
            
            # Test 4: State Vector Creation
            self.test_state_vector_creation()
            
            # Test 5: ML Agent
            self.test_ml_agent()
            
            # Test 6: FIFO Portfolio
            self.test_fifo_portfolio()
            
            # Test 7: Trading Logic
            self.test_trading_logic()
            
            # Test 8: Full Integration
            self.test_full_integration()
            
            # Print summary
            self.print_test_summary()
            
        except Exception as e:
            logging.error(f"System test failed: {e}")
            return False
        
        return True
    
    def test_configuration(self):
        """Test that configuration is properly set up."""
        print("\n📋 Testing Configuration...")
        
        try:
            # Check required config values
            assert STATE_SIZE == 12, f"STATE_SIZE should be 12, got {STATE_SIZE}"
            assert len(STATE_FEATURES) == 12, f"STATE_FEATURES should have 12 items, got {len(STATE_FEATURES)}"
            assert 'q_buy' in TRADING_THRESHOLDS, "Missing q_buy threshold"
            assert 'rsi_oversold' in TRADING_THRESHOLDS, "Missing RSI thresholds"
            
            print("✅ Configuration test passed")
            self.test_results['configuration'] = True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results['configuration'] = False
    
    def test_data_pipeline(self):
        """Test data fetching and cleaning."""
        print("\n📊 Testing Data Pipeline...")
        
        try:
            # Test data fetching
            df = fetch_stock_data('AAPL', interval='5m', period='5d')
            
            if df is None:
                raise ValueError("Data fetch returned None")
            
            if len(df) < 20:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            print(f"✅ Data pipeline test passed - {len(df)} rows fetched")
            self.test_results['data_pipeline'] = True
            return df
            
        except Exception as e:
            print(f"❌ Data pipeline test failed: {e}")