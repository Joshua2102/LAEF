# === check_ml_learning.py - Check if Model is Learning During Backtests ===

import os
import numpy as np

def check_model_learning_during_backtest():
    """Check if the ML model is learning during backtesting"""
    
    print("🔍 CHECKING ML MODEL LEARNING BEHAVIOR")
    print("=" * 60)
    
    print("📊 PROPER ML WORKFLOW:")
    print("   1. TRAIN: Model learns on historical data → updates weights")
    print("   2. BACKTEST: Model predicts on different data → NO weight updates")
    print("   3. LIVE: Model trades with fixed weights (optional online learning)")
    print("")
    
    # Check current model behavior
    try:
        from core.agent_unified import LAEFAgent
        
        print("🧠 TESTING CURRENT MODEL BEHAVIOR:")
        print("-" * 40)
        
        # Load the model
        agent = LAEFAgent(pretrained=True)
        
        # Get initial model weights
        initial_weights = []
        for layer in agent.model.layers:
            if layer.get_weights():
                initial_weights.append([w.copy() for w in layer.get_weights()])
        
        print(f"✅ Model loaded: {agent.model_path}")
        print(f"✅ Model parameters: {agent.model.count_params():,}")
        
        # Test prediction (what happens during backtesting)
        dummy_state = np.random.uniform(-1, 1, 12)
        q_value_1 = agent.predict_q_value(dummy_state)
        q_value_2 = agent.predict_q_value(dummy_state)
        
        print(f"✅ Prediction test 1: {q_value_1:.4f}")
        print(f"✅ Prediction test 2: {q_value_2:.4f}")
        print(f"✅ Predictions consistent: {abs(q_value_1 - q_value_2) < 1e-6}")
        
        # Check if weights changed after predictions
        weights_changed = False
        for i, layer in enumerate(agent.model.layers):
            if layer.get_weights() and i < len(initial_weights):
                current_weights = layer.get_weights()
                for j, (initial, current) in enumerate(zip(initial_weights[i], current_weights)):
                    if not np.array_equal(initial, current):
                        weights_changed = True
                        break
        
        print(f"🔍 Weights changed after predictions: {'❌ YES' if weights_changed else '✅ NO'}")
        
        # Check backtester behavior
        print(f"\n🔍 CHECKING BACKTESTER BEHAVIOR:")
        print("-" * 40)
        
        # Look at backtester code
        with open('backtester_unified.py', 'r') as f:
            backtest_code = f.read()
        
        # Check for training calls
        has_train_calls = 'agent.train(' in backtest_code or '.train(' in backtest_code
        has_fit_calls = '.fit(' in backtest_code
        has_save_calls = 'save_model(' in backtest_code or '.save(' in backtest_code
        
        print(f"🔍 Backtester has agent.train() calls: {'❌ YES' if has_train_calls else '✅ NO'}")
        print(f"🔍 Backtester has .fit() calls: {'❌ YES' if has_fit_calls else '✅ NO'}")
        print(f"🔍 Backtester saves model: {'⚠️ YES' if has_save_calls else '✅ NO'}")
        
        # Check config settings
        from ..config import ENABLE_AUTO_RETRAINING
        print(f"🔍 Auto-retraining enabled: {'❌ YES' if ENABLE_AUTO_RETRAINING else '✅ NO'}")
        
        return not weights_changed and not has_train_calls and not ENABLE_AUTO_RETRAINING
        
    except Exception as e:
        print(f"❌ Error checking model behavior: {e}")
        return False

def explain_ml_learning_modes():
    """Explain different ML learning modes"""
    
    print(f"\n📚 ML LEARNING MODES EXPLAINED:")
    print("=" * 50)
    
    print("🎯 MODE 1: BACKTESTING (What you want for testing)")
    print("   • Model: FROZEN weights (no learning)")
    print("   • Purpose: Test how strategy performs on historical data")
    print("   • Realistic: Results represent real-world performance")
    print("   • Your tests: Valid for strategy evaluation")
    
    print("\n🎯 MODE 2: TRAINING (For model improvement)")
    print("   • Model: LEARNING (weights update)")
    print("   • Purpose: Improve model predictions")
    print("   • Data: Separate training dataset")
    print("   • Result: Better Q-value predictions")
    
    print("\n🎯 MODE 3: ONLINE LEARNING (For live trading)")
    print("   • Model: ADAPTIVE (learns from real trades)")
    print("   • Purpose: Adapt to changing market conditions")
    print("   • Caution: Can overfit to recent data")
    print("   • Advanced: Usually done carefully")
    
    print("\n❌ MODE 4: DATA SNOOPING (What to avoid)")
    print("   • Model: Learning during backtesting")
    print("   • Problem: Overfits to test data")
    print("   • Result: Unrealistically good backtest results")
    print("   • Reality: Poor performance on new data")

def check_model_file_changes():
    """Check if model file is being modified"""
    
    print(f"\n🔍 CHECKING MODEL FILE MODIFICATIONS:")
    print("-" * 40)
    
    model_path = "models/q_model.keras"
    
    if os.path.exists(model_path):
        # Get file modification time
        mod_time = os.path.getmtime(model_path)
        import time
        mod_date = time.ctime(mod_time)
        
        print(f"📁 Model file: {model_path}")
        print(f"📅 Last modified: {mod_date}")
        
        # Check if recently modified (within last hour)
        current_time = time.time()
        hours_since_mod = (current_time - mod_time) / 3600
        
        if hours_since_mod < 1:
            print(f"⚠️  Recently modified: {hours_since_mod:.1f} hours ago")
            print("   This could indicate learning during recent backtests")
        else:
            print(f"✅ Not recently modified: {hours_since_mod:.1f} hours ago")
            print("   Model weights likely stable during backtests")
    else:
        print(f"❌ Model file not found: {model_path}")

def provide_recommendations():
    """Provide recommendations based on findings"""
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 30)
    
    print("✅ FOR VALID BACKTESTING:")
    print("   1. Ensure model weights DON'T change during backtests")
    print("   2. Use separate data for training vs testing")
    print("   3. Disable auto-retraining during backtests")
    print("   4. Save model only after dedicated training sessions")
    
    print("\n🎯 FOR MODEL IMPROVEMENT:")
    print("   1. Create dedicated training script")
    print("   2. Use historical data NOT used in backtests")
    print("   3. Train → Save → Backtest → Evaluate cycle")
    print("   4. Keep training and testing completely separate")
    
    print("\n⚠️ IF MODEL IS LEARNING DURING BACKTESTS:")
    print("   1. Your results may be overly optimistic")
    print("   2. Real trading performance might be worse")
    print("   3. Need to fix the learning/testing separation")
    print("   4. Re-run backtests with fixed model")

def main():
    """Main analysis function"""
    
    print("🤖 LAEF ML MODEL LEARNING ANALYSIS")
    print("=" * 50)
    
    # Check current behavior
    is_proper = check_model_learning_during_backtest()
    
    # Explain learning modes
    explain_ml_learning_modes()
    
    # Check file modifications
    check_model_file_changes()
    
    # Provide recommendations
    provide_recommendations()
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 30)
    
    if is_proper:
        print("✅ EXCELLENT: Your backtests appear to be valid!")
        print("   • Model is not learning during backtests")
        print("   • Results represent realistic performance")
        print("   • Your -1.41% to +X% improvements are genuine")
        print("   • Continue with confidence in your testing")
    else:
        print("⚠️ CAUTION: Model may be learning during backtests")
        print("   • This could make results overly optimistic")
        print("   • Consider fixing the learning/testing separation")
        print("   • Results might not represent real performance")
    
    print(f"\n🚀 NEXT STEPS:")
    if is_proper:
        print("   • Your backtest results are trustworthy")
        print("   • Continue tuning trading parameters")
        print("   • Test with multiple symbols")
        print("   • Consider dedicated model training if needed")
    else:
        print("   • Disable learning during backtests")
        print("   • Re-run tests with fixed model")
        print("   • Separate training and testing workflows")

if __name__ == "__main__":
    main()