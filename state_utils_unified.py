# === state_utils_unified.py - Standardized State Vector Creation ===

import numpy as np
import pandas as pd
import logging
from config import STATE_FEATURES, STATE_SIZE

def create_state_vector(df_window):
    """
    Create a normalized state vector from the latest row of a DataFrame window.
    
    Args:
        df_window: DataFrame with indicator columns matching STATE_FEATURES
        
    Returns:
        numpy array of shape (STATE_SIZE,) or None if failed
    """
    try:
        if df_window is None or len(df_window) == 0:
            logging.warning("[STATE] Empty DataFrame window provided")
            return None
        
        # Check for required columns
        available_cols = set(df_window.columns)
        required_cols = set(STATE_FEATURES)
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            logging.error(f"[STATE] Missing required columns: {missing_cols}")
            logging.info(f"[STATE] Available columns: {list(available_cols)}")
            return None
        
        # Extract the latest row values in the correct order
        latest_row = df_window.iloc[-1]
        state_values = []
        
        for feature in STATE_FEATURES:
            value = latest_row[feature]
            
            # Handle NaN values
            if pd.isna(value):
                logging.warning(f"[STATE] NaN value found for {feature}, using 0.0")
                value = 0.0
            
            # Handle infinite values  
            if np.isinf(value):
                logging.warning(f"[STATE] Infinite value found for {feature}, using 0.0")
                value = 0.0
                
            state_values.append(float(value))
        
        # Convert to numpy array
        state = np.array(state_values)
        
        # Validate state size
        if len(state) != STATE_SIZE:
            logging.error(f"[STATE] State size mismatch: {len(state)} vs expected {STATE_SIZE}")
            return None
        
        # Normalize the state vector to prevent extreme values
        # Use robust normalization to handle outliers
        state_normalized = normalize_state(state)
        
        logging.debug(f"[STATE] Created state vector: shape={state_normalized.shape}")
        return state_normalized
        
    except Exception as e:
        logging.error(f"[STATE] Failed to create state vector: {e}")
        return None

def normalize_state(state):
    """
    Normalize state vector using robust scaling to handle outliers.
    """
    try:
        # Use tanh normalization to bound values between -1 and 1
        # This is more robust than min-max or z-score for financial data
        
        # Scale different feature types appropriately
        normalized = np.zeros_like(state)
        
        # Price features (indices 0-3): open, high, low, close
        price_features = state[0:4]
        if np.max(price_features) > 0:
            price_scale = np.median(price_features)
            normalized[0:4] = np.tanh(price_features / (price_scale + 1e-6))
        
        # MACD and signal (indices 4-5)
        macd_features = state[4:6]
        macd_scale = np.std(macd_features) + 1e-6
        normalized[4:6] = np.tanh(macd_features / macd_scale)
        
        # RSI (index 6) - already bounded 0-100, just scale to -1,1
        normalized[6] = (state[6] - 50) / 50.0
        
        # Moving averages (indices 7-9): sma20, sma50, ema20
        ma_features = state[7:10]
        if np.max(ma_features) > 0:
            ma_scale = np.median(ma_features)
            normalized[7:10] = np.tanh(ma_features / (ma_scale + 1e-6))
        
        # VWAP (index 10)
        if state[10] > 0:
            normalized[10] = np.tanh(state[10] / (state[3] + 1e-6))  # Normalize by close price
        
        # Volume (index 11)
        if state[11] > 0:
            normalized[11] = np.tanh(state[11] / (np.mean(state[0:4]) * 1000 + 1e-6))
        
        # Ensure no NaN or inf values in output
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
        
    except Exception as e:
        logging.error(f"[STATE] Normalization failed: {e}")
        # Fallback: simple scaling
        return np.tanh(state / (np.max(np.abs(state)) + 1e-6))

def validate_state_pipeline(df):
    """
    Test the complete state creation pipeline on a DataFrame.
    Returns success status and any error messages.
    """
    try:
        # Test state creation
        state = create_state_vector(df)
        
        if state is None:
            return False, "State creation returned None"
        
        if len(state) != STATE_SIZE:
            return False, f"Wrong state size: {len(state)} vs {STATE_SIZE}"
        
        if np.any(np.isnan(state)):
            return False, "State contains NaN values"
        
        if np.any(np.isinf(state)):
            return False, "State contains infinite values"
        
        if np.all(state == 0):
            return False, "State is all zeros"
        
        return True, f"State pipeline validation passed: {state.shape}"
        
    except Exception as e:
        return False, f"Pipeline validation failed: {e}"

def extract_raw_indicators(df_window):
    """
    Extract raw (unnormalized) indicator values for use in trading logic.
    Returns dictionary with current indicator values.
    """
    try:
        if df_window is None or len(df_window) == 0:
            return {}
        
        latest = df_window.iloc[-1]
        
        return {
            'price': float(latest['close']),
            'vwap': float(latest['vwap']),
            'rsi': float(latest['rsi']),
            'macd': float(latest['macd']),
            'signal': float(latest['signal']),
            'sma20': float(latest['sma20']),
            'sma50': float(latest['sma50']),
            'ema20': float(latest['ema20']),
            'volume': float(latest['volume'])
        }
        
    except Exception as e:
        logging.error(f"[STATE] Failed to extract raw indicators: {e}")
        return {}

# Legacy function names for backward compatibility
format_state = create_state_vector
shape_reward = lambda pnl: float(np.tanh(pnl / 10.0))