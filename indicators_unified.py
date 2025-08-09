# === indicators_unified.py - Standardized Indicator Calculation ===

import pandas as pd
import numpy as np
import logging
from config import STATE_FEATURES

def calculate_all_indicators(df):
    """
    Calculate ALL indicators needed for LAEF state vector.
    Returns DataFrame with standardized column names matching STATE_FEATURES.
    
    Input: DataFrame with OHLCV data
    Output: DataFrame with all required indicators
    """
    try:
        df = df.copy()
        logging.info(f"[INDICATORS] Calculating indicators for {len(df)} rows")
        
        # Ensure we have required base columns
        required_base = ['open', 'high', 'low', 'close', 'volume']
        missing_base = [col for col in required_base if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required columns: {missing_base}")
        
        # === MOVING AVERAGES ===
        df['sma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # === MACD ===
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # === VWAP ===
        if 'volume' in df.columns and df['volume'].sum() > 0:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            # Fallback if no volume data
            df['vwap'] = df['close'].rolling(window=20, min_periods=1).mean()
            logging.warning("[INDICATORS] No volume data, using SMA20 as VWAP fallback")
        
        # Fill any remaining NaN values
        df = df.ffill().bfill().fillna(0)
        
        # Verify all required columns exist
        missing_cols = [col for col in STATE_FEATURES if col not in df.columns]
        if missing_cols:
            logging.error(f"[INDICATORS] Missing required columns after calculation: {missing_cols}")
            raise ValueError(f"Failed to calculate: {missing_cols}")
        
        logging.info(f"[INDICATORS] Successfully calculated all indicators")
        return df
        
    except Exception as e:
        logging.error(f"[INDICATORS] Failed to calculate indicators: {e}")
        raise

def validate_indicator_data(df):
    """
    Validate that indicator data is reasonable (no extreme values, sufficient data points)
    """
    try:
        # Check for required columns
        missing = [col for col in STATE_FEATURES if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
        
        # Check for sufficient data
        if len(df) < 20:
            return False, f"Insufficient data: {len(df)} rows"
        
        # Check for extreme values or all NaN
        for col in STATE_FEATURES:
            if df[col].isna().all():
                return False, f"Column {col} is all NaN"
            
            # Check for infinite values
            if np.isinf(df[col]).any():
                return False, f"Column {col} contains infinite values"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def calculate_momentum_indicators(df, symbol: str = None):
    """
    Calculate momentum indicators optimized for scalping
    
    Returns dict with:
    - recent_closes: List of recent close prices
    - momentum_score: Current momentum score
    - price_velocity: Rate of price change
    - price_acceleration: Change in velocity
    - volume_ratio: Current volume / average volume
    """
    try:
        if df is None or len(df) < 10:
            return {}
        
        # Get recent close prices (last 20 for momentum calculations)
        recent_closes = df['close'].tail(20).tolist()
        
        # Calculate Rate of Change (ROC) over 5 periods
        if len(recent_closes) >= 5:
            roc = ((recent_closes[-1] / recent_closes[-5]) - 1) * 100
        else:
            roc = 0.0
        
        # Calculate price velocity (average rate of change over 3 periods)
        velocity = 0.0
        if len(recent_closes) >= 3:
            velocity_sum = 0
            for i in range(-3, 0):
                if i > -len(recent_closes):
                    velocity_sum += ((recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1]) * 100
            velocity = velocity_sum / 3
        
        # Calculate price acceleration (change in velocity)
        acceleration = 0.0
        if len(recent_closes) >= 6:
            # Two velocity readings
            v1 = ((recent_closes[-2] - recent_closes[-4]) / recent_closes[-4]) * 100
            v2 = ((recent_closes[-1] - recent_closes[-3]) / recent_closes[-3]) * 100
            acceleration = v2 - v1
        
        # Combined momentum score
        momentum_score = (roc * 0.5) + (velocity * 0.3) + (acceleration * 0.2)
        
        # Volume ratio (current vs average)
        volume_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 10:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(10).mean()
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
        
        return {
            'recent_closes': recent_closes,
            'momentum_score': momentum_score,
            'price_velocity': velocity,
            'price_acceleration': acceleration,
            'volume_ratio': volume_ratio,
            'roc': roc,
            'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0,
            'volume_sma': df['volume'].tail(10).mean() if 'volume' in df.columns else 0
        }
        
    except Exception as e:
        logging.error(f"[MOMENTUM INDICATORS] Failed to calculate for {symbol}: {e}")
        return {}

def get_latest_indicators(df):
    """
    Get the most recent indicator values as a dictionary.
    Includes both traditional indicators and momentum data for AI scalping.
    """
    if len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    indicators = {col: float(latest[col]) for col in STATE_FEATURES if col in df.columns}
    
    # Add momentum data for AI scalping
    momentum_data = calculate_momentum_indicators(df)
    indicators.update(momentum_data)
    
    return indicators