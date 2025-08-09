"""
Multi-Timeframe Pattern Recognition Analyzer
Detects patterns at micro (1min-15min) and macro (daily-weekly) scales
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy.signal import find_peaks, argrelextrema
from collections import defaultdict, deque
import json
import sqlite3

logger = logging.getLogger(__name__)

class PatternRecognitionAnalyzer:
    """
    Advanced pattern recognition for LAEF's daily monitoring
    
    Analyzes patterns across multiple timeframes:
    - Micro: 1min, 5min, 15min (real-time momentum shifts)
    - Short: 30min, 1hour, 4hour (intraday trends)  
    - Macro: daily, weekly (market regime patterns)
    """
    
    def __init__(self, knowledge_db_path: str):
        self.knowledge_db_path = knowledge_db_path
        
        # Pattern detection parameters
        self.micro_patterns = {
            'breakout': {'min_bars': 10, 'threshold': 0.015},
            'reversal': {'min_bars': 5, 'threshold': 0.01},
            'consolidation': {'min_bars': 15, 'threshold': 0.005},
            'volume_spike': {'min_bars': 3, 'threshold': 2.0}
        }
        
        self.macro_patterns = {
            'trend_change': {'min_bars': 20, 'threshold': 0.03},
            'sector_rotation': {'min_stocks': 5, 'threshold': 0.02},
            'market_regime_shift': {'min_bars': 50, 'threshold': 0.05},
            'volatility_expansion': {'min_bars': 10, 'threshold': 1.5}
        }
        
        # Pattern memory for continuous learning
        self.pattern_memory = {
            'micro': deque(maxlen=500),
            'macro': deque(maxlen=100),
            'successful_patterns': defaultdict(list),
            'failed_patterns': defaultdict(list)
        }
        
        # Pattern performance tracking
        self.pattern_performance = defaultdict(lambda: {
            'detected': 0,
            'successful': 0,
            'accuracy': 0.0,
            'avg_magnitude': 0.0
        })
        
    def analyze_micro_patterns(self, symbol: str, ohlcv_data: pd.DataFrame,
                              current_price: float, volume: float) -> List[Dict[str, Any]]:
        """Analyze micro-patterns (minute-level) for real-time detection"""
        patterns = []
        
        if len(ohlcv_data) < 20:
            return patterns
            
        try:
            # Recent data for micro analysis
            recent_bars = ohlcv_data.tail(20)
            
            # 1. Breakout Pattern Detection
            breakout = self._detect_breakout(recent_bars, current_price)
            if breakout:
                patterns.append({
                    'type': 'micro_breakout',
                    'symbol': symbol,
                    'timeframe': 'micro',
                    'timestamp': datetime.now(),
                    'confidence': breakout['confidence'],
                    'direction': breakout['direction'],
                    'magnitude': breakout['magnitude'],
                    'key_level': breakout['key_level'],
                    'volume_confirmation': breakout['volume_confirmation']
                })
                
            # 2. Momentum Reversal Detection  
            reversal = self._detect_momentum_reversal(recent_bars)
            if reversal:
                patterns.append({
                    'type': 'micro_reversal',
                    'symbol': symbol,
                    'timeframe': 'micro',
                    'timestamp': datetime.now(),
                    'confidence': reversal['confidence'],
                    'direction': reversal['direction'],
                    'rsi_divergence': reversal.get('rsi_divergence', False)
                })
                
            # 3. Volume Spike Pattern
            volume_pattern = self._detect_volume_spike(recent_bars, volume)
            if volume_pattern:
                patterns.append({
                    'type': 'volume_spike',
                    'symbol': symbol,
                    'timeframe': 'micro',
                    'timestamp': datetime.now(),
                    'confidence': volume_pattern['confidence'],
                    'volume_ratio': volume_pattern['ratio'],
                    'price_movement': volume_pattern['price_movement']
                })
                
            # 4. Consolidation Break
            consolidation = self._detect_consolidation_break(recent_bars)
            if consolidation:
                patterns.append({
                    'type': 'consolidation_break',
                    'symbol': symbol,
                    'timeframe': 'micro',
                    'timestamp': datetime.now(),
                    'confidence': consolidation['confidence'],
                    'direction': consolidation['direction'],
                    'range_size': consolidation['range_size']
                })
                
        except Exception as e:
            logger.error(f"Error analyzing micro patterns for {symbol}: {e}")
            
        return patterns
    
    def analyze_macro_patterns(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze macro-patterns (market-wide, daily+ timeframes)"""
        patterns = []
        
        try:
            # 1. Market Regime Analysis
            regime_pattern = self._analyze_market_regime(market_data)
            if regime_pattern:
                patterns.append(regime_pattern)
                
            # 2. Sector Rotation Detection
            sector_rotation = self._detect_sector_rotation(market_data)
            if sector_rotation:
                patterns.append(sector_rotation)
                
            # 3. Volatility Regime Changes
            volatility_pattern = self._analyze_volatility_regime(market_data)
            if volatility_pattern:
                patterns.append(volatility_pattern)
                
            # 4. Cross-Asset Correlations
            correlation_pattern = self._analyze_cross_asset_patterns(market_data)
            if correlation_pattern:
                patterns.append(correlation_pattern)
                
        except Exception as e:
            logger.error(f"Error analyzing macro patterns: {e}")
            
        return patterns
    
    def _detect_breakout(self, bars: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Detect price breakout patterns"""
        try:
            if len(bars) < self.micro_patterns['breakout']['min_bars']:
                return None
                
            # Calculate support and resistance levels
            highs = bars['high'].values
            lows = bars['low'].values
            
            # Find recent range
            recent_high = np.max(highs[-10:])
            recent_low = np.min(lows[-10:])
            range_size = (recent_high - recent_low) / recent_low
            
            # Check for breakout
            threshold = self.micro_patterns['breakout']['threshold']
            
            if current_price > recent_high * (1 + threshold/3):
                # Upside breakout
                volume_conf = bars['volume'].iloc[-1] > bars['volume'].tail(10).mean() * 1.5
                
                return {
                    'direction': 'up',
                    'confidence': min(0.9, 0.6 + (current_price - recent_high) / recent_high),
                    'magnitude': (current_price - recent_high) / recent_high,
                    'key_level': recent_high,
                    'volume_confirmation': volume_conf
                }
                
            elif current_price < recent_low * (1 - threshold/3):
                # Downside breakout
                volume_conf = bars['volume'].iloc[-1] > bars['volume'].tail(10).mean() * 1.5
                
                return {
                    'direction': 'down', 
                    'confidence': min(0.9, 0.6 + (recent_low - current_price) / recent_low),
                    'magnitude': (recent_low - current_price) / recent_low,
                    'key_level': recent_low,
                    'volume_confirmation': volume_conf
                }
                
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            
        return None
    
    def _detect_momentum_reversal(self, bars: pd.DataFrame) -> Optional[Dict]:
        """Detect momentum reversal patterns"""
        try:
            if len(bars) < self.micro_patterns['reversal']['min_bars']:
                return None
                
            closes = bars['close'].values
            
            # Calculate short-term momentum
            momentum_3 = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
            momentum_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
            
            # Look for momentum divergence
            price_change = closes[-1] - closes[-3]
            prev_price_change = closes[-3] - closes[-5] if len(closes) >= 5 else 0
            
            # Reversal criteria
            if price_change * prev_price_change < 0 and abs(momentum_3) > 0.005:
                direction = 'up' if price_change > 0 else 'down'
                confidence = min(0.8, abs(momentum_3) * 50)
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'momentum_3': momentum_3,
                    'momentum_5': momentum_5
                }
                
        except Exception as e:
            logger.error(f"Error detecting momentum reversal: {e}")
            
        return None
    
    def _detect_volume_spike(self, bars: pd.DataFrame, current_volume: float) -> Optional[Dict]:
        """Detect volume spike patterns"""
        try:
            if len(bars) < 10:
                return None
                
            avg_volume = bars['volume'].tail(10).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > self.micro_patterns['volume_spike']['threshold']:
                # Calculate price movement during volume spike
                current_price = bars['close'].iloc[-1]
                prev_price = bars['close'].iloc[-2]
                price_movement = (current_price - prev_price) / prev_price
                
                return {
                    'ratio': volume_ratio,
                    'confidence': min(0.9, volume_ratio / 5),
                    'price_movement': price_movement,
                    'avg_volume': avg_volume
                }
                
        except Exception as e:
            logger.error(f"Error detecting volume spike: {e}")
            
        return None
    
    def _detect_consolidation_break(self, bars: pd.DataFrame) -> Optional[Dict]:
        """Detect consolidation breakout patterns"""
        try:
            if len(bars) < self.micro_patterns['consolidation']['min_bars']:
                return None
                
            # Look for consolidation period
            recent_closes = bars['close'].tail(10)
            price_range = (recent_closes.max() - recent_closes.min()) / recent_closes.mean()
            
            # If range is small, we have consolidation
            if price_range < self.micro_patterns['consolidation']['threshold']:
                current_price = bars['close'].iloc[-1]
                consolidation_high = recent_closes.max()
                consolidation_low = recent_closes.min()
                
                # Check for break
                if current_price > consolidation_high * 1.002:
                    return {
                        'direction': 'up',
                        'confidence': 0.7,
                        'range_size': price_range,
                        'breakout_level': consolidation_high
                    }
                elif current_price < consolidation_low * 0.998:
                    return {
                        'direction': 'down',
                        'confidence': 0.7,
                        'range_size': price_range,
                        'breakout_level': consolidation_low
                    }
                    
        except Exception as e:
            logger.error(f"Error detecting consolidation break: {e}")
            
        return None
    
    def _analyze_market_regime(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Analyze overall market regime patterns"""
        try:
            if 'SPY' not in market_data or 'VIX' not in market_data:
                return None
                
            spy_data = market_data['SPY']['ohlcv']
            vix_level = market_data['VIX'].get('current_price', 20)
            
            if len(spy_data) < 50:
                return None
                
            # Calculate regime indicators
            returns = spy_data['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            trend = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[-20]) / spy_data['close'].iloc[-20]
            
            # Determine regime
            regime = self._classify_market_regime(volatility, trend, vix_level)
            
            return {
                'type': 'market_regime',
                'timeframe': 'macro',
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'vix_level': vix_level,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return None
    
    def _classify_market_regime(self, volatility: float, trend: float, vix: float) -> str:
        """Classify market regime based on key metrics"""
        if vix > 30 or volatility > 0.25:
            if abs(trend) > 0.05:
                return 'crisis_trending'
            else:
                return 'crisis_volatile'
        elif vix < 15 and volatility < 0.15:
            if trend > 0.03:
                return 'calm_bullish'
            elif trend < -0.03:
                return 'calm_bearish'
            else:
                return 'calm_sideways'
        else:
            if trend > 0.02:
                return 'normal_bullish'
            elif trend < -0.02:
                return 'normal_bearish'
            else:
                return 'normal_choppy'
    
    def _detect_sector_rotation(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Detect sector rotation patterns"""
        try:
            # Look for sector ETFs in data
            sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP']
            sector_data = {}
            
            for etf in sector_etfs:
                if etf in market_data:
                    data = market_data[etf]['ohlcv']
                    if len(data) >= 10:
                        # Calculate recent performance
                        recent_return = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
                        sector_data[etf] = recent_return
            
            if len(sector_data) < 3:
                return None
                
            # Find strongest and weakest sectors
            sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
            strongest = sorted_sectors[0]
            weakest = sorted_sectors[-1]
            
            # Check if rotation is significant
            spread = strongest[1] - weakest[1]
            if abs(spread) > 0.02:  # 2% spread
                return {
                    'type': 'sector_rotation',
                    'timeframe': 'macro', 
                    'timestamp': datetime.now(),
                    'strongest_sector': strongest[0],
                    'weakest_sector': weakest[0],
                    'spread': spread,
                    'confidence': min(0.9, abs(spread) * 25)
                }
                
        except Exception as e:
            logger.error(f"Error detecting sector rotation: {e}")
            
        return None
    
    def _analyze_volatility_regime(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Analyze volatility regime changes"""
        try:
            if 'VIX' not in market_data:
                return None
                
            vix_data = market_data['VIX']['ohlcv']
            if len(vix_data) < 20:
                return None
                
            current_vix = vix_data['close'].iloc[-1]
            vix_ma = vix_data['close'].tail(10).mean()
            vix_change = (current_vix - vix_ma) / vix_ma
            
            # Detect significant volatility regime changes
            if abs(vix_change) > 0.2:  # 20% change in VIX
                regime_change = 'expansion' if vix_change > 0 else 'contraction'
                
                return {
                    'type': 'volatility_regime',
                    'timeframe': 'macro',
                    'timestamp': datetime.now(),
                    'regime_change': regime_change,
                    'current_vix': current_vix,
                    'vix_change': vix_change,
                    'confidence': min(0.9, abs(vix_change))
                }
                
        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {e}")
            
        return None
    
    def _analyze_cross_asset_patterns(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Analyze cross-asset correlation patterns"""
        try:
            # Look for divergences between stocks and bonds, commodities, etc.
            assets = ['SPY', 'QQQ', 'IWM']  # Different market caps
            
            asset_returns = {}
            for asset in assets:
                if asset in market_data:
                    data = market_data[asset]['ohlcv']
                    if len(data) >= 10:
                        # 5-day return
                        ret = (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6]
                        asset_returns[asset] = ret
                        
            if len(asset_returns) < 2:
                return None
                
            # Check for divergences
            returns_list = list(asset_returns.values())
            max_spread = max(returns_list) - min(returns_list)
            
            if max_spread > 0.03:  # 3% divergence
                return {
                    'type': 'cross_asset_divergence',
                    'timeframe': 'macro',
                    'timestamp': datetime.now(),
                    'asset_returns': asset_returns,
                    'max_spread': max_spread,
                    'confidence': min(0.8, max_spread * 15)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing cross-asset patterns: {e}")
            
        return None
    
    def store_pattern_observation(self, pattern: Dict[str, Any], outcome: Optional[str] = None):
        """Store pattern observation in knowledge database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pattern_observations 
                (timestamp, pattern_type, timeframe, symbols, pattern_data, outcome, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.get('timestamp', datetime.now()),
                pattern.get('type', 'unknown'),
                pattern.get('timeframe', 'unknown'),
                pattern.get('symbol', 'MARKET'),
                json.dumps(pattern, default=str),
                outcome or 'pending',
                pattern.get('confidence', 0.5)
            ))
            
            conn.commit()
            conn.close()
            
            # Update pattern performance tracking
            pattern_type = pattern.get('type', 'unknown')
            self.pattern_performance[pattern_type]['detected'] += 1
            
            if outcome == 'success':
                self.pattern_performance[pattern_type]['successful'] += 1
                self.pattern_performance[pattern_type]['accuracy'] = (
                    self.pattern_performance[pattern_type]['successful'] / 
                    self.pattern_performance[pattern_type]['detected']
                )
                
        except Exception as e:
            logger.error(f"Error storing pattern observation: {e}")
    
    def get_pattern_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights from recent pattern observations"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            # Get pattern success rates
            cursor.execute('''
                SELECT pattern_type, 
                       COUNT(*) as total,
                       AVG(accuracy_score) as avg_confidence,
                       SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successful
                FROM pattern_observations 
                WHERE timestamp > ?
                GROUP BY pattern_type
                ORDER BY total DESC
            ''', (since_date,))
            
            patterns = []
            for row in cursor.fetchall():
                success_rate = row[3] / row[1] if row[1] > 0 else 0
                patterns.append({
                    'type': row[0],
                    'total_detected': row[1],
                    'avg_confidence': row[2],
                    'success_rate': success_rate,
                    'successful': row[3]
                })
            
            # Get timeframe analysis
            cursor.execute('''
                SELECT timeframe, COUNT(*) as count
                FROM pattern_observations 
                WHERE timestamp > ?
                GROUP BY timeframe
            ''', (since_date,))
            
            timeframes = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'patterns': patterns,
                'timeframe_distribution': timeframes,
                'total_patterns': sum(p['total_detected'] for p in patterns),
                'overall_success_rate': (
                    sum(p['successful'] for p in patterns) / 
                    sum(p['total_detected'] for p in patterns)
                    if sum(p['total_detected'] for p in patterns) > 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern insights: {e}")
            return {}
    
    def update_pattern_memory(self, patterns: List[Dict], market_data: Dict):
        """Update pattern memory for continuous learning"""
        try:
            current_time = datetime.now()
            
            # Store patterns in appropriate memory buckets
            for pattern in patterns:
                timeframe = pattern.get('timeframe', 'unknown')
                
                pattern_record = {
                    'timestamp': current_time,
                    'pattern': pattern,
                    'market_context': {
                        'vix': market_data.get('VIX', {}).get('current_price', 20),
                        'spy_price': market_data.get('SPY', {}).get('current_price', 400)
                    }
                }
                
                if timeframe == 'micro':
                    self.pattern_memory['micro'].append(pattern_record)
                elif timeframe == 'macro':
                    self.pattern_memory['macro'].append(pattern_record)
                    
        except Exception as e:
            logger.error(f"Error updating pattern memory: {e}")
