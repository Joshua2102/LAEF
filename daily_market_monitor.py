"""
LAEF Daily Market Monitor
Automatic daily wake-up system for continuous market observation and learning
Runs every day from 8am to market close, making and tracking predictions
"""

import logging
import os
import time
import threading
import queue
from datetime import datetime, timedelta, time as datetime_time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
from collections import defaultdict, deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import schedule
import pytz

# Market monitoring components
from live_market_learner import LiveMarketLearner
from prediction_tracker import PredictionTracker

# Strategy components for prediction generation
try:
    from core.agent_unified import LAEFAgent
    from core.state_utils_unified import create_state_vector, extract_raw_indicators
    from data.data_fetcher_unified import fetch_stock_data, fetch_multiple_symbols
    from core.indicators_unified import calculate_all_indicators
except ImportError:
    print("Warning: Some LAEF components not available. Running in simulation mode.")


class DailyMarketMonitor:
    """
    LAEF's Daily Market Monitoring and Learning System
    
    Features:
    - Automatic daily wake-up at 8am EST
    - Continuous market observation during trading hours
    - Multi-timeframe prediction generation (1min, 5min, 1hr, daily)
    - Macro and micro pattern recognition
    - Knowledge accumulation and learning
    - No actual trading - pure observation and learning
    """
    
    def __init__(self):
        # Time management
        self.timezone = pytz.timezone('US/Eastern')
        self.wake_time = datetime_time(8, 0)  # 8:00 AM EST
        self.market_open = datetime_time(9, 30)
        self.market_close = datetime_time(16, 0)
        
        # Core components
        self.agent = None  # Will be initialized when needed
        self.prediction_tracker = PredictionTracker()
        
        # Monitoring configuration
        self.watch_symbols = [
            # Major indices for macro analysis
            'SPY', 'QQQ', 'IWM', 'VIX',
            # Core stocks for micro analysis
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            # Sector representatives
            'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP'
        ]
        
        # Prediction intervals (in minutes)
        self.prediction_intervals = {
            'micro': {'1min': 1, '5min': 5, '15min': 15},
            'short': {'30min': 30, '1hour': 60, '2hour': 120},
            'macro': {'4hour': 240, '1day': 1440}
        }
        
        # Pattern recognition settings
        self.pattern_memory = {
            'micro_patterns': deque(maxlen=1000),  # Minute-level patterns
            'macro_patterns': deque(maxlen=100),   # Daily macro patterns
            'market_regime_history': deque(maxlen=50)
        }
        
        # Knowledge accumulation
        self.knowledge_db_path = "logs/knowledge/market_observations.db"
        self.daily_insights = defaultdict(list)
        
        # Threading and scheduling
        self.monitoring_active = False
        self.scheduler_thread = None
        self.monitor_thread = None
        self.prediction_queue = queue.PriorityQueue()
        
        # Performance tracking
        self.session_stats = {
            'predictions_made': 0,
            'accuracy_by_timeframe': defaultdict(list),
            'patterns_detected': defaultdict(int),
            'market_regime_accuracy': []
        }
        
        # Setup logging
        self._setup_logging()
        self._setup_knowledge_db()
        
        logger.info("LAEF Daily Market Monitor initialized")
        
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = "logs/daily_monitoring"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f"market_monitor_{today}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        global logger
        logger = logging.getLogger(__name__)
        
    def _setup_knowledge_db(self):
        """Setup knowledge database for storing observations"""
        os.makedirs(os.path.dirname(self.knowledge_db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.knowledge_db_path)
        cursor = conn.cursor()
        
        # Market observations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                observation_type TEXT,
                symbol TEXT,
                data TEXT,
                confidence REAL,
                notes TEXT
            )
        ''')
        
        # Pattern recognition table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                pattern_type TEXT,
                timeframe TEXT,
                symbols TEXT,
                pattern_data TEXT,
                outcome TEXT,
                accuracy_score REAL
            )
        ''')
        
        # Daily insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                insight_type TEXT,
                content TEXT,
                confidence REAL,
                supporting_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_daily_scheduler(self):
        """Start the daily scheduler that wakes LAEF at 8am"""
        logger.info("Starting LAEF daily scheduler...")
        
        # Schedule daily wake-up
        schedule.every().day.at("08:00").do(self._daily_wakeup)
        
        # Run scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Daily scheduler started - LAEF will wake up at 8:00 AM EST every day")
        
        # If it's already past 8am today, start monitoring now
        now = datetime.now(self.timezone)
        if now.time() >= self.wake_time:
            logger.info("It's already past wake time today - starting monitoring now")
            self._daily_wakeup()
    
    def _daily_wakeup(self):
        """Daily wake-up routine - starts market monitoring"""
        now = datetime.now(self.timezone)
        logger.info(f"🌅 LAEF DAILY WAKE-UP - {now.strftime('%A, %B %d, %Y at %I:%M %p')}")
        
        # Reset daily stats
        self.session_stats = {
            'predictions_made': 0,
            'accuracy_by_timeframe': defaultdict(list),
            'patterns_detected': defaultdict(int),
            'market_regime_accuracy': [],
            'start_time': now
        }
        
        # Initialize AI components
        self._initialize_ai_components()
        
        # Start market monitoring
        if not self.monitoring_active:
            self.start_market_monitoring()
        
        # Schedule end-of-day summary
        end_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
        if now < end_time:
            schedule_time = end_time.strftime('%H:%M')
            schedule.every().day.at(schedule_time).do(self._end_of_day_summary).tag('daily_summary')
    
    def _initialize_ai_components(self):
        """Initialize LAEF's AI components for the day"""
        try:
            logger.info("Initializing LAEF AI components...")
            
            # Initialize the main agent
            self.agent = LAEFAgent(pretrained=True)
            
            # Load any overnight learning updates
            self._load_overnight_insights()
            
            logger.info("✅ AI components ready for market observation")
            
        except Exception as e:
            logger.error(f"Error initializing AI components: {e}")
            logger.info("Running in observation-only mode without AI predictions")
    
    def start_market_monitoring(self):
        """Start continuous market monitoring"""
        if self.monitoring_active:
            logger.warning("Market monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("🔍 Starting continuous market monitoring...")
        
        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start prediction processing thread
        prediction_thread = threading.Thread(target=self._process_predictions, daemon=True)
        prediction_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs during market hours"""
        logger.info("Market monitoring loop started")
        
        while self.monitoring_active:
            try:
                now = datetime.now(self.timezone)
                current_time = now.time()
                
                # Check if market is open or if we're in pre-market analysis
                if current_time >= datetime_time(8, 0) and current_time <= datetime_time(17, 0):
                    
                    # Determine monitoring intensity based on time
                    if current_time < self.market_open:
                        # Pre-market: less frequent monitoring
                        self._pre_market_analysis()
                        sleep_time = 300  # 5 minutes
                        
                    elif self.market_open <= current_time <= self.market_close:
                        # Market hours: active monitoring
                        self._active_market_monitoring()
                        sleep_time = 60  # 1 minute
                        
                    else:
                        # After-hours: pattern analysis and learning
                        self._post_market_analysis()
                        sleep_time = 600  # 10 minutes
                        
                else:
                    # Outside monitoring hours
                    logger.info("Outside monitoring hours - LAEF is resting")
                    break
                
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
        
        logger.info("Market monitoring loop ended")
    
    def _pre_market_analysis(self):
        """Pre-market analysis and preparation"""
        logger.info("📊 Conducting pre-market analysis...")
        
        try:
            # Fetch overnight news and events
            macro_conditions = self._analyze_macro_conditions()
            
            # Analyze futures and pre-market movements
            premarket_data = self._get_premarket_data()
            
            # Generate daily market predictions
            daily_predictions = self._generate_daily_predictions(macro_conditions)
            
            # Store insights
            self._store_observation('premarket_analysis', {
                'macro_conditions': macro_conditions,
                'daily_predictions': daily_predictions,
                'premarket_strength': premarket_data.get('strength', 'neutral')
            })
            
        except Exception as e:
            logger.error(f"Error in pre-market analysis: {e}")
    
    def _active_market_monitoring(self):
        """Active monitoring during market hours"""
        try:
            # Fetch current market data
            current_data = self._fetch_current_market_data()
            
            if not current_data:
                return
            
            # Generate predictions for different timeframes
            self._generate_realtime_predictions(current_data)
            
            # Detect micro patterns
            micro_patterns = self._detect_micro_patterns(current_data)
            
            # Analyze market regime changes
            regime_change = self._detect_regime_changes(current_data)
            
            # Update pattern memory
            self._update_pattern_memory(micro_patterns, current_data)
            
            # Check prediction outcomes
            self._check_prediction_outcomes()
            
            logger.debug(f"Monitored {len(current_data)} symbols, "
                        f"detected {len(micro_patterns)} micro patterns")
            
        except Exception as e:
            logger.error(f"Error in active market monitoring: {e}")
    
    def _post_market_analysis(self):
        """Post-market analysis and learning"""
        logger.info("📈 Conducting post-market analysis...")
        
        try:
            # Analyze the day's patterns
            daily_patterns = self._analyze_daily_patterns()
            
            # Review prediction accuracy
            accuracy_summary = self._calculate_daily_accuracy()
            
            # Identify learning opportunities
            learning_insights = self._extract_learning_insights()
            
            # Store daily insights
            self._store_daily_insights(daily_patterns, accuracy_summary, learning_insights)
            
        except Exception as e:
            logger.error(f"Error in post-market analysis: {e}")
    
    def _fetch_current_market_data(self) -> Dict[str, Any]:
        """Fetch real-time market data for all watched symbols"""
        try:
            current_data = {}
            
            # Fetch data for all symbols
            for symbol in self.watch_symbols:
                try:
                    # Get recent price data
                    data = fetch_stock_data(
                        symbol,
                        start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                        end=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if data is not None and len(data) > 0:
                        # Calculate indicators
                        data_with_indicators = calculate_all_indicators(data)
                        
                        current_data[symbol] = {
                            'ohlcv': data_with_indicators,
                            'current_price': data_with_indicators.iloc[-1]['close'],
                            'volume': data_with_indicators.iloc[-1]['volume'],
                            'indicators': extract_raw_indicators(data_with_indicators)
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {e}")
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _generate_realtime_predictions(self, market_data: Dict[str, Any]):
        """Generate predictions for multiple timeframes"""
        try:
            for symbol, data in market_data.items():
                # Skip if no agent available
                if self.agent is None:
                    continue
                    
                # Generate state vector
                state = create_state_vector(data['ohlcv'])
                if state is None:
                    continue
                
                # Generate predictions for different timeframes
                for category, intervals in self.prediction_intervals.items():
                    for timeframe, minutes in intervals.items():
                        
                        # Get model prediction
                        q_values = self.agent.predict_q_values(state)
                        action = np.argmax(q_values)
                        confidence = float(np.max(q_values) - np.mean(q_values))
                        
                        # Calculate price target based on action
                        current_price = data['current_price']
                        
                        if action == 1:  # Buy signal
                            predicted_change = confidence * 0.01  # Scale to reasonable %
                        elif action == 2:  # Sell signal
                            predicted_change = -confidence * 0.01
                        else:  # Hold
                            predicted_change = 0
                        
                        predicted_price = current_price * (1 + predicted_change)
                        
                        # Store prediction
                        pred_id = self.prediction_tracker.add_prediction(
                            symbol=symbol,
                            prediction_type=category,
                            timeframe=timeframe,
                            current_price=current_price,
                            predicted_price=predicted_price,
                            confidence=confidence,
                            q_value=float(q_values[action]),
                            ml_score=confidence,
                            indicators=data['indicators'],
                            market_conditions={'category': category, 'agent_action': int(action)}
                        )
                        
                        self.session_stats['predictions_made'] += 1
                        
                        # Log significant predictions
                        if confidence > 0.7:
                            logger.info(f"High confidence {timeframe} prediction for {symbol}: "
                                      f"{predicted_change:+.1%} (conf: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    def _detect_micro_patterns(self, market_data: Dict[str, Any]) -> List[Dict]:
        """Detect minute-level patterns across symbols"""
        patterns = []
        
        try:
            for symbol, data in market_data.items():
                ohlcv = data['ohlcv']
                if len(ohlcv) < 20:
                    continue
                
                # Price action patterns
                recent_bars = ohlcv.tail(10)
                
                # Detect breakout patterns
                if self._is_breakout_pattern(recent_bars):
                    patterns.append({
                        'type': 'breakout',
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'confidence': 0.8,
                        'data': recent_bars.tail(5).to_dict()
                    })
                
                # Detect reversal patterns
                if self._is_reversal_pattern(recent_bars):
                    patterns.append({
                        'type': 'reversal',
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'confidence': 0.7,
                        'data': recent_bars.tail(3).to_dict()
                    })
                
                # Volume surge patterns
                if self._is_volume_surge(recent_bars):
                    patterns.append({
                        'type': 'volume_surge',
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'confidence': 0.6,
                        'volume_ratio': recent_bars['volume'].iloc[-1] / recent_bars['volume'].mean()
                    })
        
        except Exception as e:
            logger.error(f"Error detecting micro patterns: {e}")
        
        return patterns
    
    def _is_breakout_pattern(self, bars: pd.DataFrame) -> bool:
        """Detect breakout patterns"""
        if len(bars) < 10:
            return False
        
        # Simple breakout detection
        recent_high = bars['high'].tail(5).max()
        prev_high = bars['high'].head(5).max()
        
        return recent_high > prev_high * 1.02  # 2% breakout
    
    def _is_reversal_pattern(self, bars: pd.DataFrame) -> bool:
        """Detect reversal patterns"""
        if len(bars) < 5:
            return False
        
        # Simple reversal pattern
        recent_direction = bars['close'].iloc[-1] - bars['close'].iloc[-3]
        prev_direction = bars['close'].iloc[-3] - bars['close'].iloc[-5]
        
        # Direction reversal
        return (recent_direction > 0 and prev_direction < 0) or \
               (recent_direction < 0 and prev_direction > 0)
    
    def _is_volume_surge(self, bars: pd.DataFrame) -> bool:
        """Detect volume surge"""
        if len(bars) < 10:
            return False
        
        recent_volume = bars['volume'].iloc[-1]
        avg_volume = bars['volume'].head(9).mean()
        
        return recent_volume > avg_volume * 2  # 2x volume surge
    
    def _detect_regime_changes(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Detect market regime changes"""
        try:
            if 'SPY' not in market_data:
                return None
                
            spy_data = market_data['SPY']
            ohlcv = spy_data['ohlcv']
            
            if len(ohlcv) < 50:
                return None
            
            # Calculate regime indicators
            returns = ohlcv['close'].pct_change().tail(20)
            volatility = returns.std()
            trend = (ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-20]) / ohlcv['close'].iloc[-20]
            
            # VIX level if available
            vix_level = market_data.get('VIX', {}).get('current_price', 20)
            
            # Determine regime
            if volatility > 0.025:
                if abs(trend) > 0.05:
                    regime = 'high_vol_trending'
                else:
                    regime = 'high_vol_choppy'
            elif abs(trend) > 0.03:
                regime = 'low_vol_trending'
            else:
                regime = 'low_vol_sideways'
            
            # Check if regime changed
            if len(self.pattern_memory['market_regime_history']) > 0:
                last_regime = self.pattern_memory['market_regime_history'][-1]['regime']
                if regime != last_regime:
                    regime_change = {
                        'timestamp': datetime.now(),
                        'from_regime': last_regime,
                        'to_regime': regime,
                        'volatility': volatility,
                        'trend': trend,
                        'vix': vix_level
                    }
                    
                    logger.info(f"🔄 Market regime change detected: {last_regime} → {regime}")
                    self._store_observation('regime_change', regime_change)
                    return regime_change
            
            # Store current regime
            self.pattern_memory['market_regime_history'].append({
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'vix': vix_level
            })
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
        
        return None
    
    def _check_prediction_outcomes(self):
        """Check outcomes of pending predictions"""
        try:
            pending = self.prediction_tracker.get_pending_predictions()
            
            for pred in pending:
                # Parse prediction time and check if outcome time has passed
                pred_time = datetime.fromisoformat(pred['timestamp'])
                timeframe_minutes = self.prediction_intervals.get(
                    pred['prediction_type'], {}
                ).get(pred['timeframe'], 60)  # Default 1 hour
                
                outcome_time = pred_time + timedelta(minutes=timeframe_minutes)
                
                if datetime.now() >= outcome_time:
                    # Fetch current price and update outcome
                    try:
                        current_data = fetch_stock_data(
                            pred['symbol'],
                            start=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                            end=datetime.now().strftime('%Y-%m-%d')
                        )
                        
                        if current_data is not None and len(current_data) > 0:
                            outcome_price = current_data.iloc[-1]['close']
                            self.prediction_tracker.update_outcome(pred['id'], outcome_price)
                            
                            # Calculate accuracy
                            actual_return = (outcome_price - pred['current_price']) / pred['current_price']
                            predicted_return = (pred['predicted_price'] - pred['current_price']) / pred['current_price']
                            
                            # Store accuracy
                            self.session_stats['accuracy_by_timeframe'][pred['timeframe']].append({
                                'predicted': predicted_return,
                                'actual': actual_return,
                                'error': abs(actual_return - predicted_return)
                            })
                            
                            if abs(actual_return - predicted_return) < 0.01:  # Within 1%
                                logger.info(f"✅ Accurate {pred['timeframe']} prediction for {pred['symbol']}")
                            
                    except Exception as e:
                        logger.error(f"Error checking outcome for prediction {pred['id']}: {e}")
            
        except Exception as e:
            logger.error(f"Error checking prediction outcomes: {e}")
    
    def _store_observation(self, obs_type: str, data: Dict[str, Any], 
                          symbol: str = 'MARKET', confidence: float = 1.0):
        """Store market observation in knowledge database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_observations 
                (timestamp, observation_type, symbol, data, confidence, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                obs_type,
                symbol,
                json.dumps(data, default=str),
                confidence,
                f"Auto-generated observation during daily monitoring"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing observation: {e}")
    
    def _end_of_day_summary(self):
        """Generate end-of-day summary and insights"""
        logger.info("📊 Generating end-of-day summary...")
        
        try:
            # Calculate session statistics
            end_time = datetime.now()
            session_duration = end_time - self.session_stats['start_time']
            
            # Prediction accuracy summary
            accuracy_summary = {}
            for timeframe, results in self.session_stats['accuracy_by_timeframe'].items():
                if results:
                    avg_error = np.mean([r['error'] for r in results])
                    accuracy_rate = sum(1 for r in results if r['error'] < 0.02) / len(results)
                    
                    accuracy_summary[timeframe] = {
                        'predictions': len(results),
                        'avg_error': avg_error,
                        'accuracy_rate': accuracy_rate
                    }
            
            # Generate daily insights
            insights = []
            
            # Best performing timeframe
            if accuracy_summary:
                best_timeframe = max(accuracy_summary.keys(), 
                                   key=lambda k: accuracy_summary[k]['accuracy_rate'])
                insights.append(f"Most accurate timeframe today: {best_timeframe}")
            
            # Pattern insights
            total_patterns = sum(self.session_stats['patterns_detected'].values())
            if total_patterns > 0:
                most_common_pattern = max(self.session_stats['patterns_detected'].keys(),
                                        key=lambda k: self.session_stats['patterns_detected'][k])
                insights.append(f"Most detected pattern: {most_common_pattern}")
            
            # Summary report
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'session_duration_hours': session_duration.total_seconds() / 3600,
                'total_predictions': self.session_stats['predictions_made'],
                'accuracy_by_timeframe': accuracy_summary,
                'patterns_detected': dict(self.session_stats['patterns_detected']),
                'key_insights': insights
            }
            
            # Store summary
            self._store_daily_insights('end_of_day_summary', summary, insights)
            
            # Log summary
            logger.info("📈 END-OF-DAY SUMMARY")
            logger.info(f"Session Duration: {session_duration.total_seconds()/3600:.1f} hours")
            logger.info(f"Total Predictions: {self.session_stats['predictions_made']}")
            logger.info(f"Patterns Detected: {total_patterns}")
            
            for timeframe, stats in accuracy_summary.items():
                logger.info(f"{timeframe}: {stats['predictions']} predictions, "
                           f"{stats['accuracy_rate']:.1%} accuracy")
            
            # Stop monitoring
            self.monitoring_active = False
            
            # Clear daily schedule
            schedule.clear('daily_summary')
            
            logger.info("🌙 LAEF going to sleep until tomorrow's wake-up")
            
        except Exception as e:
            logger.error(f"Error in end-of-day summary: {e}")
    
    def _store_daily_insights(self, insight_type: str, summary: Dict, insights: List[str]):
        """Store daily insights in knowledge database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            for insight in insights:
                cursor.execute('''
                    INSERT INTO daily_insights 
                    (date, insight_type, content, confidence, supporting_data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().date(),
                    insight_type,
                    insight,
                    0.8,
                    json.dumps(summary, default=str)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing daily insights: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'scheduler_running': self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
            'current_session_stats': dict(self.session_stats),
            'watch_symbols': self.watch_symbols,
            'knowledge_db_entries': self._count_knowledge_entries(),
            'next_wakeup': 'Tomorrow at 8:00 AM EST'
        }
    
    def _count_knowledge_entries(self) -> Dict[str, int]:
        """Count entries in knowledge database"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            counts = {}
            for table in ['market_observations', 'pattern_observations', 'daily_insights']:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                counts[table] = cursor.fetchone()[0]
            
            conn.close()
            return counts
            
        except Exception as e:
            logger.error(f"Error counting knowledge entries: {e}")
            return {}
    
    def manual_start(self):
        """Manually start monitoring (for testing)"""
        logger.info("🔧 Manual start requested")
        self._daily_wakeup()
    
    def stop_monitoring(self):
        """Stop all monitoring and scheduling"""
        logger.info("🛑 Stopping LAEF monitoring system...")
        
        self.monitoring_active = False
        schedule.clear()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Monitoring system stopped")


def main():
    """Main function to start LAEF's daily monitoring system"""
    print("🤖 Starting LAEF Daily Market Monitor...")
    
    monitor = DailyMarketMonitor()
    
    try:
        # Start the daily scheduler
        monitor.start_daily_scheduler()
        
        print("\n" + "="*60)
        print("LAEF DAILY MARKET MONITOR ACTIVE")
        print("="*60)
        print("• LAEF will wake up every day at 8:00 AM EST")
        print("• Continuous market observation during trading hours")
        print("• No actual trading - pure learning and observation")
        print("• Real-time pattern recognition and prediction tracking")
        print("• Knowledge accumulation for improved decision making")
        print("\nPress Ctrl+C to stop the system")
        print("="*60)
        
        # Keep the main thread alive
        while True:
            time.sleep(300)  # Check every 5 minutes
            
            # Show status update
            status = monitor.get_monitoring_status()
            if status['monitoring_active']:
                stats = status['current_session_stats']
                print(f"📊 Monitoring active - Predictions: {stats['predictions_made']}, "
                      f"Patterns: {sum(stats['patterns_detected'].values())}")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down LAEF Daily Monitor...")
        monitor.stop_monitoring()
        print("System stopped. LAEF is now offline.")
    
    except Exception as e:
        print(f"\n❌ Error in main loop: {e}")
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
